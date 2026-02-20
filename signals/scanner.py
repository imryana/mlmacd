"""
scanner.py — Run inference on every S&P 500 ticker and return ranked signals.

For each ticker with a processed feature file:
  1. Load features from data/processed/equities/{ticker}.parquet
  2. Predict per-bar regime with the fitted HMM (full sequence)
  3. Get the most recent bar's feature vector
  4. Run XGBoost predict_signal()
  5. Apply three filters:
       - confidence  < config.model.confidence_threshold   → skip
       - ADX         < config.indicators.adx_min_trend     → skip
       - regime == choppy AND signal != flat                → skip
  6. Collect and rank passing signals by confidence (descending)

Output saved to data/processed/latest_scan.parquet.
"""

import logging
import pathlib

import pandas as pd
import yaml

from features.indicators import FEATURE_COLUMNS
from models.trainer import _get_feature_columns
from models.xgboost_model import XGBoostModel
from regime.hmm_detector import CHOPPY, HMMDetector

log = logging.getLogger(__name__)

_ROOT = pathlib.Path(__file__).resolve().parents[1]

# ── Path helpers ──────────────────────────────────────────────────────────


def _load_config() -> dict:
    """Load and return config.yaml as a dict."""
    with open(_ROOT / "config" / "config.yaml") as fh:
        return yaml.safe_load(fh)


def _processed_path(ticker: str) -> pathlib.Path:
    """Return the processed parquet path for *ticker*."""
    return _ROOT / "data" / "processed" / "equities" / f"{ticker}.parquet"


def _universe_path() -> pathlib.Path:
    return _ROOT / "data" / "universe" / "sp500_tickers.csv"


def _scan_output_path() -> pathlib.Path:
    return _ROOT / "data" / "processed" / "latest_scan.parquet"


def _final_model_path() -> pathlib.Path:
    return _ROOT / "models" / "saved" / "xgboost_final.pkl"


# ── Feature column helper ─────────────────────────────────────────────────

#: All feature columns fed to XGBoost (indicators + regime)
_XGB_FEATURE_COLS = FEATURE_COLUMNS + ["regime"]


# ── Signal name helper ────────────────────────────────────────────────────

_SIGNAL_NAMES = {1: "LONG", -1: "SHORT", 0: "FLAT"}
_REGIME_NAMES = {0: "choppy", 1: "bull", 2: "bear"}


# ── Core scan function ────────────────────────────────────────────────────


def run_scan(
    cfg: dict,
    xgb_model: XGBoostModel | None = None,
    detector: HMMDetector | None = None,
    universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Scan all tickers with available processed data and return ranked signals.

    Parameters
    ----------
    cfg : dict
        Full config dict from config.yaml.
    xgb_model : XGBoostModel, optional
        Pre-loaded model (avoids repeated disk reads in batch jobs).
    detector : HMMDetector, optional
        Pre-loaded HMM detector.
    universe : pd.DataFrame, optional
        Universe DataFrame with ``ticker``, ``company``, ``sector`` columns.
        If None, loaded from disk.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, company, sector, signal, signal_name, confidence,
                 regime, regime_name, adx, rsi, macd_histogram, price,
                 atr, date.
        Sorted by confidence descending.  Empty if no signals pass filters.
    """
    conf_thresh = cfg["model"]["confidence_threshold"]
    adx_thresh  = cfg["indicators"]["adx_min_trend"]

    # ── Load models ───────────────────────────────────────────────────────
    if xgb_model is None:
        model_path = _final_model_path()
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. "
                "Run `python -m models.trainer` first."
            )
        xgb_model = XGBoostModel.load(model_path, cfg)
        log.info("XGBoost model loaded.")

    if detector is None:
        detector = HMMDetector.load(cfg)
        log.info("HMM detector loaded.")

    # ── Load universe ─────────────────────────────────────────────────────
    if universe is None:
        universe = pd.read_csv(_universe_path())

    ticker_info: dict[str, dict] = {
        row["ticker"]: {"company": row["company"], "sector": row["sector"]}
        for _, row in universe.iterrows()
    }

    # ── Per-ticker inference ──────────────────────────────────────────────
    signals: list[dict] = []
    skipped_no_file  = 0
    skipped_filter   = 0
    processed_count  = 0

    for ticker in universe["ticker"].tolist():
        path = _processed_path(ticker)
        if not path.exists():
            skipped_no_file += 1
            continue

        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            log.warning("Could not load %s: %s", ticker, exc)
            continue

        if len(df) < 10:
            continue

        # Predict regime on the full sequence for this ticker
        regime_series = detector.predict_regime(df)
        df = df.copy()
        df["regime"] = regime_series

        # Latest bar
        latest    = df.iloc[-1]
        feat_row  = latest[_XGB_FEATURE_COLS].to_frame().T

        sig_arr, conf_arr = xgb_model.predict_signal(feat_row)
        signal     = int(sig_arr[0])
        confidence = float(conf_arr[0])
        regime     = int(latest["regime"])

        processed_count += 1

        # ── Filters ───────────────────────────────────────────────────────
        if confidence < conf_thresh:
            skipped_filter += 1
            continue
        if float(latest["adx"]) < adx_thresh:
            skipped_filter += 1
            continue
        if regime == CHOPPY and signal != 0:
            skipped_filter += 1
            continue
        if signal == 0:   # flat signals are not actionable trade alerts
            continue

        info = ticker_info.get(ticker, {"company": "", "sector": ""})
        signals.append(
            {
                "ticker":         ticker,
                "company":        info["company"],
                "sector":         info["sector"],
                "signal":         signal,
                "signal_name":    _SIGNAL_NAMES[signal],
                "confidence":     round(confidence, 4),
                "regime":         regime,
                "regime_name":    _REGIME_NAMES[regime],
                "adx":            round(float(latest["adx"]),           2),
                "rsi":            round(float(latest["rsi"]),           2),
                "macd_histogram": round(float(latest["macd_histogram"]),4),
                "price":          round(float(latest["close"]),         4),
                "atr":            round(float(latest["atr"]),           4),
                "date":           df.index[-1].date().isoformat(),
            }
        )

    log.info(
        "Scan complete: %d tickers processed, %d signals after filters "
        "(%d skipped no-file, %d skipped filters).",
        processed_count, len(signals), skipped_no_file, skipped_filter,
    )

    if not signals:
        return pd.DataFrame(
            columns=[
                "ticker", "company", "sector", "signal", "signal_name",
                "confidence", "regime", "regime_name", "adx", "rsi",
                "macd_histogram", "price", "atr", "date",
            ]
        )

    result = (
        pd.DataFrame(signals)
        .sort_values("confidence", ascending=False)
        .reset_index(drop=True)
    )
    result.index += 1   # 1-based rank
    result.index.name = "rank"
    return result


# ── Entry point ───────────────────────────────────────────────────────────


def main():
    """Run the full scanner, print top 20 signals, save results."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    cfg     = _load_config()
    results = run_scan(cfg)

    if results.empty:
        print("\nNo signals passed all filters.")
    else:
        out_path = _scan_output_path()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_parquet(out_path)

        print(f"\nScan Results — {len(results)} signal(s) found")
        print("=" * 70)
        display_cols = [
            "ticker", "signal_name", "confidence", "regime_name",
            "adx", "rsi", "price",
        ]
        print(results[display_cols].head(20).to_string())
        print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
