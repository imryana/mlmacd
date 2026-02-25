"""
scanner.py — Run inference on every S&P 500 ticker and return ranked signals.

For each ticker with a processed feature file:
  1. Load features from data/processed/equities/{ticker}.parquet
  2. Predict per-bar regime with the fitted HMM (full sequence)
  3. Get the most recent bar's feature vector
  4. Run long model and short model independently:
       long_conf  = P(return > +threshold)
       short_conf = P(return < -threshold)
  5. Apply signal logic:
       long_conf  > threshold → LONG
       short_conf > threshold → SHORT  (only if LONG not triggered)
       otherwise               → FLAT (skip)
  6. Apply three additional filters:
       - ADX         < config.indicators.adx_min_trend     → skip
       - regime == choppy AND signal != flat                → skip
  7. Collect and rank passing signals by confidence (descending)

Output saved to data/processed/latest_scan.parquet.
"""

import logging
import pathlib

import pandas as pd
import yaml

from features.indicators import FEATURE_COLUMNS
from features.macro import compute_macro_features
from models.trainer import _get_feature_columns
from models.xgboost_model import XGBoostBinaryModel
from regime.hmm_detector import BEAR, BULL, CHOPPY, HMMDetector

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


def _long_model_path() -> pathlib.Path:
    return _ROOT / "models" / "saved" / "long_model.pkl"


def _short_model_path() -> pathlib.Path:
    return _ROOT / "models" / "saved" / "short_model.pkl"


# ── Feature column helper ─────────────────────────────────────────────────

#: All feature columns fed to XGBoost (indicators + macro + regime + derived)
_XGB_FEATURE_COLS = FEATURE_COLUMNS + ["regime", "macd_regime_alignment"]


# ── Signal name helper ────────────────────────────────────────────────────

_SIGNAL_NAMES = {1: "LONG", -1: "SHORT", 0: "FLAT"}
_REGIME_NAMES = {0: "choppy", 1: "bull", 2: "bear"}


# ── Core scan function ────────────────────────────────────────────────────


def run_scan(
    cfg: dict,
    long_model: XGBoostBinaryModel | None = None,
    short_model: XGBoostBinaryModel | None = None,
    detector: HMMDetector | None = None,
    universe: pd.DataFrame | None = None,
    # Legacy parameter kept for backward compatibility in tests
    xgb_model=None,
) -> pd.DataFrame:
    """
    Scan all tickers with available processed data and return ranked signals.

    Parameters
    ----------
    cfg : dict
        Full config dict from config.yaml.
    long_model : XGBoostBinaryModel, optional
        Pre-loaded long-side binary model.
    short_model : XGBoostBinaryModel, optional
        Pre-loaded short-side binary model.
    detector : HMMDetector, optional
        Pre-loaded HMM detector.
    universe : pd.DataFrame, optional
        Universe DataFrame with ``ticker``, ``company``, ``sector`` columns.
        If None, loaded from disk.
    xgb_model : (deprecated)
        Legacy single-model argument; ignored if long_model is provided.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, company, sector, signal, signal_name, confidence,
                 regime, regime_name, adx, rsi, macd_histogram, price,
                 atr, date, long_conf, short_conf.
        Sorted by confidence descending.  Empty if no signals pass filters.
    """
    conf_thresh = cfg["model"]["confidence_threshold"]
    adx_thresh  = cfg["indicators"]["adx_min_trend"]

    # ── Load models ───────────────────────────────────────────────────────
    # Support legacy single-model callers (e.g. older tests)
    if long_model is None and xgb_model is not None:
        # Wrap the legacy model: treat its predict_signal as long-only
        long_model  = _LegacyModelAdapter(xgb_model)
        short_model = None

    if long_model is None:
        lpath = _long_model_path()
        if not lpath.exists():
            raise FileNotFoundError(
                f"Long model not found at {lpath}. "
                "Run `python -m models.trainer` first."
            )
        long_model = XGBoostBinaryModel.load(lpath, cfg)
        log.info("Long model loaded.")

    if short_model is None and xgb_model is None:
        spath = _short_model_path()
        if not spath.exists():
            raise FileNotFoundError(
                f"Short model not found at {spath}. "
                "Run `python -m models.trainer` first."
            )
        short_model = XGBoostBinaryModel.load(spath, cfg)
        log.info("Short model loaded.")

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

    # ── Load latest macro features (once, shared across all tickers) ─────────
    macro_df = compute_macro_features(cfg)
    latest_macro: dict = {}
    if not macro_df.empty:
        latest_macro = macro_df.iloc[-1].to_dict()
        log.info("Latest macro loaded (date=%s).", macro_df.index[-1].date())
    else:
        log.warning("Macro features unavailable — macro cols will be zeroed.")

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

        # Latest bar — inject macro features, then build feature row
        latest   = df.iloc[-1].copy()
        for col, val in latest_macro.items():
            latest[col] = val
        # Compute derived feature: MACD cross aligned with HMM regime
        crossover = latest.get("macd_crossover", 0)
        regime_val = latest.get("regime", -1)
        latest["macd_regime_alignment"] = int(
            (crossover == 1 and regime_val == BULL) or
            (crossover == -1 and regime_val == BEAR)
        )
        feat_cols = [c for c in _XGB_FEATURE_COLS if c in latest.index]
        feat_row  = latest[feat_cols].to_frame().T

        # ── Dual model inference ──────────────────────────────────────────
        long_conf  = float(long_model.predict_proba(feat_row)[0])
        if short_model is not None:
            short_conf = float(short_model.predict_proba(feat_row)[0])
        else:
            short_conf = 0.0

        if long_conf > conf_thresh:
            signal, confidence = 1, long_conf
        elif short_conf > conf_thresh:
            signal, confidence = -1, short_conf
        else:
            signal, confidence = 0, max(long_conf, short_conf)

        regime = int(latest["regime"])
        processed_count += 1

        # ── Filters ───────────────────────────────────────────────────────
        if signal == 0:
            continue
        if float(latest["adx"]) < adx_thresh:
            skipped_filter += 1
            continue
        if regime == CHOPPY and signal != 0:
            skipped_filter += 1
            continue
        if signal == -1 and regime != BEAR:
            skipped_filter += 1
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
                "long_conf":      round(long_conf,  4),
                "short_conf":     round(short_conf, 4),
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
                "long_conf", "short_conf",
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


# ── Legacy adapter ────────────────────────────────────────────────────────

class _LegacyModelAdapter:
    """Wraps the old XGBoostModel (multi-class) to behave like a binary long model."""

    def __init__(self, model):
        self._m = model

    def predict_proba(self, X):
        """Return P(long) from the multi-class model (column index 2 = class +1)."""
        import numpy as np
        proba = self._m.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 3:
            return proba[:, 2]          # P(long)
        return proba[:, -1]


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
