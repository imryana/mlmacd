"""
trade_setup.py — Generate a complete trade card for each scanner signal.

For every signal row produced by scanner.run_scan(), this module computes:

  Entry
    limit_entry      — close ± ATR × entry_buffer_atr
    next_open_est    — current close (proxy for next open)
    entry_expiry     — today + entry_expiry_bars business days

  Exit
    stop_loss        — entry ± ATR × stop_multiplier
    take_profit      — entry ± risk × rr_ratio
    stop_pct / target_pct
    trailing_activation — entry ± ATR × trailing_stop_activation_atr
    trailing_distance   — ATR × trailing_stop_distance_atr
    time_exit           — today + time_exit_bars business days

  Position sizing (from config.portfolio)
    risk_amount      — portfolio_size × risk_per_trade
    position_units   — risk_amount / risk_per_unit
    position_value   — units × entry_price
    position_pct     — position_value / portfolio_size × 100

  SHAP top-5 drivers
    Per-bar SHAP values computed from the saved XGBoost model.

Full trade cards saved to data/processed/trade_cards.parquet.
"""

import logging
import pathlib
import warnings
from datetime import date

import numpy as np
import pandas as pd
import shap
import yaml

from features.indicators import FEATURE_COLUMNS
from models.xgboost_model import XGBoostModel
from regime.hmm_detector import HMMDetector
from signals.scanner import (
    _SIGNAL_NAMES,
    _final_model_path,
    _processed_path,
    _universe_path,
    run_scan,
)

log = logging.getLogger(__name__)

_ROOT = pathlib.Path(__file__).resolve().parents[1]

_XGB_FEATURE_COLS = FEATURE_COLUMNS + ["regime"]


# ── Path helpers ──────────────────────────────────────────────────────────


def _load_config() -> dict:
    """Load and return config.yaml as a dict."""
    with open(_ROOT / "config" / "config.yaml") as fh:
        return yaml.safe_load(fh)


def _trade_cards_path() -> pathlib.Path:
    """Return the output path for trade cards parquet."""
    return _ROOT / "data" / "processed" / "trade_cards.parquet"


# ── Business-day arithmetic ───────────────────────────────────────────────


def _add_bdays(ref_date: date, n: int) -> date:
    """
    Return *ref_date* + *n* business days.

    Parameters
    ----------
    ref_date : date
    n : int

    Returns
    -------
    date
    """
    ts = pd.Timestamp(ref_date) + pd.tseries.offsets.BDay(n)
    return ts.date()


# ── Per-bar SHAP ──────────────────────────────────────────────────────────


def _compute_shap_top5(
    model: XGBoostModel,
    X_row: pd.DataFrame,
) -> list[dict]:
    """
    Compute SHAP values for a single feature row and return the top 5
    features by absolute SHAP value (averaged across classes).

    Parameters
    ----------
    model : XGBoostModel
    X_row : pd.DataFrame
        Single-row feature DataFrame.

    Returns
    -------
    list[dict]
        Up to 5 dicts with keys ``feature``, ``shap_value``, ``direction``
        (``"positive"`` or ``"negative"``).
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer   = shap.TreeExplainer(model.model)
            shap_values = explainer.shap_values(X_row)

        # shap_values: list of (1, n_features) arrays or (1, n_features, n_classes)
        if isinstance(shap_values, list):
            mean_shap = np.mean([sv[0] for sv in shap_values], axis=0)
        else:
            mean_shap = shap_values[0].mean(axis=-1)   # average over classes

        features   = list(X_row.columns)
        top_idx    = np.argsort(np.abs(mean_shap))[::-1][:5]
        return [
            {
                "feature":    features[i],
                "shap_value": round(float(mean_shap[i]), 6),
                "direction":  "positive" if mean_shap[i] >= 0 else "negative",
            }
            for i in top_idx
        ]
    except Exception as exc:
        log.debug("SHAP computation failed: %s", exc)
        return []


# ── Single trade card ─────────────────────────────────────────────────────


def build_trade_card(
    signal_row: pd.Series,
    cfg: dict,
    xgb_model: XGBoostModel | None = None,
    detector: HMMDetector | None = None,
) -> dict:
    """
    Build a complete trade card dictionary for one scanner signal row.

    Parameters
    ----------
    signal_row : pd.Series
        One row from the DataFrame returned by ``scanner.run_scan()``.
        Must contain: ticker, signal, price, atr.
    cfg : dict
        Full config dict.
    xgb_model : XGBoostModel, optional
        Pre-loaded model for SHAP computation.
    detector : HMMDetector, optional
        Pre-loaded HMM (used to rehydrate regime for SHAP features).

    Returns
    -------
    dict
        Complete trade card with entry, exit, sizing, and SHAP fields.
    """
    ticker     = signal_row["ticker"]
    signal     = int(signal_row["signal"])
    price      = float(signal_row["price"])
    atr        = float(signal_row["atr"])
    confidence = float(signal_row["confidence"])

    s_entry    = cfg["signals"]
    s_port     = cfg["portfolio"]

    is_long = signal == 1

    # ── Entry ─────────────────────────────────────────────────────────────
    buf        = atr * s_entry["entry_buffer_atr"]
    limit_entry = price - buf if is_long else price + buf
    entry_price = limit_entry   # used for all downstream calcs
    today       = date.today()
    entry_expiry = _add_bdays(today, s_entry["entry_expiry_bars"])

    # ── Stop loss ─────────────────────────────────────────────────────────
    stop_dist  = atr * s_entry["stop_multiplier"]
    stop_loss  = entry_price - stop_dist if is_long else entry_price + stop_dist
    risk_per_unit = abs(entry_price - stop_loss)
    stop_pct   = (stop_loss - entry_price) / entry_price * 100

    # ── Take profit ───────────────────────────────────────────────────────
    reward        = risk_per_unit * s_entry["rr_ratio"]
    take_profit   = entry_price + reward if is_long else entry_price - reward
    target_pct    = (take_profit - entry_price) / entry_price * 100

    # ── Trailing stop ─────────────────────────────────────────────────────
    trail_act_dist = atr * s_entry["trailing_stop_activation_atr"]
    trail_activation = (
        entry_price + trail_act_dist if is_long else entry_price - trail_act_dist
    )
    trailing_distance = atr * s_entry["trailing_stop_distance_atr"]

    # ── Time exit ─────────────────────────────────────────────────────────
    time_exit = _add_bdays(today, s_entry["time_exit_bars"])

    # ── Position sizing ───────────────────────────────────────────────────
    portfolio_size = s_port["size"]
    risk_amount    = portfolio_size * s_port["risk_per_trade"]
    position_units = risk_amount / risk_per_unit if risk_per_unit > 0 else 0.0
    position_value = position_units * entry_price
    position_pct   = position_value / portfolio_size * 100 if portfolio_size > 0 else 0.0

    # ── SHAP top-5 ────────────────────────────────────────────────────────
    shap_drivers: list[dict] = []
    if xgb_model is not None and detector is not None:
        try:
            df = pd.read_parquet(_processed_path(ticker))
            df = df.copy()
            df["regime"] = detector.predict_regime(df)
            latest_feat  = df.iloc[-1][_XGB_FEATURE_COLS].to_frame().T
            shap_drivers = _compute_shap_top5(xgb_model, latest_feat)
        except Exception as exc:
            log.debug("SHAP for %s failed: %s", ticker, exc)

    return {
        # Identity
        "ticker":               ticker,
        "signal":               signal,
        "signal_name":          _SIGNAL_NAMES[signal],
        "confidence":           round(confidence, 4),
        "regime":               signal_row.get("regime", 0),
        "regime_name":          signal_row.get("regime_name", ""),
        "scan_date":            signal_row.get("date", today.isoformat()),
        # Market context
        "price":                round(price, 4),
        "atr":                  round(atr, 4),
        "adx":                  round(float(signal_row.get("adx", 0)), 2),
        "rsi":                  round(float(signal_row.get("rsi", 0)), 2),
        "macd_histogram":       round(float(signal_row.get("macd_histogram", 0)), 4),
        # Entry
        "next_open_estimate":   round(price, 4),
        "limit_entry":          round(limit_entry, 4),
        "entry_expiry_date":    entry_expiry.isoformat(),
        # Exit
        "stop_loss":            round(stop_loss, 4),
        "stop_pct":             round(stop_pct, 2),
        "take_profit":          round(take_profit, 4),
        "target_pct":           round(target_pct, 2),
        "rr_ratio":             round(abs(target_pct / stop_pct), 2) if stop_pct != 0 else 0.0,
        "trailing_activation":  round(trail_activation, 4),
        "trailing_distance":    round(trailing_distance, 4),
        "time_exit_date":       time_exit.isoformat(),
        # Position sizing
        "risk_amount":          round(risk_amount, 2),
        "position_units":       round(position_units, 4),
        "position_value":       round(position_value, 2),
        "position_pct":         round(position_pct, 2),
        # SHAP top-5 drivers (stored as JSON string for parquet compatibility)
        "shap_top5":            str(shap_drivers),
    }


# ── Batch trade cards ─────────────────────────────────────────────────────


def build_all_trade_cards(
    signals_df: pd.DataFrame,
    cfg: dict,
    xgb_model: XGBoostModel | None = None,
    detector: HMMDetector | None = None,
) -> pd.DataFrame:
    """
    Build trade cards for every row in *signals_df*.

    Parameters
    ----------
    signals_df : pd.DataFrame
        Output of ``scanner.run_scan()``.
    cfg : dict
    xgb_model : XGBoostModel, optional
    detector : HMMDetector, optional

    Returns
    -------
    pd.DataFrame
        One row per signal, all trade-card fields as columns.
    """
    if signals_df.empty:
        log.info("No signals — no trade cards generated.")
        return pd.DataFrame()

    cards = [
        build_trade_card(row, cfg, xgb_model=xgb_model, detector=detector)
        for _, row in signals_df.iterrows()
    ]
    return pd.DataFrame(cards)


# ── Entry point ───────────────────────────────────────────────────────────


def main():
    """Run scanner then build and save all trade cards."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    cfg      = _load_config()
    model    = XGBoostModel.load(_final_model_path(), cfg)
    detector = HMMDetector.load(cfg)
    universe = pd.read_csv(_universe_path())

    log.info("Running scanner...")
    signals = run_scan(cfg, xgb_model=model, detector=detector, universe=universe)

    if signals.empty:
        print("No signals passed filters — no trade cards generated.")
        return

    log.info("Building trade cards for %d signal(s)...", len(signals))
    cards_df = build_all_trade_cards(signals, cfg, xgb_model=model, detector=detector)

    out_path = _trade_cards_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cards_df.to_parquet(out_path, index=False)

    print(f"\n{len(cards_df)} trade card(s) generated.")
    print("=" * 70)
    display_cols = [
        "ticker", "signal_name", "confidence", "price",
        "limit_entry", "stop_loss", "take_profit",
        "stop_pct", "target_pct", "rr_ratio",
        "position_units", "position_value", "position_pct",
    ]
    print(cards_df[display_cols].to_string(index=False))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
