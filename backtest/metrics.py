"""
metrics.py — Performance metrics for the backtested trade log.

Computes and returns a comprehensive set of statistics from the trade log
produced by ``backtest.engine.run_backtest()``:

  Core metrics
    total_return_pct       — cumulative P&L across all trades
    annualised_return_pct  — total return scaled to one year
    sharpe_ratio           — annualised Sharpe (rf = 0) on per-trade returns
    sortino_ratio          — annualised Sortino (downside only)
    max_drawdown_pct       — peak-to-trough equity drawdown
    max_drawdown_duration  — calendar days of longest drawdown period
    calmar_ratio           — annualised return / |max drawdown|

  Trade statistics
    total_trades           — filled trades (excludes missed entries)
    win_rate               — fraction with positive net P&L
    avg_win_pct            — mean P&L of winning trades
    avg_loss_pct           — mean P&L of losing trades (negative)
    win_loss_ratio         — |avg_win / avg_loss|
    profit_factor          — gross profit / |gross loss|
    avg_hold_bars          — average bars held per trade
    trades_per_year        — annualised trade frequency

  Exit breakdown
    exit_type_counts       — {exit_type: count}
    exit_type_win_rates    — {exit_type: win_rate}

  Direction breakdown
    long_win_rate, short_win_rate
    long_mean_pnl, short_mean_pnl

  Regime breakdown
    regime_win_rates       — {regime_label: win_rate}
    regime_mean_pnl        — {regime_label: mean_pnl_pct}

  Sector breakdown
    sector_win_rates       — {sector: win_rate}
    sector_mean_pnl        — {sector: mean_pnl_pct}

  Equity curve
    equity_curve           — list of (date_str, equity_value) for each trade
    monthly_returns        — {\"YYYY-MM\": monthly_pnl_pct} for heatmap

All results saved to models/saved/backtest_metrics.json.
Trade log saved to models/saved/trade_log.parquet.
"""

import json
import logging
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

_ROOT = pathlib.Path(__file__).resolve().parents[1]

_REGIME_NAMES = {0: "choppy", 1: "bull", 2: "bear"}


# ── Path helpers ──────────────────────────────────────────────────────────


def _load_config() -> dict:
    """Load and return config.yaml as a dict."""
    with open(_ROOT / "config" / "config.yaml") as fh:
        return yaml.safe_load(fh)


def _trade_log_path() -> pathlib.Path:
    """Return path for the saved trade log."""
    return _ROOT / "models" / "saved" / "trade_log.parquet"


def _backtest_metrics_path() -> pathlib.Path:
    """Return path for the saved backtest metrics JSON."""
    return _ROOT / "models" / "saved" / "backtest_metrics.json"


# ── Helper statistics ─────────────────────────────────────────────────────


def _sharpe(returns: np.ndarray, scale: float = 1.0) -> float:
    """
    Annualised Sharpe ratio (rf = 0).

    Parameters
    ----------
    returns : np.ndarray
        Per-trade or per-period returns.
    scale : float
        Annualisation factor (e.g. sqrt(252) for daily, sqrt(n_trades_yr)
        for per-trade).

    Returns
    -------
    float
        Sharpe ratio, or 0.0 if variance is zero.
    """
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * scale)


def _sortino(returns: np.ndarray, scale: float = 1.0) -> float:
    """
    Annualised Sortino ratio (rf = 0, uses downside deviation).

    Parameters
    ----------
    returns : np.ndarray
    scale : float

    Returns
    -------
    float
        Sortino ratio, or 0.0 if downside deviation is zero.
    """
    downside = returns[returns < 0]
    if len(downside) < 2 or downside.std() == 0:
        return 0.0
    return float(returns.mean() / downside.std() * scale)


def _max_drawdown(equity: np.ndarray) -> tuple[float, int]:
    """
    Maximum peak-to-trough drawdown of the equity curve.

    Parameters
    ----------
    equity : np.ndarray
        Cumulative equity values (must be > 0).

    Returns
    -------
    tuple[float, int]
        ``(max_drawdown_fraction, max_drawdown_duration_bars)``
        Drawdown fraction is expressed as a positive number (e.g. 0.15 = 15%).
    """
    peak         = np.maximum.accumulate(equity)
    drawdown     = (peak - equity) / np.where(peak == 0, 1, peak)
    max_dd       = float(drawdown.max())

    # Duration of the longest drawdown period (in bars)
    in_dd        = drawdown > 0
    max_duration = 0
    current      = 0
    for flag in in_dd:
        if flag:
            current += 1
            max_duration = max(max_duration, current)
        else:
            current = 0

    return max_dd, max_duration


def _annualised_return(total_return: float, n_days: float) -> float:
    """
    Convert a total return to an annualised figure.

    Parameters
    ----------
    total_return : float
        Total fractional return (e.g. 0.25 = 25%).
    n_days : float
        Number of calendar days over which the return was earned.

    Returns
    -------
    float
        Annualised return (fractional).
    """
    if n_days <= 0:
        return 0.0
    base = 1 + total_return
    if base <= 0:
        return -1.0   # total wipe-out or beyond
    return float(base ** (365 / n_days) - 1)


# ── Equity curve builder ──────────────────────────────────────────────────


def _build_equity_curve(
    filled: pd.DataFrame,
    initial_capital: float,
) -> tuple[list[dict], dict[str, float]]:
    """
    Build an equity curve and monthly return series from the trade log.

    Parameters
    ----------
    filled : pd.DataFrame
        Trade log restricted to filled trades, sorted by exit_date.
        Must contain columns: exit_date, pnl_pct.
    initial_capital : float

    Returns
    -------
    tuple[list[dict], dict[str, float]]
        ``(equity_curve, monthly_returns)``

        *equity_curve*: list of dicts with keys ``date`` (ISO str) and
        ``equity`` (float).

        *monthly_returns*: mapping of ``"YYYY-MM"`` → monthly P&L pct.
    """
    equity = initial_capital
    curve  = [{"date": None, "equity": equity}]

    monthly: dict[str, float] = {}

    for _, row in filled.iterrows():
        exit_dt = row["exit_date"]
        pnl_pct = float(row["pnl_pct"]) / 100  # convert % to fraction

        # Approximate position value = initial_capital (1% risk ≈ position sizing)
        equity   += equity * pnl_pct * 0.01  # scaled by risk_per_trade
        date_str  = str(exit_dt)[:10] if exit_dt is not None else "unknown"
        curve.append({"date": date_str, "equity": round(equity, 2)})

        # Monthly bucket
        month_key = date_str[:7]  # "YYYY-MM"
        monthly[month_key] = monthly.get(month_key, 0.0) + float(row["pnl_pct"])

    return curve, monthly


# ── Breakdown helpers ─────────────────────────────────────────────────────


def _group_metrics(filled: pd.DataFrame, group_col: str) -> tuple[dict, dict]:
    """
    Compute per-group win rate and mean P&L.

    Parameters
    ----------
    filled : pd.DataFrame
        Filled trades with columns: pnl_pct, and *group_col*.
    group_col : str

    Returns
    -------
    tuple[dict, dict]
        ``(win_rates, mean_pnl)`` both keyed by group label.
    """
    win_rates: dict[str, float] = {}
    mean_pnl:  dict[str, float] = {}

    for group_val, grp in filled.groupby(group_col):
        key = str(group_val)
        wins = (grp["pnl_pct"] > 0).sum()
        win_rates[key] = round(wins / len(grp), 4) if len(grp) else 0.0
        mean_pnl[key]  = round(float(grp["pnl_pct"].mean()), 4)

    return win_rates, mean_pnl


# ── Main metrics function ─────────────────────────────────────────────────


def compute_metrics(
    trade_log: pd.DataFrame,
    cfg: dict,
) -> dict[str, Any]:
    """
    Compute the full set of performance metrics from a trade log.

    Parameters
    ----------
    trade_log : pd.DataFrame
        Output of ``backtest.engine.run_backtest()``.
        Required columns: ticker, entry_date, exit_date, pnl_pct, pnl_abs,
        exit_type, hold_bars, mae_pct, mfe_pct, signal, regime, sector.
    cfg : dict
        Full config dict.

    Returns
    -------
    dict
        All performance metrics.  Safe to serialise to JSON.
    """
    initial_capital = cfg["backtest"]["initial_capital"]

    if trade_log.empty:
        log.info("Empty trade log — returning zero metrics.")
        return {"error": "No trades in log."}

    # Separate filled vs missed
    missed = trade_log[trade_log["exit_type"] == "missed"]
    filled = trade_log[trade_log["exit_type"] != "missed"].copy()

    n_total  = len(trade_log)
    n_filled = len(filled)
    n_missed = len(missed)

    if n_filled == 0:
        return {
            "total_signals": n_total,
            "filled_trades": 0,
            "missed_trades": n_missed,
            "error": "All entries were missed — no trades to evaluate.",
        }

    # Sort by exit date for equity curve
    filled_sorted = filled.copy()
    filled_sorted["_exit_dt"] = pd.to_datetime(filled_sorted["exit_date"], errors="coerce")
    filled_sorted = filled_sorted.sort_values("_exit_dt")

    pnl_arr = filled_sorted["pnl_pct"].values  # in percentage points

    # ── Core return metrics ──────────────────────────────────────────────
    total_return_pct = float(pnl_arr.sum())

    # Date span
    exit_dates = filled_sorted["_exit_dt"].dropna()
    if len(exit_dates) >= 2:
        span_days = (exit_dates.max() - exit_dates.min()).days
    else:
        span_days = 252   # fallback

    ann_return_pct = _annualised_return(total_return_pct / 100, span_days) * 100

    # ── Risk metrics ─────────────────────────────────────────────────────
    # Use per-trade returns in % for Sharpe/Sortino (not converted to fraction)
    # Scale factor: assume roughly n_per_year trades
    trades_per_year = (n_filled / max(span_days, 1)) * 252
    scale           = float(np.sqrt(max(trades_per_year, 1)))

    sharpe  = _sharpe(pnl_arr, scale)
    sortino = _sortino(pnl_arr, scale)

    # Equity curve for drawdown
    equity_values = np.cumsum(pnl_arr)  # cumulative % P&L
    equity_curve, monthly_returns = _build_equity_curve(filled_sorted, initial_capital)

    max_dd, max_dd_dur = _max_drawdown(equity_values + 100)   # shift to avoid zeros

    calmar = (ann_return_pct / (max_dd * 100)) if max_dd > 0 else 0.0

    # ── Win / loss statistics ────────────────────────────────────────────
    wins  = filled[filled["pnl_pct"] > 0]["pnl_pct"]
    losses = filled[filled["pnl_pct"] <= 0]["pnl_pct"]

    win_rate     = round(len(wins) / n_filled, 4)
    avg_win      = round(float(wins.mean()), 4)    if len(wins)   else 0.0
    avg_loss     = round(float(losses.mean()), 4)  if len(losses) else 0.0
    win_loss_rat = round(abs(avg_win / avg_loss), 4) if avg_loss != 0 else float("inf")
    profit_factor= round(wins.sum() / abs(losses.sum()), 4) if losses.sum() != 0 else float("inf")

    avg_hold     = round(float(filled["hold_bars"].mean()), 2)

    # ── Exit type breakdown ───────────────────────────────────────────────
    exit_counts: dict[str, int] = filled["exit_type"].value_counts().to_dict()
    exit_win_rates: dict[str, float] = {}
    for exit_type, grp in filled.groupby("exit_type"):
        wr = (grp["pnl_pct"] > 0).sum() / len(grp)
        exit_win_rates[str(exit_type)] = round(float(wr), 4)

    # ── Direction breakdown ───────────────────────────────────────────────
    longs  = filled[filled["signal"] == 1]
    shorts = filled[filled["signal"] == -1]
    long_wr  = round((longs["pnl_pct"] > 0).sum() / len(longs), 4)  if len(longs)  else 0.0
    short_wr = round((shorts["pnl_pct"] > 0).sum() / len(shorts), 4) if len(shorts) else 0.0
    long_mean  = round(float(longs["pnl_pct"].mean()), 4)  if len(longs)  else 0.0
    short_mean = round(float(shorts["pnl_pct"].mean()), 4) if len(shorts) else 0.0

    # ── Regime breakdown ──────────────────────────────────────────────────
    filled_regime = filled.copy()
    filled_regime["regime_label"] = filled_regime["regime"].map(
        lambda r: _REGIME_NAMES.get(int(r), str(r))
    )
    regime_wr, regime_mpnl = _group_metrics(filled_regime, "regime_label")

    # ── Sector breakdown ──────────────────────────────────────────────────
    sector_wr, sector_mpnl = _group_metrics(filled, "sector")

    # ── MAE / MFE averages ────────────────────────────────────────────────
    avg_mae = round(float(filled["mae_pct"].mean()), 4)
    avg_mfe = round(float(filled["mfe_pct"].mean()), 4)

    metrics: dict[str, Any] = {
        # Summary
        "total_signals":         n_total,
        "filled_trades":         n_filled,
        "missed_trades":         n_missed,
        "initial_capital":       initial_capital,
        # Return
        "total_return_pct":      round(total_return_pct, 4),
        "annualised_return_pct": round(ann_return_pct, 4),
        # Risk
        "sharpe_ratio":          round(sharpe, 4),
        "sortino_ratio":         round(sortino, 4),
        "max_drawdown_pct":      round(max_dd * 100, 4),
        "max_drawdown_duration": max_dd_dur,
        "calmar_ratio":          round(calmar, 4),
        # Trade stats
        "win_rate":              win_rate,
        "avg_win_pct":           avg_win,
        "avg_loss_pct":          avg_loss,
        "win_loss_ratio":        win_loss_rat,
        "profit_factor":         profit_factor,
        "avg_hold_bars":         avg_hold,
        "trades_per_year":       round(trades_per_year, 2),
        # MAE / MFE
        "avg_mae_pct":           avg_mae,
        "avg_mfe_pct":           avg_mfe,
        # Exit breakdown
        "exit_type_counts":      {str(k): int(v) for k, v in exit_counts.items()},
        "exit_type_win_rates":   exit_win_rates,
        # Direction
        "long_win_rate":         long_wr,
        "short_win_rate":        short_wr,
        "long_mean_pnl":         long_mean,
        "short_mean_pnl":        short_mean,
        # Regime
        "regime_win_rates":      regime_wr,
        "regime_mean_pnl":       regime_mpnl,
        # Sector
        "sector_win_rates":      sector_wr,
        "sector_mean_pnl":       sector_mpnl,
        # Equity curve & monthly returns (for Streamlit charts)
        "equity_curve":          equity_curve,
        "monthly_returns":       monthly_returns,
    }

    return metrics


# ── Entry point ───────────────────────────────────────────────────────────


def main():
    """Load saved trade log, compute metrics, and save to JSON."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    cfg      = _load_config()
    log_path = _trade_log_path()

    if not log_path.exists():
        print(f"Trade log not found at {log_path}.")
        print("Run `python -m backtest.engine` first.")
        return

    trade_log = pd.read_parquet(log_path)
    log.info("Loaded trade log: %d rows.", len(trade_log))

    metrics = compute_metrics(trade_log, cfg)

    out_path = _backtest_metrics_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)

    print(f"\n{'='*60}")
    print("BACKTEST METRICS")
    print(f"{'='*60}")
    for key in (
        "filled_trades", "win_rate", "avg_win_pct", "avg_loss_pct",
        "profit_factor", "sharpe_ratio", "sortino_ratio",
        "max_drawdown_pct", "calmar_ratio", "avg_hold_bars",
        "total_return_pct", "annualised_return_pct",
    ):
        print(f"  {key:<30} {metrics.get(key, 'n/a')}")
    print(f"\nFull metrics saved to: {out_path}")


if __name__ == "__main__":
    main()
