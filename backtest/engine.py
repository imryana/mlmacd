"""
engine.py — Simulate full trade lifecycle using OHLC data.

For each historical signal generated during walk-forward cross-validation,
the engine:
  1. Determines the limit entry price  (close ± ATR × entry_buffer_atr)
  2. Tries to fill the limit order within entry_expiry_bars bars
  3. Once filled, simulates bar by bar until one of four exits fires:
       - Stop loss       (bar low/high breaches stop)
       - Take profit     (bar high/low reaches target)
       - Signal reversal (opposing signal appears in walk-forward results)
       - Time exit       (max hold bars elapsed)
  4. Tracks MAE (max adverse excursion) and MFE (max favourable excursion)
  5. Applies per-side commission and ATR-based slippage from config

Outputs
-------
trade_log : pd.DataFrame
    One row per simulated trade.  Columns:
    ticker, entry_date, exit_date, entry_price, exit_price,
    exit_type, pnl_pct, pnl_abs, hold_bars, mae_pct, mfe_pct,
    signal, confidence, regime, sector.

summary : dict
    Aggregate statistics (win_rate, total_trades, …).
"""

import logging
import pathlib

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

_ROOT = pathlib.Path(__file__).resolve().parents[1]

# ── Exit type labels ──────────────────────────────────────────────────────

EXIT_STOP    = "stop"
EXIT_TARGET  = "target"
EXIT_SIGNAL  = "signal_exit"
EXIT_TIME    = "time_exit"
EXIT_MISSED  = "missed"


# ── Path helpers ──────────────────────────────────────────────────────────


def _load_config() -> dict:
    """Load and return config.yaml as a dict."""
    with open(_ROOT / "config" / "config.yaml") as fh:
        return yaml.safe_load(fh)


def _processed_path(ticker: str) -> pathlib.Path:
    """Return the processed parquet path for *ticker*."""
    return _ROOT / "data" / "processed" / "equities" / f"{ticker}.parquet"


def _wf_results_path() -> pathlib.Path:
    """Return path to saved walk-forward results parquet."""
    return _ROOT / "models" / "saved" / "wf_results.parquet"


def _trade_log_path() -> pathlib.Path:
    """Return output path for trade log parquet."""
    return _ROOT / "models" / "saved" / "trade_log.parquet"


def _backtest_metrics_path() -> pathlib.Path:
    """Return output path for backtest metrics JSON."""
    return _ROOT / "models" / "saved" / "backtest_metrics.json"


# ── Single trade simulation ───────────────────────────────────────────────


def _simulate_trade(
    ticker: str,
    signal_date: pd.Timestamp,
    signal: int,
    confidence: float,
    regime: int,
    sector: str,
    ohlc: pd.DataFrame,
    future_signals: pd.Series,
    cfg: dict,
) -> dict:
    """
    Simulate one trade from signal generation to final exit.

    Parameters
    ----------
    ticker : str
    signal_date : pd.Timestamp
        Date the signal was generated (last bar used for features).
    signal : int
        1 for long, -1 for short.
    confidence : float
    regime : int
    sector : str
    ohlc : pd.DataFrame
        OHLC + ATR DataFrame for this ticker, indexed by date.
        Must contain columns: open, high, low, close, atr.
    future_signals : pd.Series
        Walk-forward ``signal`` values for this ticker on dates *after*
        signal_date, indexed by date.  Used for signal-reversal exit.
    cfg : dict
        Full config dict.

    Returns
    -------
    dict
        Trade record.  ``exit_type`` is ``EXIT_MISSED`` if limit
        was never filled.
    """
    s_sig  = cfg["signals"]
    s_bt   = cfg["backtest"]
    is_long = signal == 1

    commission     = s_bt["commission_per_trade"]
    slippage_frac  = s_bt["slippage_atr_fraction"]
    entry_buf_atr  = s_sig["entry_buffer_atr"]
    stop_mult      = s_sig["stop_multiplier"]
    rr_ratio       = s_sig["rr_ratio"]
    expiry_bars    = s_sig["entry_expiry_bars"]
    time_exit_bars = s_sig["time_exit_bars"]

    # Signal row from ohlc
    if signal_date not in ohlc.index:
        return _missed_record(ticker, signal_date, signal, confidence, regime, sector)

    sig_row    = ohlc.loc[signal_date]
    close_px   = float(sig_row["close"])
    atr        = float(sig_row["atr"])

    slippage   = atr * slippage_frac

    # Limit price
    limit_px   = close_px - atr * entry_buf_atr if is_long else close_px + atr * entry_buf_atr

    # Bars after signal date
    future_ohlc = ohlc[ohlc.index > signal_date]
    if future_ohlc.empty:
        return _missed_record(ticker, signal_date, signal, confidence, regime, sector)

    # ── Phase 1 : try to fill limit entry ────────────────────────────────
    entry_price  = None
    entry_date   = None
    bars_scanned = 0

    for bar_date, bar in future_ohlc.iterrows():
        if bars_scanned >= expiry_bars:
            break

        bar_low  = float(bar["low"])
        bar_high = float(bar["high"])

        if is_long and bar_low <= limit_px:
            # Long filled: add slippage (buy slightly above limit)
            entry_price = limit_px + slippage
            entry_date  = bar_date
            break
        elif not is_long and bar_high >= limit_px:
            # Short filled: subtract slippage (sell slightly below limit)
            entry_price = limit_px - slippage
            entry_date  = bar_date
            break

        bars_scanned += 1

    if entry_price is None:
        return _missed_record(ticker, signal_date, signal, confidence, regime, sector)

    # ── Compute stop and target from entry ────────────────────────────────
    risk_per_unit = atr * stop_mult
    stop_loss     = entry_price - risk_per_unit if is_long else entry_price + risk_per_unit
    take_profit   = (entry_price + risk_per_unit * rr_ratio
                     if is_long
                     else entry_price - risk_per_unit * rr_ratio)

    # ── Phase 2 : bar-by-bar exit simulation ─────────────────────────────
    post_entry_bars = future_ohlc[future_ohlc.index >= entry_date]

    exit_price = None
    exit_date  = None
    exit_type  = None
    hold_bars  = 0

    # Running MAE / MFE (in price units, then converted to pct at end)
    mae_price  = 0.0   # most adverse move (always >= 0)
    mfe_price  = 0.0   # most favourable move (always >= 0)

    for bar_date, bar in post_entry_bars.iterrows():
        bar_open  = float(bar["open"])
        bar_high  = float(bar["high"])
        bar_low   = float(bar["low"])
        bar_close = float(bar["close"])

        # Adverse / favourable excursion for this bar
        if is_long:
            adverse    = entry_price - bar_low    # how far below entry
            favourable = bar_high - entry_price   # how far above entry
        else:
            adverse    = bar_high - entry_price
            favourable = entry_price - bar_low

        mae_price = max(mae_price, adverse)
        mfe_price = max(mfe_price, favourable)

        # Check signal reversal BEFORE stop/target (intraday not observable)
        if bar_date != entry_date and bar_date in future_signals.index:
            rev_signal = future_signals.loc[bar_date]
            # Reversal if non-flat signal opposite to our position
            if (is_long and rev_signal == -1) or (not is_long and rev_signal == 1):
                exit_price = bar_close
                exit_date  = bar_date
                exit_type  = EXIT_SIGNAL
                break

        # Check time exit (BEFORE stop/target on that bar)
        if hold_bars >= time_exit_bars:
            exit_price = bar_close
            exit_date  = bar_date
            exit_type  = EXIT_TIME
            break

        # Check stop loss
        if is_long and bar_low <= stop_loss:
            exit_price = stop_loss - slippage   # gap-through slippage
            exit_date  = bar_date
            exit_type  = EXIT_STOP
            break
        elif not is_long and bar_high >= stop_loss:
            exit_price = stop_loss + slippage
            exit_date  = bar_date
            exit_type  = EXIT_STOP
            break

        # Check take profit
        if is_long and bar_high >= take_profit:
            exit_price = take_profit - slippage
            exit_date  = bar_date
            exit_type  = EXIT_TARGET
            break
        elif not is_long and bar_low <= take_profit:
            exit_price = take_profit + slippage
            exit_date  = bar_date
            exit_type  = EXIT_TARGET
            break

        hold_bars += 1

    # If we exhausted bars without a defined exit, close at last close
    if exit_price is None:
        last_bar   = post_entry_bars.iloc[-1]
        exit_price = float(last_bar["close"])
        exit_date  = post_entry_bars.index[-1]
        exit_type  = EXIT_TIME

    # ── Compute P&L ───────────────────────────────────────────────────────
    if is_long:
        gross_pnl_pct = (exit_price - entry_price) / entry_price
    else:
        gross_pnl_pct = (entry_price - exit_price) / entry_price

    # Commission both sides
    net_pnl_pct = gross_pnl_pct - 2 * commission
    net_pnl_abs = net_pnl_pct * cfg["backtest"]["initial_capital"] * cfg["portfolio"]["risk_per_trade"]

    mae_pct = (mae_price / entry_price) * 100 if entry_price else 0.0
    mfe_pct = (mfe_price / entry_price) * 100 if entry_price else 0.0

    return {
        "ticker":       ticker,
        "entry_date":   entry_date,
        "exit_date":    exit_date,
        "entry_price":  round(entry_price, 4),
        "exit_price":   round(exit_price, 4),
        "exit_type":    exit_type,
        "pnl_pct":      round(net_pnl_pct * 100, 4),   # as percentage
        "pnl_abs":      round(net_pnl_abs, 2),
        "hold_bars":    hold_bars,
        "mae_pct":      round(mae_pct, 4),
        "mfe_pct":      round(mfe_pct, 4),
        "signal":       signal,
        "confidence":   round(confidence, 4),
        "regime":       regime,
        "sector":       sector,
        "signal_date":  signal_date,
    }


def _missed_record(
    ticker: str,
    signal_date: pd.Timestamp,
    signal: int,
    confidence: float,
    regime: int,
    sector: str,
) -> dict:
    """Return a placeholder trade record for a missed limit entry."""
    return {
        "ticker":       ticker,
        "entry_date":   None,
        "exit_date":    None,
        "entry_price":  None,
        "exit_price":   None,
        "exit_type":    EXIT_MISSED,
        "pnl_pct":      0.0,
        "pnl_abs":      0.0,
        "hold_bars":    0,
        "mae_pct":      0.0,
        "mfe_pct":      0.0,
        "signal":       signal,
        "confidence":   round(confidence, 4),
        "regime":       regime,
        "sector":       sector,
        "signal_date":  signal_date,
    }


# ── Main backtest function ────────────────────────────────────────────────


def run_backtest(
    wf_results: pd.DataFrame,
    cfg: dict,
    ohlc_data: dict | None = None,
    processed_dir: pathlib.Path | None = None,
    universe: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Simulate all trades from walk-forward results and return a trade log.

    Parameters
    ----------
    wf_results : pd.DataFrame
        Walk-forward prediction DataFrame as saved by ``models.trainer``.
        Must contain columns: ticker, signal, confidence, regime.
        Indexed by trading date.
    cfg : dict
        Full config dict.
    ohlc_data : dict[str, pd.DataFrame], optional
        Pre-loaded mapping of {ticker: ohlc_df}.  If None, each ticker's
        processed parquet is read from disk (``processed_dir`` or the
        default ``data/processed/equities/`` directory).
    processed_dir : pathlib.Path, optional
        Override directory for processed parquet files.
    universe : pd.DataFrame, optional
        Universe with ``ticker`` and ``sector`` columns.  Used to enrich
        trade records with sector information.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        ``(trade_log, summary_dict)``
    """
    if wf_results.empty:
        log.info("Empty walk-forward results — no trades to simulate.")
        empty = pd.DataFrame(
            columns=[
                "ticker", "entry_date", "exit_date", "entry_price", "exit_price",
                "exit_type", "pnl_pct", "pnl_abs", "hold_bars", "mae_pct", "mfe_pct",
                "signal", "confidence", "regime", "sector", "signal_date",
            ]
        )
        return empty, {}

    # Build sector lookup
    sector_map: dict[str, str] = {}
    if universe is not None and "ticker" in universe.columns and "sector" in universe.columns:
        sector_map = dict(zip(universe["ticker"], universe["sector"]))

    # Only process non-flat signals
    active = wf_results[wf_results["signal"] != 0].copy()
    log.info("Running backtest on %d non-flat signals.", len(active))

    pdir = processed_dir or (_ROOT / "data" / "processed" / "equities")

    all_trades: list[dict] = []

    for ticker, ticker_wf in active.groupby("ticker"):
        # Load OHLC data
        if ohlc_data is not None and ticker in ohlc_data:
            ohlc = ohlc_data[ticker]
        else:
            path = pdir / f"{ticker}.parquet" if processed_dir else _processed_path(str(ticker))
            if not path.exists():
                log.warning("No processed file for %s — skipping.", ticker)
                continue
            try:
                ohlc = pd.read_parquet(path)
            except Exception as exc:
                log.warning("Could not load %s: %s", ticker, exc)
                continue

        # Ensure required columns exist
        required = {"open", "high", "low", "close", "atr"}
        if not required.issubset(ohlc.columns):
            log.warning("Missing OHLC columns for %s — skipping.", ticker)
            continue

        # All future signals for this ticker (for reversal detection)
        all_ticker_signals = wf_results[wf_results["ticker"] == ticker]["signal"]

        sector = sector_map.get(str(ticker), "")

        for signal_date, row in ticker_wf.iterrows():
            future_signals = all_ticker_signals[all_ticker_signals.index > signal_date]

            trade = _simulate_trade(
                ticker      = str(ticker),
                signal_date = signal_date,
                signal      = int(row["signal"]),
                confidence  = float(row["confidence"]),
                regime      = int(row["regime"]),
                sector      = sector,
                ohlc        = ohlc,
                future_signals = future_signals,
                cfg         = cfg,
            )
            all_trades.append(trade)

    if not all_trades:
        log.info("No trades were simulated.")
        empty = pd.DataFrame(
            columns=[
                "ticker", "entry_date", "exit_date", "entry_price", "exit_price",
                "exit_type", "pnl_pct", "pnl_abs", "hold_bars", "mae_pct", "mfe_pct",
                "signal", "confidence", "regime", "sector", "signal_date",
            ]
        )
        return empty, {}

    trade_log = pd.DataFrame(all_trades)

    # ── Quick summary ────────────────────────────────────────────────────
    filled = trade_log[trade_log["exit_type"] != EXIT_MISSED]
    n_total  = len(trade_log)
    n_filled = len(filled)
    n_wins   = int((filled["pnl_pct"] > 0).sum()) if n_filled else 0

    summary = {
        "total_signals":  n_total,
        "filled_trades":  n_filled,
        "missed_trades":  n_total - n_filled,
        "win_trades":     n_wins,
        "loss_trades":    n_filled - n_wins,
        "win_rate":       round(n_wins / n_filled, 4) if n_filled else 0.0,
        "mean_pnl_pct":   round(float(filled["pnl_pct"].mean()), 4) if n_filled else 0.0,
    }

    log.info(
        "Backtest complete: %d signals → %d filled (%d wins, %d losses).",
        n_total, n_filled, n_wins, n_filled - n_wins,
    )

    return trade_log, summary


# ── Entry point ───────────────────────────────────────────────────────────


def main():
    """Load walk-forward results, run backtest, and save trade log."""
    import json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    cfg = _load_config()
    wf_path = _wf_results_path()

    if not wf_path.exists():
        print(f"Walk-forward results not found at {wf_path}.")
        print("Run `python -m models.trainer` first.")
        return

    wf_results = pd.read_parquet(wf_path)
    log.info("Loaded walk-forward results: %d rows.", len(wf_results))

    trade_log, summary = run_backtest(wf_results, cfg)

    if trade_log.empty:
        print("No trades simulated.")
        return

    out_path = _trade_log_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trade_log.to_parquet(out_path, index=False)

    print(f"\nBacktest complete — {summary['filled_trades']} trades simulated.")
    print(f"  Win rate : {summary['win_rate']:.1%}")
    print(f"  Mean P&L : {summary['mean_pnl_pct']:.2f}%")
    print(f"  Saved to : {out_path}")

    # Compute and save full metrics
    from backtest.metrics import compute_metrics
    metrics = compute_metrics(trade_log, cfg)
    metrics_path = _backtest_metrics_path()
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    print(f"  Metrics  : {metrics_path}")


if __name__ == "__main__":
    main()
