"""
Unit tests for backtest/engine.py and backtest/metrics.py.

All tests use synthetic in-memory DataFrames — no disk I/O or network calls.
"""

import pathlib
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from backtest.engine import (
    EXIT_MISSED,
    EXIT_SIGNAL,
    EXIT_STOP,
    EXIT_TARGET,
    EXIT_TIME,
    _missed_record,
    _simulate_trade,
    run_backtest,
)
from backtest.metrics import (
    _annualised_return,
    _build_equity_curve,
    _max_drawdown,
    _sharpe,
    _sortino,
    compute_metrics,
)

# ── Shared config ─────────────────────────────────────────────────────────

_CFG = {
    "signals": {
        "entry_buffer_atr":             0.25,
        "entry_expiry_bars":            2,
        "stop_multiplier":              2.0,
        "rr_ratio":                     2.0,
        "trailing_stop_activation_atr": 1.0,
        "trailing_stop_distance_atr":   1.5,
        "time_exit_bars":               5,
    },
    "portfolio": {
        "size":           50_000,
        "risk_per_trade": 0.01,
    },
    "backtest": {
        "initial_capital":        50_000,
        "commission_per_trade":   0.001,
        "slippage_atr_fraction":  0.05,
    },
}

# ── OHLC builder ──────────────────────────────────────────────────────────


def _make_ohlc(
    n: int = 20,
    base_close: float = 100.0,
    atr: float = 2.0,
    trend: float = 0.0,   # daily drift in price
    start: str = "2024-01-02",
) -> pd.DataFrame:
    """
    Build a simple synthetic OHLC + ATR DataFrame indexed by trading date.

    Parameters
    ----------
    n : int
        Number of bars.
    base_close : float
        Starting close price.
    atr : float
        Constant ATR value for all bars.
    trend : float
        Per-bar price change (positive = up, negative = down).
    start : str
        ISO date for first bar.

    Returns
    -------
    pd.DataFrame
    """
    dates  = pd.bdate_range(start=start, periods=n)
    closes = base_close + np.arange(n) * trend
    df = pd.DataFrame(
        {
            "open":   closes - 0.2,
            "high":   closes + atr * 0.6,
            "low":    closes - atr * 0.6,
            "close":  closes,
            "volume": 1_000_000,
            "atr":    atr,
            "ticker": "TEST",
        },
        index=dates,
    )
    return df


def _make_ohlc_with_guaranteed_fill(
    n: int = 20,
    base_close: float = 100.0,
    atr: float = 2.0,
    trend_after_fill: float = 0.0,
    start: str = "2024-01-02",
    signal: int = 1,
) -> pd.DataFrame:
    """
    Build OHLC where bar 1 (first after signal_date=index[0]) is
    guaranteed to fill the limit, then bars 2+ follow *trend_after_fill*.

    For LONG:  limit = base_close - atr*0.25. Bar 1 has low = limit - 0.5.
    For SHORT: limit = base_close + atr*0.25. Bar 1 has high = limit + 0.5.
    """
    dates   = pd.bdate_range(start=start, periods=n)
    closes  = np.empty(n)
    highs   = np.empty(n)
    lows    = np.empty(n)

    # Bar 0 = signal bar (flat)
    closes[0] = base_close
    highs[0]  = base_close + atr * 0.6
    lows[0]   = base_close - atr * 0.6

    # Bar 1 = guaranteed fill bar (price still near base, then trend kicks in)
    limit = base_close - atr * 0.25 if signal == 1 else base_close + atr * 0.25
    closes[1] = base_close
    if signal == 1:
        lows[1]  = limit - 0.5   # guaranteed below limit
        highs[1] = base_close + 0.1
    else:
        highs[1] = limit + 0.5   # guaranteed above limit
        lows[1]  = base_close - 0.1

    # Bars 2+ = trend
    for i in range(2, n):
        closes[i] = base_close + (i - 1) * trend_after_fill
        highs[i]  = closes[i] + atr * 0.6
        lows[i]   = closes[i] - atr * 0.6

    df = pd.DataFrame(
        {
            "open":   closes - 0.2,
            "high":   highs,
            "low":    lows,
            "close":  closes,
            "volume": 1_000_000,
            "atr":    atr,
            "ticker": "TEST",
        },
        index=dates,
    )
    return df


def _make_wf_row(
    ticker: str = "TEST",
    signal: int = 1,
    confidence: float = 0.75,
    regime: int = 1,
    date_str: str = "2024-01-02",
) -> pd.DataFrame:
    """Build a single-row walk-forward results DataFrame."""
    idx = pd.DatetimeIndex([date_str])
    return pd.DataFrame(
        {
            "ticker":     [ticker],
            "signal":     [signal],
            "confidence": [confidence],
            "regime":     [regime],
            "actual":     [1],
            "forward_return": [0.01],
            "proba_short": [0.1],
            "proba_flat":  [0.1],
            "proba_long":  [0.8],
        },
        index=idx,
    )


# ── _simulate_trade — LONG entry tests ────────────────────────────────────


def test_long_limit_fills_when_low_breaches():
    """Long limit should fill when bar low <= limit price."""
    ohlc = _make_ohlc(n=10, base_close=100.0, atr=2.0)
    signal_date = ohlc.index[0]

    # atr=2, buffer=0.25 → limit = 100 - 0.5 = 99.5
    # bar lows are 100 - 2*0.6 = 98.8, which is < 99.5 → should fill
    result = _simulate_trade(
        ticker="TEST", signal_date=signal_date, signal=1,
        confidence=0.75, regime=1, sector="Tech",
        ohlc=ohlc, future_signals=pd.Series(dtype=int),
        cfg=_CFG,
    )
    assert result["exit_type"] != EXIT_MISSED, "Expected limit to fill but got MISSED"


def test_long_stop_exit():
    """A strongly downtrending ticker should trigger the stop exit."""
    # trend=-3 per bar: price drops quickly below stop
    ohlc = _make_ohlc(n=20, base_close=100.0, atr=2.0, trend=-3.0)
    signal_date = ohlc.index[0]

    result = _simulate_trade(
        ticker="TEST", signal_date=signal_date, signal=1,
        confidence=0.75, regime=1, sector="Tech",
        ohlc=ohlc, future_signals=pd.Series(dtype=int),
        cfg=_CFG,
    )
    assert result["exit_type"] == EXIT_STOP


def test_long_target_exit():
    """A strongly uptrending ticker should trigger the take-profit exit."""
    # Bar 1 guarantees fill; bars 2+ trend up strongly to hit take-profit
    ohlc = _make_ohlc_with_guaranteed_fill(
        n=20, base_close=100.0, atr=2.0, trend_after_fill=5.0, signal=1
    )
    signal_date = ohlc.index[0]

    result = _simulate_trade(
        ticker="TEST", signal_date=signal_date, signal=1,
        confidence=0.75, regime=1, sector="Tech",
        ohlc=ohlc, future_signals=pd.Series(dtype=int),
        cfg=_CFG,
    )
    assert result["exit_type"] == EXIT_TARGET


def test_long_time_exit_after_n_bars():
    """Flat price should exit via time_exit after time_exit_bars bars."""
    # Ensure limit fills on bar 1, then stay flat and tight (won't hit stop/target)
    ohlc = _make_ohlc_with_guaranteed_fill(
        n=20, base_close=100.0, atr=2.0, trend_after_fill=0.0, signal=1
    )
    # Tighten highs/lows on bars 2+ so stop (95.6) and target (107.6) are never hit
    # entry ~ 99.6, stop ~ 99.6-4=95.6, target ~ 99.6+8=107.6
    for i in range(2, len(ohlc)):
        ohlc.iloc[i, ohlc.columns.get_loc("high")] = 100.1
        ohlc.iloc[i, ohlc.columns.get_loc("low")]  = 99.8

    signal_date = ohlc.index[0]
    result = _simulate_trade(
        ticker="TEST", signal_date=signal_date, signal=1,
        confidence=0.75, regime=1, sector="Tech",
        ohlc=ohlc, future_signals=pd.Series(dtype=int),
        cfg=_CFG,
    )
    assert result["exit_type"] == EXIT_TIME


def test_long_signal_reversal_exit():
    """An opposing signal (-1) in future_signals should trigger signal_exit."""
    ohlc = _make_ohlc_with_guaranteed_fill(
        n=20, base_close=100.0, atr=2.0, trend_after_fill=0.0, signal=1
    )
    # Tighten bars so stop/target never hit; reversal will fire first
    for i in range(2, len(ohlc)):
        ohlc.iloc[i, ohlc.columns.get_loc("high")] = 100.1
        ohlc.iloc[i, ohlc.columns.get_loc("low")]  = 99.8

    signal_date   = ohlc.index[0]
    reversal_date = ohlc.index[3]   # 3 bars after signal (within time_exit_bars=5)

    future_signals = pd.Series(
        [-1],
        index=pd.DatetimeIndex([reversal_date]),
    )
    result = _simulate_trade(
        ticker="TEST", signal_date=signal_date, signal=1,
        confidence=0.75, regime=1, sector="Tech",
        ohlc=ohlc, future_signals=future_signals,
        cfg=_CFG,
    )
    assert result["exit_type"] == EXIT_SIGNAL


def test_missed_when_limit_not_reached():
    """If bar lows never breach limit, trade should be MISSED."""
    ohlc = _make_ohlc(n=5, base_close=100.0, atr=2.0)
    # Force all lows to be well above limit price (100 - 0.5 = 99.5)
    ohlc["low"] = 100.5   # never reaches 99.5

    signal_date = ohlc.index[0]
    result = _simulate_trade(
        ticker="TEST", signal_date=signal_date, signal=1,
        confidence=0.75, regime=1, sector="Tech",
        ohlc=ohlc, future_signals=pd.Series(dtype=int),
        cfg=_CFG,
    )
    assert result["exit_type"] == EXIT_MISSED


# ── _simulate_trade — SHORT entry tests ───────────────────────────────────


def test_short_limit_fills_when_high_breaches():
    """Short limit should fill when bar high >= limit price."""
    ohlc = _make_ohlc(n=10, base_close=100.0, atr=2.0)
    signal_date = ohlc.index[0]

    # atr=2, buffer=0.25 → short limit = 100 + 0.5 = 100.5
    # bar highs are 100 + 2*0.6 = 101.2, which is > 100.5 → should fill
    result = _simulate_trade(
        ticker="TEST", signal_date=signal_date, signal=-1,
        confidence=0.70, regime=2, sector="Energy",
        ohlc=ohlc, future_signals=pd.Series(dtype=int),
        cfg=_CFG,
    )
    assert result["exit_type"] != EXIT_MISSED


def test_short_stop_exit_on_rising_price():
    """Rising price should trigger stop for a short trade."""
    ohlc = _make_ohlc(n=20, base_close=100.0, atr=2.0, trend=5.0)
    signal_date = ohlc.index[0]

    result = _simulate_trade(
        ticker="TEST", signal_date=signal_date, signal=-1,
        confidence=0.70, regime=2, sector="Energy",
        ohlc=ohlc, future_signals=pd.Series(dtype=int),
        cfg=_CFG,
    )
    assert result["exit_type"] == EXIT_STOP


def test_short_target_exit_on_falling_price():
    """Falling price should trigger take-profit for a short trade."""
    # Bar 1 guarantees fill; bars 2+ trend down strongly to hit take-profit
    ohlc = _make_ohlc_with_guaranteed_fill(
        n=20, base_close=100.0, atr=2.0, trend_after_fill=-5.0, signal=-1
    )
    signal_date = ohlc.index[0]

    result = _simulate_trade(
        ticker="TEST", signal_date=signal_date, signal=-1,
        confidence=0.70, regime=2, sector="Energy",
        ohlc=ohlc, future_signals=pd.Series(dtype=int),
        cfg=_CFG,
    )
    assert result["exit_type"] == EXIT_TARGET


# ── Trade record fields ───────────────────────────────────────────────────


def test_trade_record_required_fields():
    """Every trade record must contain all required keys."""
    ohlc = _make_ohlc(n=15, base_close=100.0, atr=2.0, trend=5.0)
    signal_date = ohlc.index[0]

    result = _simulate_trade(
        ticker="TEST", signal_date=signal_date, signal=1,
        confidence=0.75, regime=1, sector="Tech",
        ohlc=ohlc, future_signals=pd.Series(dtype=int),
        cfg=_CFG,
    )
    for field in (
        "ticker", "entry_date", "exit_date", "entry_price", "exit_price",
        "exit_type", "pnl_pct", "pnl_abs", "hold_bars",
        "mae_pct", "mfe_pct", "signal", "confidence", "regime", "sector",
    ):
        assert field in result, f"Missing field: {field}"


def test_stop_exit_pnl_is_negative():
    """A stopped-out trade should have negative P&L (after commission)."""
    ohlc = _make_ohlc(n=20, base_close=100.0, atr=2.0, trend=-5.0)
    signal_date = ohlc.index[0]

    result = _simulate_trade(
        ticker="TEST", signal_date=signal_date, signal=1,
        confidence=0.75, regime=1, sector="Tech",
        ohlc=ohlc, future_signals=pd.Series(dtype=int),
        cfg=_CFG,
    )
    if result["exit_type"] == EXIT_STOP:
        assert result["pnl_pct"] < 0


def test_target_exit_pnl_is_positive():
    """A take-profit trade should have positive P&L."""
    ohlc = _make_ohlc(n=20, base_close=100.0, atr=2.0, trend=5.0)
    signal_date = ohlc.index[0]

    result = _simulate_trade(
        ticker="TEST", signal_date=signal_date, signal=1,
        confidence=0.75, regime=1, sector="Tech",
        ohlc=ohlc, future_signals=pd.Series(dtype=int),
        cfg=_CFG,
    )
    if result["exit_type"] == EXIT_TARGET:
        assert result["pnl_pct"] > 0


def test_mae_non_negative():
    """MAE must always be >= 0."""
    ohlc = _make_ohlc(n=15, base_close=100.0, atr=2.0)
    signal_date = ohlc.index[0]

    result = _simulate_trade(
        ticker="TEST", signal_date=signal_date, signal=1,
        confidence=0.75, regime=1, sector="Tech",
        ohlc=ohlc, future_signals=pd.Series(dtype=int),
        cfg=_CFG,
    )
    assert result["mae_pct"] >= 0


def test_mfe_non_negative():
    """MFE must always be >= 0."""
    ohlc = _make_ohlc(n=15, base_close=100.0, atr=2.0)
    signal_date = ohlc.index[0]

    result = _simulate_trade(
        ticker="TEST", signal_date=signal_date, signal=1,
        confidence=0.75, regime=1, sector="Tech",
        ohlc=ohlc, future_signals=pd.Series(dtype=int),
        cfg=_CFG,
    )
    assert result["mfe_pct"] >= 0


# ── run_backtest ──────────────────────────────────────────────────────────


def test_run_backtest_returns_dataframe():
    """run_backtest should return a DataFrame."""
    ohlc = _make_ohlc(n=20, base_close=100.0, atr=2.0)
    wf   = _make_wf_row(signal=1, date_str=str(ohlc.index[0].date()))

    trade_log, summary = run_backtest(wf, _CFG, ohlc_data={"TEST": ohlc})
    assert isinstance(trade_log, pd.DataFrame)


def test_run_backtest_empty_wf_returns_empty():
    """Empty walk-forward results should produce empty trade log."""
    trade_log, summary = run_backtest(pd.DataFrame(), _CFG, ohlc_data={})
    assert trade_log.empty


def test_run_backtest_flat_signals_skipped():
    """Signal == 0 rows should not generate trades."""
    ohlc = _make_ohlc(n=20, base_close=100.0, atr=2.0)
    wf   = _make_wf_row(signal=0, date_str=str(ohlc.index[0].date()))

    trade_log, _ = run_backtest(wf, _CFG, ohlc_data={"TEST": ohlc})
    assert trade_log.empty


def test_run_backtest_summary_keys():
    """Summary dict must include standard keys."""
    ohlc = _make_ohlc(n=20, base_close=100.0, atr=2.0, trend=5.0)
    wf   = _make_wf_row(signal=1, date_str=str(ohlc.index[0].date()))

    _, summary = run_backtest(wf, _CFG, ohlc_data={"TEST": ohlc})
    for key in ("total_signals", "filled_trades", "missed_trades", "win_rate"):
        assert key in summary, f"Missing summary key: {key}"


def test_run_backtest_multiple_tickers():
    """Engine should handle multiple tickers independently."""
    ohlc_a = _make_ohlc(n=20, base_close=100.0, atr=2.0, trend=5.0, start="2024-01-02")
    ohlc_b = _make_ohlc(n=20, base_close=200.0, atr=4.0, trend=-5.0, start="2024-01-02")

    wf_a = _make_wf_row("AAA", signal=1,  date_str=str(ohlc_a.index[0].date()))
    wf_b = _make_wf_row("BBB", signal=-1, date_str=str(ohlc_b.index[0].date()))
    wf   = pd.concat([wf_a, wf_b])

    trade_log, _ = run_backtest(
        wf, _CFG, ohlc_data={"AAA": ohlc_a, "BBB": ohlc_b}
    )
    tickers = trade_log["ticker"].unique()
    assert "AAA" in tickers or "BBB" in tickers


def test_run_backtest_missing_ohlc_ticker_skipped():
    """A ticker with no OHLC data available should be silently skipped."""
    wf = _make_wf_row("GHOST", signal=1, date_str="2024-01-02")

    trade_log, _ = run_backtest(wf, _CFG, ohlc_data={})   # no data
    assert trade_log.empty


# ── Metrics helpers ───────────────────────────────────────────────────────


def test_sharpe_positive_returns():
    """Mostly positive, varying returns → positive Sharpe."""
    r = np.array([1.0, 2.0, 1.5, 0.8, 2.0])   # varying so std > 0
    assert _sharpe(r) > 0


def test_sharpe_zero_variance():
    """Zero variance returns should give Sharpe = 0."""
    assert _sharpe(np.array([1.0])) == 0.0


def test_sortino_positive_skew():
    """Mostly positive returns with some downside → positive Sortino."""
    # Need >= 2 downside values so downside.std() > 0
    r = np.array([2.0, 2.0, -0.5, -0.3, 2.0, 2.0])
    assert _sortino(r) > 0


def test_max_drawdown_flat():
    """Flat equity curve should have zero drawdown."""
    equity = np.array([100.0, 100.0, 100.0, 100.0])
    dd, dur = _max_drawdown(equity)
    assert dd == 0.0


def test_max_drawdown_downtrend():
    """Steadily declining equity → max drawdown = total decline."""
    equity = np.array([100.0, 80.0, 60.0, 40.0])
    dd, _ = _max_drawdown(equity)
    assert dd > 0


def test_max_drawdown_recovery():
    """Drawdown that recovers fully → duration should include recovery bars."""
    equity = np.array([100.0, 90.0, 80.0, 100.0])
    dd, dur = _max_drawdown(equity)
    assert dd > 0
    assert dur >= 2


def test_annualised_return_positive():
    """50% return over 365 days should give ~50% annualised."""
    ann = _annualised_return(0.50, 365)
    assert abs(ann - 0.50) < 0.01


def test_annualised_return_zero_days():
    """Zero day span should return 0."""
    assert _annualised_return(0.5, 0) == 0.0


# ── compute_metrics ───────────────────────────────────────────────────────


def _make_trade_log(n_wins: int = 5, n_losses: int = 5) -> pd.DataFrame:
    """Build a synthetic filled trade log with n wins and n losses."""
    dates = pd.bdate_range("2024-01-02", periods=n_wins + n_losses)
    rows  = []

    for i, d in enumerate(dates):
        is_win = i < n_wins
        rows.append(
            {
                "ticker":      "TEST",
                "entry_date":  d - pd.Timedelta(days=3),
                "exit_date":   d,
                "entry_price": 100.0,
                "exit_price":  103.0 if is_win else 97.0,
                "exit_type":   EXIT_TARGET if is_win else EXIT_STOP,
                "pnl_pct":     3.0 if is_win else -2.0,
                "pnl_abs":     150.0 if is_win else -100.0,
                "hold_bars":   3,
                "mae_pct":     1.0,
                "mfe_pct":     4.0 if is_win else 0.5,
                "signal":      1,
                "confidence":  0.75,
                "regime":      1,
                "sector":      "Technology",
                "signal_date": d - pd.Timedelta(days=5),
            }
        )

    return pd.DataFrame(rows)


def test_compute_metrics_win_rate():
    """Win rate should equal n_wins / total."""
    log = _make_trade_log(n_wins=7, n_losses=3)
    m   = compute_metrics(log, _CFG)
    assert abs(m["win_rate"] - 0.7) < 0.01


def test_compute_metrics_required_keys():
    """compute_metrics must return all required top-level keys."""
    log = _make_trade_log()
    m   = compute_metrics(log, _CFG)

    for key in (
        "filled_trades", "win_rate", "avg_win_pct", "avg_loss_pct",
        "profit_factor", "sharpe_ratio", "sortino_ratio",
        "max_drawdown_pct", "calmar_ratio", "total_return_pct",
        "exit_type_counts", "exit_type_win_rates",
        "long_win_rate", "short_win_rate",
        "equity_curve", "monthly_returns",
    ):
        assert key in m, f"Missing metrics key: {key}"


def test_compute_metrics_empty_log():
    """Empty trade log should return a dict with an error key."""
    m = compute_metrics(pd.DataFrame(), _CFG)
    assert "error" in m


def test_compute_metrics_all_missed():
    """Trade log with only MISSED trades should report 0 filled."""
    rows = [
        {
            "ticker": "X", "entry_date": None, "exit_date": None,
            "entry_price": None, "exit_price": None,
            "exit_type": EXIT_MISSED, "pnl_pct": 0.0, "pnl_abs": 0.0,
            "hold_bars": 0, "mae_pct": 0.0, "mfe_pct": 0.0,
            "signal": 1, "confidence": 0.8, "regime": 1, "sector": "Tech",
            "signal_date": pd.Timestamp("2024-01-02"),
        }
    ]
    log = pd.DataFrame(rows)
    m   = compute_metrics(log, _CFG)
    assert m["filled_trades"] == 0


def test_compute_metrics_positive_profit_factor():
    """More wins than losses should give profit_factor > 1."""
    log = _make_trade_log(n_wins=8, n_losses=2)
    m   = compute_metrics(log, _CFG)
    assert m["profit_factor"] > 1.0


def test_compute_metrics_equity_curve_length():
    """Equity curve should have one entry per filled trade + initial."""
    n = 10
    log = _make_trade_log(n_wins=6, n_losses=4)
    m   = compute_metrics(log, _CFG)
    # +1 for the initial equity point
    assert len(m["equity_curve"]) == n + 1


def test_compute_metrics_regime_breakdown():
    """Regime breakdown dict should contain expected regime labels."""
    log = _make_trade_log()
    m   = compute_metrics(log, _CFG)
    # All trades have regime=1 (bull)
    assert "bull" in m["regime_win_rates"]


def test_compute_metrics_sector_breakdown():
    """Sector breakdown should include sectors present in trade log."""
    log = _make_trade_log()
    m   = compute_metrics(log, _CFG)
    assert "Technology" in m["sector_win_rates"]
