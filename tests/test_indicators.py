"""
Unit tests for features/indicators.py.

Uses synthetic OHLCV data — no network calls, no file I/O.
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from features.indicators import (
    FEATURE_COLUMNS,
    calculate_indicators,
    _ema,
    _rma,
)

# ── fixtures ──────────────────────────────────────────────────────────────

_CFG = {
    "indicators": {
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30,
        "adx_period": 14, "adx_min_trend": 20,
        "ema_fast": 50, "ema_slow": 200,
        "atr_period": 14,
        "bb_period": 20, "bb_std": 2,
        "obv_slope_period": 5,
        "volume_ma_period": 20,
        "realised_vol_period": 20,
        "autocorr_period": 20,
        "rolling_return_period": 60,
    }
}


def _make_ohlcv(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Create a synthetic trending OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    price = 100 + np.cumsum(rng.normal(0.1, 1.0, n))  # slight uptrend
    price = np.clip(price, 10, None)
    return pd.DataFrame(
        {
            "open":   price * (1 + rng.normal(0, 0.002, n)),
            "high":   price * (1 + rng.uniform(0.001, 0.01, n)),
            "low":    price * (1 - rng.uniform(0.001, 0.01, n)),
            "close":  price,
            "volume": rng.integers(500_000, 5_000_000, n).astype(float),
        },
        index=dates,
    )


# ── helper function tests ─────────────────────────────────────────────────


def test_ema_length():
    s = pd.Series(np.ones(100))
    result = _ema(s, 10)
    assert len(result) == 100


def test_ema_converges_to_constant():
    """EMA of a constant series should equal that constant."""
    s = pd.Series(np.full(200, 5.0))
    result = _ema(s, 10)
    assert abs(result.iloc[-1] - 5.0) < 1e-6


def test_rma_warmup():
    """RMA should produce NaN for the first (period-1) rows."""
    s = pd.Series(np.ones(50))
    result = _rma(s, 14)
    assert result.iloc[:13].isna().all()
    assert result.iloc[13:].notna().all()


# ── calculate_indicators tests ────────────────────────────────────────────


_MACRO_COLS = {
    "vix_level", "vix_pct_rank", "vix_high",
    "spy_above_200", "spy_rsi", "spy_return_20",
}


def test_output_has_all_feature_columns():
    """
    calculate_indicators must produce every non-macro column in FEATURE_COLUMNS.

    The 6 macro columns (vix_*, spy_*) are joined in by features.pipeline
    after downloading VIX/SPY data — they are NOT produced by calculate_indicators.
    """
    df = _make_ohlcv(500)
    result = calculate_indicators(df, _CFG, sector_code=3)
    missing = [
        c for c in FEATURE_COLUMNS
        if c not in result.columns and c not in _MACRO_COLS
    ]
    assert missing == [], f"Missing columns: {missing}"


def test_original_columns_preserved():
    """OHLCV columns should still be present in the output."""
    df = _make_ohlcv(500)
    result = calculate_indicators(df, _CFG)
    for col in ("open", "high", "low", "close", "volume"):
        assert col in result.columns


def test_output_length_matches_input():
    """Row count must be unchanged."""
    df = _make_ohlcv(500)
    result = calculate_indicators(df, _CFG)
    assert len(result) == len(df)


def test_input_not_mutated():
    """The original DataFrame must not be modified."""
    df = _make_ohlcv(300)
    cols_before = set(df.columns)
    calculate_indicators(df, _CFG)
    assert set(df.columns) == cols_before


def test_rsi_range():
    """RSI values (where not NaN) must be in [0, 100]."""
    df = _make_ohlcv(500)
    result = calculate_indicators(df, _CFG)
    valid = result["rsi"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_adx_non_negative():
    """ADX must be >= 0 everywhere it is defined."""
    df = _make_ohlcv(500)
    result = calculate_indicators(df, _CFG)
    valid = result["adx"].dropna()
    assert (valid >= 0).all()


def test_binary_columns_are_0_or_1():
    """Binary indicator columns must contain only 0 and 1."""
    df = _make_ohlcv(500)
    result = calculate_indicators(df, _CFG)
    binary_cols = [
        "rsi_overbought", "rsi_oversold",
        "adx_trending", "golden_cross", "death_cross", "above_ema_slow",
        "volume_confirmation",
        "macd_above_zero", "macd_weekly_bull",
    ]
    for col in binary_cols:
        unique = set(result[col].dropna().unique())
        assert unique <= {0, 1}, f"{col} has values outside {{0,1}}: {unique}"


def test_sector_code_propagated():
    """The sector_code parameter must appear in the sector column."""
    df = _make_ohlcv(300)
    result = calculate_indicators(df, _CFG, sector_code=7)
    assert (result["sector"] == 7).all()


def test_macd_crossover_values():
    """macd_crossover must only contain {-1, 0, 1}."""
    df = _make_ohlcv(500)
    result = calculate_indicators(df, _CFG)
    unique = set(result["macd_crossover"].dropna().unique())
    assert unique <= {-1, 0, 1}


def test_volume_ratio_positive():
    """volume_ratio must be positive where defined."""
    df = _make_ohlcv(500)
    result = calculate_indicators(df, _CFG)
    valid = result["volume_ratio"].dropna()
    assert (valid > 0).all()


def test_macd_zero_cross_values():
    """macd_zero_cross must only contain {-1, 0, 1}."""
    df = _make_ohlcv(500)
    result = calculate_indicators(df, _CFG)
    unique = set(result["macd_zero_cross"].dropna().unique())
    assert unique <= {-1, 0, 1}


def test_macd_bars_since_cross_non_negative():
    """macd_bars_since_cross must be >= 0 everywhere it is defined."""
    df = _make_ohlcv(500)
    result = calculate_indicators(df, _CFG)
    valid = result["macd_bars_since_cross"].dropna()
    assert (valid >= 0).all()


def test_macd_cross_strength_non_negative():
    """macd_cross_strength (absolute gap) must be >= 0."""
    df = _make_ohlcv(500)
    result = calculate_indicators(df, _CFG)
    valid = result["macd_cross_strength"].dropna()
    assert (valid >= 0).all()


def test_warmup_nans_at_start():
    """The very first row should have NaNs in slow-EMA derived features."""
    df = _make_ohlcv(500)
    result = calculate_indicators(df, _CFG)
    # ema_slow (period 200) cannot be defined in first 199 rows
    assert result["ema_slow"].iloc[:199].isna().all()
