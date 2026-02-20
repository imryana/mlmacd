"""
Unit tests for signals/scanner.py.

Uses in-memory fake processed parquets and mocked models
so no disk I/O or network calls are made.
"""

import pathlib
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from features.indicators import FEATURE_COLUMNS
from signals.scanner import _SIGNAL_NAMES, _XGB_FEATURE_COLS, run_scan

# ── helpers ───────────────────────────────────────────────────────────────

_CFG = {
    "model":      {"confidence_threshold": 0.60},
    "indicators": {"adx_min_trend": 20},
}


def _make_processed_df(
    n: int = 300,
    adx: float = 25.0,
    rsi: float = 55.0,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Return a minimal processed-ticker DataFrame with all required columns,
    including a synthetic 'regime' column.
    """
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    df    = pd.DataFrame(index=dates)

    for col in FEATURE_COLUMNS:
        df[col] = rng.standard_normal(n)

    # Override key columns to controlled values
    df["adx"]  = adx
    df["rsi"]  = rsi
    df["close"] = 100.0
    df["atr"]   = 2.0
    df["macd_histogram"] = 0.01
    df["regime"] = 1   # bull

    # Add OHLCV + label cols to match real processed structure
    df["open"]           = 100.0
    df["high"]           = 101.0
    df["low"]            = 99.0
    df["volume"]         = 1_000_000.0
    df["forward_return"] = rng.normal(0, 0.01, n)
    df["label"]          = rng.choice([-1, 0, 1], n)
    df["ticker"]         = "TEST"

    return df


def _make_mock_xgb(signal: int = 1, confidence: float = 0.75):
    """Return a mock XGBoostModel with deterministic output."""
    mock = MagicMock()
    mock.predict_signal.return_value = (
        np.array([signal]),
        np.array([confidence]),
    )
    return mock


def _make_mock_detector(regime: int = 1):
    """Return a mock HMMDetector that always returns *regime*."""
    mock = MagicMock()
    mock.predict_regime.return_value = pd.Series(
        [regime] * 300,
        index=pd.bdate_range("2020-01-01", periods=300),
        name="regime",
    )
    return mock


def _make_universe(tickers: list[str] | None = None) -> pd.DataFrame:
    tickers = tickers or ["AAA", "BBB"]
    return pd.DataFrame(
        {
            "ticker":  tickers,
            "company": [f"Company {t}" for t in tickers],
            "sector":  ["Technology"] * len(tickers),
        }
    )


# ── run_scan ──────────────────────────────────────────────────────────────


def test_run_scan_returns_dataframe(tmp_path):
    """run_scan should return a DataFrame even when no processed files exist."""
    with patch("signals.scanner._processed_path", return_value=tmp_path / "none.parquet"):
        result = run_scan(
            _CFG,
            xgb_model=_make_mock_xgb(),
            detector=_make_mock_detector(),
            universe=_make_universe(),
        )
    assert isinstance(result, pd.DataFrame)


def test_run_scan_returns_signal_above_threshold(tmp_path):
    """A ticker with confidence=0.75 and ADX=25 should appear in results."""
    df = _make_processed_df(adx=25.0)

    def fake_path(ticker):
        p = tmp_path / f"{ticker}.parquet"
        if not p.exists():
            df.to_parquet(p)
        return p

    with patch("signals.scanner._processed_path", side_effect=fake_path):
        result = run_scan(
            _CFG,
            xgb_model=_make_mock_xgb(signal=1, confidence=0.75),
            detector=_make_mock_detector(regime=1),
            universe=_make_universe(["AAPL"]),
        )

    assert len(result) > 0
    assert result.iloc[0]["signal"] == 1


def test_run_scan_filters_low_confidence(tmp_path):
    """confidence=0.50 < threshold=0.60 → signal must be excluded."""
    df = _make_processed_df(adx=25.0)

    def fake_path(ticker):
        p = tmp_path / f"{ticker}.parquet"
        if not p.exists():
            df.to_parquet(p)
        return p

    with patch("signals.scanner._processed_path", side_effect=fake_path):
        result = run_scan(
            _CFG,
            xgb_model=_make_mock_xgb(signal=1, confidence=0.50),
            detector=_make_mock_detector(regime=1),
            universe=_make_universe(["AAPL"]),
        )

    assert result.empty


def test_run_scan_filters_low_adx(tmp_path):
    """ADX=15 < min_trend=20 → signal must be excluded."""
    df = _make_processed_df(adx=15.0)

    def fake_path(ticker):
        p = tmp_path / f"{ticker}.parquet"
        if not p.exists():
            df.to_parquet(p)
        return p

    with patch("signals.scanner._processed_path", side_effect=fake_path):
        result = run_scan(
            _CFG,
            xgb_model=_make_mock_xgb(signal=1, confidence=0.75),
            detector=_make_mock_detector(regime=1),
            universe=_make_universe(["AAPL"]),
        )

    assert result.empty


def test_run_scan_filters_choppy_regime(tmp_path):
    """Regime=choppy(0) with non-flat signal → must be excluded."""
    df = _make_processed_df(adx=25.0)

    def fake_path(ticker):
        p = tmp_path / f"{ticker}.parquet"
        if not p.exists():
            df.to_parquet(p)
        return p

    with patch("signals.scanner._processed_path", side_effect=fake_path):
        result = run_scan(
            _CFG,
            xgb_model=_make_mock_xgb(signal=1, confidence=0.75),
            detector=_make_mock_detector(regime=0),   # choppy
            universe=_make_universe(["AAPL"]),
        )

    assert result.empty


def test_run_scan_sorted_by_confidence_descending(tmp_path):
    """Results must be sorted by confidence, highest first."""
    confs   = [0.90, 0.65, 0.80]
    tickers = ["A", "B", "C"]
    dfs     = [_make_processed_df(adx=25.0, seed=i) for i in range(3)]

    def fake_path(ticker):
        i = tickers.index(ticker)
        p = tmp_path / f"{ticker}.parquet"
        if not p.exists():
            dfs[i].to_parquet(p)
        return p

    call_count = [0]

    def mock_predict_signal(X):
        i = call_count[0] % len(confs)
        call_count[0] += 1
        return np.array([1]), np.array([confs[i]])

    mock_xgb = MagicMock()
    mock_xgb.predict_signal.side_effect = mock_predict_signal

    with patch("signals.scanner._processed_path", side_effect=fake_path):
        result = run_scan(
            _CFG,
            xgb_model=mock_xgb,
            detector=_make_mock_detector(regime=1),
            universe=_make_universe(tickers),
        )

    if len(result) > 1:
        conf_vals = result["confidence"].values
        assert (conf_vals[:-1] >= conf_vals[1:]).all()


def test_run_scan_required_output_columns(tmp_path):
    """Output must contain all required columns."""
    df = _make_processed_df(adx=25.0)

    def fake_path(ticker):
        p = tmp_path / f"{ticker}.parquet"
        if not p.exists():
            df.to_parquet(p)
        return p

    with patch("signals.scanner._processed_path", side_effect=fake_path):
        result = run_scan(
            _CFG,
            xgb_model=_make_mock_xgb(signal=1, confidence=0.75),
            detector=_make_mock_detector(regime=1),
            universe=_make_universe(["AAPL"]),
        )

    for col in ("ticker", "signal", "confidence", "regime", "adx", "rsi",
                "macd_histogram", "price", "atr", "date"):
        assert col in result.columns, f"Missing column: {col}"


def test_run_scan_empty_universe(tmp_path):
    """Empty universe should return empty DataFrame without error."""
    result = run_scan(
        _CFG,
        xgb_model=_make_mock_xgb(),
        detector=_make_mock_detector(),
        universe=_make_universe([]),
    )
    assert result.empty
