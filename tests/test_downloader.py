"""
Unit tests for ingestion/downloader.py.

Validation tests use synthetic DataFrames (no network).
Smoke tests make a real yfinance call for a short date range.
"""

import pathlib
import sys
from datetime import date

import pandas as pd
import pytest

# Allow running from the repo root without installation.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ingestion.downloader import download_ticker, validate_ticker

# ── helpers ───────────────────────────────────────────────────────────────

_CFG = {
    "data": {
        "start_date": "2020-01-01",
        "end_date": "today",
        "min_price": 5.0,
        "min_history_days": 504,
    }
}


def _make_ohlcv(n: int = 600, price: float = 50.0) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame with *n* consecutive business days."""
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {
            "open": price,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "volume": 1_000_000,
        },
        index=dates,
    )


# ── validate_ticker ───────────────────────────────────────────────────────


def test_validate_ok():
    """A clean 600-bar DataFrame should pass validation."""
    df = _make_ohlcv(600)
    ok, reason = validate_ticker(df, _CFG)
    assert ok, f"Expected pass but got: {reason}"


def test_validate_too_few_bars():
    """Fewer bars than min_history_days should fail."""
    df = _make_ohlcv(100)
    ok, reason = validate_ticker(df, _CFG)
    assert not ok
    assert "bars" in reason.lower()


def test_validate_price_below_min():
    """Prices below min_price should fail."""
    df = _make_ohlcv(600, price=2.0)
    ok, reason = validate_ticker(df, _CFG)
    assert not ok
    assert "min_price" in reason.lower() or "price" in reason.lower()


def test_validate_zero_volume():
    """A bar with zero volume should fail."""
    df = _make_ohlcv(600)
    df.loc[df.index[50], "volume"] = 0
    ok, reason = validate_ticker(df, _CFG)
    assert not ok
    assert "volume" in reason.lower()


def test_validate_large_gap():
    """A gap of > 7 calendar days between adjacent rows should fail."""
    df = _make_ohlcv(600)
    # Remove 10 rows in the middle to create a calendar gap > 7 days.
    drop_slice = df.index[200:210]
    df = df.drop(drop_slice)
    ok, reason = validate_ticker(df, _CFG)
    assert not ok
    assert "gap" in reason.lower()


# ── download_ticker ───────────────────────────────────────────────────────


def test_download_ticker_returns_dataframe():
    """Smoke test: download AAPL for a short period and check shape/columns."""
    df = download_ticker("AAPL", "2024-01-02", "2024-01-31")
    assert df is not None, "download_ticker returned None for AAPL"
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    for col in ("open", "high", "low", "close", "volume"):
        assert col in df.columns, f"Missing column: {col}"
    assert df.index.name == "date"


def test_download_ticker_incremental_up_to_date():
    """If existing data ends at today, the function returns it unchanged."""
    today = pd.Timestamp(date.today())
    existing = _make_ohlcv(600)
    existing.index = pd.bdate_range(end=today, periods=600)

    result = download_ticker("AAPL", "2020-01-01", "today", existing=existing)
    assert result is not None
    # Should return existing without modification (already up to date).
    assert len(result) == len(existing)


def test_download_ticker_bad_symbol_returns_none():
    """An obviously invalid ticker should return None (no data)."""
    result = download_ticker("ZZZZZZZ_INVALID", "2024-01-02", "2024-01-10")
    assert result is None
