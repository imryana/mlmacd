"""
Unit tests for signals/trade_setup.py.

All tests use synthetic signal rows — no disk I/O or network calls.
"""

import pathlib
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from signals.trade_setup import (
    _add_bdays,
    build_all_trade_cards,
    build_trade_card,
)

# ── fixtures ──────────────────────────────────────────────────────────────

_CFG = {
    "signals": {
        "entry_method":                 "limit",
        "entry_buffer_atr":             0.25,
        "entry_expiry_bars":            2,
        "stop_multiplier":              2.0,
        "rr_ratio":                     2.0,
        "trailing_stop_activation_atr": 1.0,
        "trailing_stop_distance_atr":   1.5,
        "time_exit_bars":               5,
    },
    "portfolio": {
        "size":              50_000,
        "risk_per_trade":    0.01,
        "max_open_positions": 10,
        "max_sector_exposure": 0.30,
    },
}

_LONG_ROW = pd.Series(
    {
        "ticker":         "AAPL",
        "signal":         1,
        "signal_name":    "LONG",
        "confidence":     0.82,
        "regime":         1,
        "regime_name":    "bull",
        "price":          200.0,
        "atr":            4.0,
        "adx":            28.0,
        "rsi":            55.0,
        "macd_histogram": 0.5,
        "date":           "2026-01-15",
    }
)

_SHORT_ROW = pd.Series(
    {
        "ticker":         "TSLA",
        "signal":         -1,
        "signal_name":    "SHORT",
        "confidence":     0.70,
        "regime":         2,
        "regime_name":    "bear",
        "price":          250.0,
        "atr":            5.0,
        "adx":            35.0,
        "rsi":            72.0,
        "macd_histogram": -0.3,
        "date":           "2026-01-15",
    }
)


# ── _add_bdays ────────────────────────────────────────────────────────────


def test_add_bdays_positive():
    """Adding 2 business days to a Monday should give Wednesday."""
    monday = date(2026, 1, 5)   # Monday
    result = _add_bdays(monday, 2)
    assert result == date(2026, 1, 7)   # Wednesday


def test_add_bdays_skips_weekend():
    """Adding 2 business days to a Friday should give Tuesday."""
    friday = date(2026, 1, 9)
    result = _add_bdays(friday, 2)
    assert result == date(2026, 1, 13)   # Tuesday


def test_add_bdays_zero():
    d = date(2026, 1, 5)
    assert _add_bdays(d, 0) == d


# ── build_trade_card — LONG ───────────────────────────────────────────────


def test_long_limit_entry_below_close():
    """Limit entry for a LONG should be below current close."""
    card = build_trade_card(_LONG_ROW, _CFG)
    assert card["limit_entry"] < _LONG_ROW["price"]


def test_long_limit_entry_formula():
    """limit_entry = close - atr × entry_buffer_atr."""
    card     = build_trade_card(_LONG_ROW, _CFG)
    expected = 200.0 - 4.0 * 0.25
    assert abs(card["limit_entry"] - expected) < 1e-6


def test_long_stop_below_entry():
    """Stop loss for a LONG must be below limit_entry."""
    card = build_trade_card(_LONG_ROW, _CFG)
    assert card["stop_loss"] < card["limit_entry"]


def test_long_stop_formula():
    """stop_loss = entry - atr × stop_multiplier."""
    card    = build_trade_card(_LONG_ROW, _CFG)
    entry   = 200.0 - 4.0 * 0.25   # = 199.0
    expected_stop = entry - 4.0 * 2.0    # = 191.0
    assert abs(card["stop_loss"] - expected_stop) < 1e-6


def test_long_take_profit_above_entry():
    """Take profit for a LONG must be above limit_entry."""
    card = build_trade_card(_LONG_ROW, _CFG)
    assert card["take_profit"] > card["limit_entry"]


def test_long_rr_ratio():
    """Actual R:R = target_pct / |stop_pct| should equal config rr_ratio."""
    card = build_trade_card(_LONG_ROW, _CFG)
    assert abs(card["rr_ratio"] - _CFG["signals"]["rr_ratio"]) < 0.01


def test_long_trailing_activation_above_entry():
    """Trailing activation for a LONG should be above entry."""
    card = build_trade_card(_LONG_ROW, _CFG)
    assert card["trailing_activation"] > card["limit_entry"]


def test_long_stop_pct_is_negative():
    """Stop % for a LONG should be negative (move against us)."""
    card = build_trade_card(_LONG_ROW, _CFG)
    assert card["stop_pct"] < 0


def test_long_target_pct_is_positive():
    """Target % for a LONG should be positive."""
    card = build_trade_card(_LONG_ROW, _CFG)
    assert card["target_pct"] > 0


# ── build_trade_card — SHORT ──────────────────────────────────────────────


def test_short_limit_entry_above_close():
    """Limit entry for a SHORT should be above current close."""
    card = build_trade_card(_SHORT_ROW, _CFG)
    assert card["limit_entry"] > _SHORT_ROW["price"]


def test_short_stop_above_entry():
    """Stop loss for a SHORT must be above limit_entry."""
    card = build_trade_card(_SHORT_ROW, _CFG)
    assert card["stop_loss"] > card["limit_entry"]


def test_short_take_profit_below_entry():
    """Take profit for a SHORT must be below limit_entry."""
    card = build_trade_card(_SHORT_ROW, _CFG)
    assert card["take_profit"] < card["limit_entry"]


def test_short_trailing_activation_below_entry():
    """Trailing activation for a SHORT should be below entry."""
    card = build_trade_card(_SHORT_ROW, _CFG)
    assert card["trailing_activation"] < card["limit_entry"]


def test_short_rr_ratio():
    """SHORT R:R should also equal config rr_ratio."""
    card = build_trade_card(_SHORT_ROW, _CFG)
    assert abs(card["rr_ratio"] - _CFG["signals"]["rr_ratio"]) < 0.01


# ── Position sizing ───────────────────────────────────────────────────────


def test_position_risk_amount():
    """risk_amount = portfolio_size × risk_per_trade."""
    card     = build_trade_card(_LONG_ROW, _CFG)
    expected = 50_000 * 0.01
    assert abs(card["risk_amount"] - expected) < 1e-6


def test_position_units_formula():
    """position_units = risk_amount / risk_per_unit."""
    card          = build_trade_card(_LONG_ROW, _CFG)
    entry         = 200.0 - 4.0 * 0.25   # 199.0
    stop          = entry - 4.0 * 2.0    # 191.0
    risk_per_unit = abs(entry - stop)     # 8.0
    risk_amount   = 50_000 * 0.01        # 500.0
    expected_units = risk_amount / risk_per_unit  # 62.5
    assert abs(card["position_units"] - expected_units) < 1e-4


def test_position_value_formula():
    """position_value = units × entry_price."""
    card = build_trade_card(_LONG_ROW, _CFG)
    assert abs(card["position_value"] - card["position_units"] * card["limit_entry"]) < 0.01


def test_position_pct_formula():
    """position_pct = position_value / portfolio_size × 100."""
    card     = build_trade_card(_LONG_ROW, _CFG)
    expected = card["position_value"] / 50_000 * 100
    assert abs(card["position_pct"] - expected) < 0.01


# ── Dates ─────────────────────────────────────────────────────────────────


def test_entry_expiry_is_future():
    """entry_expiry_date must be after today."""
    card   = build_trade_card(_LONG_ROW, _CFG)
    expiry = date.fromisoformat(card["entry_expiry_date"])
    assert expiry > date.today()


def test_time_exit_is_future():
    """time_exit_date must be after today."""
    card      = build_trade_card(_LONG_ROW, _CFG)
    time_exit = date.fromisoformat(card["time_exit_date"])
    assert time_exit > date.today()


def test_time_exit_after_entry_expiry():
    """time_exit_date (5 days) should be >= entry_expiry_date (2 days)."""
    card = build_trade_card(_LONG_ROW, _CFG)
    assert (
        date.fromisoformat(card["time_exit_date"])
        >= date.fromisoformat(card["entry_expiry_date"])
    )


# ── Required fields ───────────────────────────────────────────────────────


def test_trade_card_required_fields():
    card = build_trade_card(_LONG_ROW, _CFG)
    for field in (
        "ticker", "signal", "signal_name", "confidence",
        "limit_entry", "stop_loss", "take_profit", "stop_pct", "target_pct",
        "rr_ratio", "trailing_activation", "trailing_distance", "time_exit_date",
        "risk_amount", "position_units", "position_value", "position_pct",
    ):
        assert field in card, f"Missing field: {field}"


# ── build_all_trade_cards ─────────────────────────────────────────────────


def test_build_all_trade_cards_shape():
    signals = pd.DataFrame([_LONG_ROW, _SHORT_ROW])
    cards   = build_all_trade_cards(signals, _CFG)
    assert len(cards) == 2


def test_build_all_trade_cards_empty():
    cards = build_all_trade_cards(pd.DataFrame(), _CFG)
    assert cards.empty


def test_build_all_trade_cards_preserves_direction():
    signals = pd.DataFrame([_LONG_ROW, _SHORT_ROW])
    cards   = build_all_trade_cards(signals, _CFG)
    assert cards.iloc[0]["signal"] == 1
    assert cards.iloc[1]["signal"] == -1
