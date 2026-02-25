"""
Unit tests for features/labels.py.
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from features.labels import add_labels, label_distribution

_CFG = {
    "labels": {
        "forward_window": 5,
        "long_threshold":  0.01,
        "short_threshold": -0.01,
    }
}


def _make_close(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Return a minimal DataFrame with only a close column."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    price = 100 + np.cumsum(rng.normal(0, 1, n))
    price = np.clip(price, 5, None)
    return pd.DataFrame({"close": price}, index=dates)


# ── add_labels ────────────────────────────────────────────────────────────


def test_rows_dropped_equal_forward_window():
    """Last forward_window rows must be removed."""
    df  = _make_close(100)
    out = add_labels(df, _CFG)
    assert len(out) == 100 - _CFG["labels"]["forward_window"]


def test_label_values_are_ternary():
    """All label values must be in {-1, 0, 1}."""
    df  = _make_close(200)
    out = add_labels(df, _CFG)
    assert set(out["label"].unique()).issubset({-1, 0, 1})


def test_label_consistency_long():
    """Rows with forward_return > long_threshold must have label == 1."""
    df  = _make_close(200)
    out = add_labels(df, _CFG)
    mask = out["forward_return"] > _CFG["labels"]["long_threshold"]
    assert (out.loc[mask, "label"] == 1).all()


def test_label_consistency_short():
    """Rows with forward_return < short_threshold must have label == -1."""
    df  = _make_close(200)
    out = add_labels(df, _CFG)
    mask = out["forward_return"] < _CFG["labels"]["short_threshold"]
    assert (out.loc[mask, "label"] == -1).all()


def test_label_consistency_flat():
    """Rows within thresholds must have label == 0."""
    df  = _make_close(200)
    out = add_labels(df, _CFG)
    mask = (
        (out["forward_return"] >= _CFG["labels"]["short_threshold"]) &
        (out["forward_return"] <= _CFG["labels"]["long_threshold"])
    )
    assert (out.loc[mask, "label"] == 0).all()


def test_input_not_mutated():
    """Original DataFrame must not gain new columns."""
    df   = _make_close(100)
    cols = set(df.columns)
    add_labels(df, _CFG)
    assert set(df.columns) == cols


def test_forward_return_formula():
    """Verify the forward_return calculation for a known sequence."""
    n  = 10
    df = pd.DataFrame(
        {"close": np.arange(1.0, n + 1.0)},
        index=pd.bdate_range("2020-01-01", periods=n),
    )
    out = add_labels(df, {"labels": {"forward_window": 1, "long_threshold": 0.01, "short_threshold": -0.01}})
    # Row 0: close=1, forward close=2 → return = 1.0 (100%)
    assert abs(out["forward_return"].iloc[0] - 1.0) < 1e-9


def test_all_rows_labeled():
    """No NaN labels in the output."""
    df  = _make_close(100)
    out = add_labels(df, _CFG)
    assert out["label"].isna().sum() == 0


# ── label_distribution ────────────────────────────────────────────────────


def test_distribution_sums_to_total():
    """Counts must sum to the total number of rows."""
    df   = _make_close(200)
    out  = add_labels(df, _CFG)
    dist = label_distribution(out)
    assert dist["count"].sum() == len(out)


def test_distribution_has_all_classes():
    """With a large random dataset all three classes should appear."""
    rng = np.random.default_rng(0)
    n   = 2000
    df  = pd.DataFrame(
        {"close": 100 + np.cumsum(rng.normal(0, 2, n))},
        index=pd.bdate_range("2015-01-01", periods=n),
    )
    out  = add_labels(df, _CFG)
    dist = label_distribution(out)
    assert set(dist.index) == {-1, 0, 1}


# ── Binary label columns ──────────────────────────────────────────────────


def test_label_long_column_exists():
    """add_labels must produce a label_long column."""
    df  = _make_close(100)
    out = add_labels(df, _CFG)
    assert "label_long" in out.columns


def test_label_short_column_exists():
    """add_labels must produce a label_short column."""
    df  = _make_close(100)
    out = add_labels(df, _CFG)
    assert "label_short" in out.columns


def test_label_long_binary_values():
    """label_long must contain only 0 and 1."""
    df  = _make_close(200)
    out = add_labels(df, _CFG)
    assert set(out["label_long"].unique()).issubset({0, 1})


def test_label_short_binary_values():
    """label_short must contain only 0 and 1."""
    df  = _make_close(200)
    out = add_labels(df, _CFG)
    assert set(out["label_short"].unique()).issubset({0, 1})


def test_label_long_consistent_with_forward_return():
    """label_long == 1 iff forward_return > long_threshold."""
    df  = _make_close(200)
    out = add_labels(df, _CFG)
    expected = (out["forward_return"] > _CFG["labels"]["long_threshold"]).astype(int)
    pd.testing.assert_series_equal(
        out["label_long"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_label_short_consistent_with_forward_return():
    """label_short == 1 iff forward_return < short_threshold."""
    df  = _make_close(200)
    out = add_labels(df, _CFG)
    expected = (out["forward_return"] < _CFG["labels"]["short_threshold"]).astype(int)
    pd.testing.assert_series_equal(
        out["label_short"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_label_long_short_mutually_exclusive():
    """label_long and label_short cannot both be 1 on the same row."""
    df  = _make_close(200)
    out = add_labels(df, _CFG)
    both = (out["label_long"] == 1) & (out["label_short"] == 1)
    assert not both.any(), "label_long and label_short are both 1 on some rows"
