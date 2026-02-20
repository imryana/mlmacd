"""
Unit tests for regime/hmm_detector.py.

Uses synthetic multi-ticker data so no file I/O or network is required.
The synthetic dataset has three well-separated regimes so the HMM can
reliably discover them.
"""

import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from regime.hmm_detector import (
    BEAR, BULL, CHOPPY,
    HMMDetector,
    _REGIME_NAMES,
    extract_hmm_features,
)

# ── helpers ───────────────────────────────────────────────────────────────

_CFG = {
    "model": {
        "hmm_n_states": 3,
        "hmm_n_iter":   50,
    }
}


def _make_ticker_df(
    n: int = 400,
    trend: float = 0.05,
    vol: float = 0.01,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Return a minimal processed-ticker DataFrame suitable for HMM training.

    ``trend`` controls the daily log-return mean; ``vol`` controls noise.
    Columns present: close, realised_vol, atr_pct, plus a ticker label.
    """
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)

    log_ret = rng.normal(trend / 252, vol, n)
    price   = 100 * np.exp(np.cumsum(log_ret))

    rv  = pd.Series(log_ret).rolling(20, min_periods=20).std() * np.sqrt(252)
    atr = pd.Series(price).pct_change().abs().rolling(14, min_periods=14).mean() * 100

    return pd.DataFrame(
        {
            "close":        price,
            "realised_vol": rv.values,
            "atr_pct":      atr.values,
        },
        index=dates,
    )


def _make_pooled(n_tickers: int = 5) -> pd.DataFrame:
    """Create a pooled DataFrame with *n_tickers* synthetic tickers."""
    trends = np.linspace(-0.1, 0.3, n_tickers)
    frames = []
    for i, trend in enumerate(trends):
        df         = _make_ticker_df(n=400, trend=trend, seed=i)
        df["ticker"] = f"T{i:02d}"
        frames.append(df)
    return pd.concat(frames)


# ── extract_hmm_features ─────────────────────────────────────────────────


def test_extract_hmm_features_columns():
    df   = _make_ticker_df()
    feat = extract_hmm_features(df)
    assert list(feat.columns) == ["rolling_5d_return", "realised_vol", "atr_pct"]


def test_extract_hmm_features_no_nans():
    df   = _make_ticker_df()
    feat = extract_hmm_features(df)
    assert feat.isna().sum().sum() == 0


def test_extract_hmm_features_warmup_dropped():
    """The first 5 rows should be dropped (rolling 5-day return is NaN)."""
    df   = _make_ticker_df(n=200)
    feat = extract_hmm_features(df)
    assert len(feat) < len(df)


# ── HMMDetector.fit ───────────────────────────────────────────────────────


def test_fit_returns_self():
    pooled   = _make_pooled()
    detector = HMMDetector(_CFG)
    result   = detector.fit(pooled)
    assert result is detector


def test_fit_sets_model_and_state_map():
    pooled   = _make_pooled()
    detector = HMMDetector(_CFG)
    detector.fit(pooled)
    assert detector._model     is not None
    assert detector._state_map is not None
    assert len(detector._state_map) == 3


def test_state_map_covers_all_states():
    """Every HMM state must be assigned a regime code."""
    pooled   = _make_pooled()
    detector = HMMDetector(_CFG)
    detector.fit(pooled)
    for state in range(detector.n_states):
        assert state in detector._state_map


def test_state_map_values_are_valid_regime_codes():
    """Regime codes must only be CHOPPY, BULL, or BEAR."""
    pooled   = _make_pooled()
    detector = HMMDetector(_CFG)
    detector.fit(pooled)
    valid = {CHOPPY, BULL, BEAR}
    assert set(detector._state_map.values()).issubset(valid)


def test_bull_has_highest_mean_return():
    """The BULL state must have a higher mean 5-day return than BEAR."""
    pooled   = _make_pooled(n_tickers=6)
    detector = HMMDetector(_CFG)
    detector.fit(pooled)

    means = detector._model.means_[:, 0]
    bull_state = [s for s, r in detector._state_map.items() if r == BULL][0]
    bear_state = [s for s, r in detector._state_map.items() if r == BEAR][0]
    assert means[bull_state] > means[bear_state]


# ── HMMDetector.predict_regime ────────────────────────────────────────────


def test_predict_regime_length():
    """Output series must have the same length as input DataFrame."""
    pooled   = _make_pooled()
    detector = HMMDetector(_CFG)
    detector.fit(pooled)

    sample = _make_ticker_df(n=300)
    regime = detector.predict_regime(sample)
    assert len(regime) == len(sample)


def test_predict_regime_valid_codes():
    """All predicted regime codes must be in {0, 1, 2}."""
    pooled   = _make_pooled()
    detector = HMMDetector(_CFG)
    detector.fit(pooled)

    sample = _make_ticker_df(n=300)
    regime = detector.predict_regime(sample)
    assert set(regime.unique()).issubset({CHOPPY, BULL, BEAR})


def test_predict_regime_warmup_is_choppy():
    """Warmup rows (NaN features) must be filled with CHOPPY (0)."""
    pooled   = _make_pooled()
    detector = HMMDetector(_CFG)
    detector.fit(pooled)

    sample = _make_ticker_df(n=300)
    feat   = extract_hmm_features(sample)
    regime = detector.predict_regime(sample)

    # Rows not in feat.index are warmup rows
    warmup_idx = sample.index.difference(feat.index)
    assert (regime.loc[warmup_idx] == CHOPPY).all()


# ── HMMDetector.add_regime_column ─────────────────────────────────────────


def test_add_regime_column_present():
    pooled   = _make_pooled()
    detector = HMMDetector(_CFG)
    detector.fit(pooled)

    sample = _make_ticker_df(n=300)
    out    = detector.add_regime_column(sample)
    assert "regime" in out.columns


def test_add_regime_column_does_not_mutate():
    pooled   = _make_pooled()
    detector = HMMDetector(_CFG)
    detector.fit(pooled)

    sample = _make_ticker_df(n=200)
    cols   = set(sample.columns)
    detector.add_regime_column(sample)
    assert set(sample.columns) == cols   # original unchanged


# ── save / load ───────────────────────────────────────────────────────────


def test_save_and_load_produces_same_predictions():
    pooled   = _make_pooled()
    detector = HMMDetector(_CFG)
    detector.fit(pooled)

    sample    = _make_ticker_df(n=300)
    preds_pre = detector.predict_regime(sample).values

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "hmm_model.pkl"
        detector.save(path)

        loaded    = HMMDetector.load(_CFG, path)
        preds_post = loaded.predict_regime(sample).values

    np.testing.assert_array_equal(preds_pre, preds_post)


def test_load_raises_if_file_missing():
    with pytest.raises(Exception):
        HMMDetector.load(_CFG, pathlib.Path("/nonexistent/hmm_model.pkl"))


# ── state_statistics ─────────────────────────────────────────────────────


def test_state_statistics_shape():
    pooled   = _make_pooled()
    detector = HMMDetector(_CFG)
    detector.fit(pooled)

    stats = detector.state_statistics(pooled)
    assert len(stats) == detector.n_states
    assert "regime_name" in stats.columns
    assert "n_observations" in stats.columns


def test_state_statistics_observations_sum():
    """Total observations across states must equal the usable dataset size."""
    pooled   = _make_pooled()
    detector = HMMDetector(_CFG)
    detector.fit(pooled)

    stats    = detector.state_statistics(pooled)
    total    = stats["n_observations"].sum()

    # Compute expected total (rows with non-NaN HMM features)
    expected = sum(
        len(extract_hmm_features(g.sort_index()))
        for _, g in pooled.groupby("ticker")
        if len(extract_hmm_features(g.sort_index())) >= detector.n_states * 2
    )
    assert total == expected
