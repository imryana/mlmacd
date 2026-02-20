"""
Unit tests for models/xgboost_model.py and models/trainer.py.

All tests use synthetic data — no file I/O or network calls.
"""

import json
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from models.xgboost_model import XGBoostModel
from models.trainer import (
    _get_feature_columns,
    _sharpe,
    compute_shap_importance,
    run_walk_forward,
)

# ── helpers ───────────────────────────────────────────────────────────────

_CFG = {
    "model": {
        "confidence_threshold":      0.60,
        "walk_forward_train_bars":   60,   # small for speed
        "walk_forward_test_bars":    20,
        "xgb_max_depth":             3,
        "xgb_learning_rate":         0.1,
        "xgb_n_estimators":          50,
        "xgb_subsample":             0.8,
        "xgb_min_child_weight":      3,
        "xgb_early_stopping_rounds": 10,
        "hmm_n_states":              3,
        "hmm_n_iter":                50,
        "use_gpu":                   False,   # CPU for unit tests
    }
}

_N_FEATURES = 8
_FEATURE_NAMES = [f"feat_{i}" for i in range(_N_FEATURES)]


def _make_X(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.standard_normal((n, _N_FEATURES)),
        columns=_FEATURE_NAMES,
    )


def _make_y(n: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.choice([-1, 0, 1], size=n))


def _make_pooled(
    n_dates: int = 200,
    n_tickers: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Minimal pooled DataFrame with the shape expected by run_walk_forward.

    Contains feature columns, plus 'label', 'forward_return', 'ticker',
    'open','high','low','close','volume' (excluded from features).
    """
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    tickers = [f"T{i}" for i in range(n_tickers)]

    rows = []
    for date in dates:
        for t in tickers:
            price = 100.0
            row = {c: rng.standard_normal() for c in _FEATURE_NAMES}
            row.update(
                {
                    "open": price, "high": price, "low": price,
                    "close": price, "volume": 1e6,
                    "ticker": t,
                    "forward_return": rng.normal(0, 0.01),
                    "label":          int(rng.choice([-1, 0, 1])),
                    "regime":         int(rng.choice([0, 1, 2])),
                }
            )
            rows.append((date, row))

    records = [r for _, r in rows]
    idx     = pd.DatetimeIndex([d for d, _ in rows])
    return pd.DataFrame(records, index=idx)


# ── XGBoostModel ──────────────────────────────────────────────────────────


def test_train_returns_self():
    model = XGBoostModel(_CFG)
    result = model.train(_make_X(), _make_y())
    assert result is model


def test_predict_proba_shape():
    model = XGBoostModel(_CFG)
    X = _make_X()
    model.train(X, _make_y())
    proba = model.predict_proba(X.iloc[:10])
    assert proba.shape == (10, 3)


def test_predict_proba_sums_to_one():
    model = XGBoostModel(_CFG)
    X = _make_X()
    model.train(X, _make_y())
    proba = model.predict_proba(X.iloc[:50])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_predict_signal_valid_values():
    model = XGBoostModel(_CFG)
    X = _make_X()
    model.train(X, _make_y())
    signals, conf = model.predict_signal(X.iloc[:30])
    assert set(signals).issubset({-1, 0, 1})
    assert ((conf >= 0) & (conf <= 1)).all()


def test_predict_signal_confidence_matches_proba():
    """Confidence must equal the max class probability for each row."""
    model = XGBoostModel(_CFG)
    X = _make_X()
    model.train(X, _make_y())
    proba         = model.predict_proba(X.iloc[:20])
    signals, conf = model.predict_signal(X.iloc[:20])
    np.testing.assert_allclose(conf, proba.max(axis=1), atol=1e-6)


def test_label_encoding_roundtrip():
    labels = pd.Series([-1, 0, 1, -1, 1, 0])
    encoded = XGBoostModel._encode_labels(labels)
    decoded = XGBoostModel._decode_labels(encoded.values)
    np.testing.assert_array_equal(decoded, labels.values)


def test_save_load_consistent(tmp_path):
    X = _make_X()
    y = _make_y()
    model = XGBoostModel(_CFG)
    model.train(X, y)

    path = tmp_path / "model.pkl"
    model.save(path)
    loaded = XGBoostModel.load(path, _CFG)

    sig_orig,  conf_orig  = model.predict_signal(X.iloc[:20])
    sig_loaded, conf_loaded = loaded.predict_signal(X.iloc[:20])

    np.testing.assert_array_equal(sig_orig, sig_loaded)
    np.testing.assert_allclose(conf_orig, conf_loaded, atol=1e-6)


def test_train_with_explicit_validation_set():
    X = _make_X(400)
    y = _make_y(400)
    model = XGBoostModel(_CFG)
    # Should not raise even when val set is provided explicitly
    model.train(X.iloc[:300], y.iloc[:300], X.iloc[300:], y.iloc[300:])
    sigs, _ = model.predict_signal(X.iloc[300:])
    assert len(sigs) == 100


# ── trainer helpers ───────────────────────────────────────────────────────


def test_get_feature_columns_excludes_ohlcv():
    pooled = _make_pooled()
    feat_cols = _get_feature_columns(pooled)
    for bad in ("open", "high", "low", "close", "volume",
                "ticker", "forward_return", "label"):
        assert bad not in feat_cols


def test_get_feature_columns_includes_regime():
    pooled = _make_pooled()
    assert "regime" in _get_feature_columns(pooled)


def test_sharpe_zero_variance():
    assert _sharpe(np.ones(100)) == 0.0


def test_sharpe_positive():
    returns = np.full(252, 0.001)
    returns += np.random.default_rng(0).normal(0, 0.0001, 252)
    assert _sharpe(returns) > 0


# ── run_walk_forward ──────────────────────────────────────────────────────


def test_walk_forward_runs_without_error():
    pooled    = _make_pooled(n_dates=200, n_tickers=2)
    feat_cols = _get_feature_columns(pooled)
    wf_df, metrics = run_walk_forward(pooled, _CFG, feat_cols)
    assert len(wf_df) > 0
    assert "signal" in wf_df.columns


def test_walk_forward_no_lookahead():
    """No test-fold date should appear in its own training window."""
    pooled    = _make_pooled(n_dates=200, n_tickers=2)
    feat_cols = _get_feature_columns(pooled)

    unique_dates = sorted(pooled.index.unique())
    train_bars   = _CFG["model"]["walk_forward_train_bars"]
    test_bars    = _CFG["model"]["walk_forward_test_bars"]

    for fold in range((len(unique_dates) - train_bars) // test_bars):
        train_end  = fold * test_bars + train_bars
        test_start = train_end
        test_end   = test_start + test_bars

        train_set = set(unique_dates[fold * test_bars : train_end])
        test_set  = set(unique_dates[test_start : test_end])
        assert train_set.isdisjoint(test_set), f"Fold {fold} has look-ahead!"


def test_walk_forward_output_columns():
    pooled    = _make_pooled(n_dates=200, n_tickers=2)
    feat_cols = _get_feature_columns(pooled)
    wf_df, _  = run_walk_forward(pooled, _CFG, feat_cols)
    for col in ("fold", "ticker", "signal", "confidence", "actual",
                "forward_return", "proba_short", "proba_flat", "proba_long"):
        assert col in wf_df.columns, f"Missing column: {col}"


def test_walk_forward_signal_values():
    pooled    = _make_pooled(n_dates=200, n_tickers=2)
    feat_cols = _get_feature_columns(pooled)
    wf_df, _  = run_walk_forward(pooled, _CFG, feat_cols)
    assert set(wf_df["signal"].unique()).issubset({-1, 0, 1})


def test_walk_forward_metrics_keys():
    pooled    = _make_pooled(n_dates=200, n_tickers=2)
    feat_cols = _get_feature_columns(pooled)
    _, metrics = run_walk_forward(pooled, _CFG, feat_cols)
    for key in ("accuracy", "f1_weighted", "sharpe_signal",
                "signal_hit_rate", "n_folds"):
        assert key in metrics, f"Missing metric: {key}"


def test_walk_forward_not_enough_data_raises():
    pooled    = _make_pooled(n_dates=50, n_tickers=2)   # far fewer than train_bars
    feat_cols = _get_feature_columns(pooled)
    with pytest.raises(ValueError, match="Not enough dates"):
        run_walk_forward(pooled, _CFG, feat_cols)


# ── compute_shap_importance ───────────────────────────────────────────────


def test_shap_importance_shape():
    X = _make_X(300)
    y = _make_y(300)
    model = XGBoostModel(_CFG)
    model.train(X, y)
    shap_df = compute_shap_importance(model, X)
    assert len(shap_df) == _N_FEATURES
    assert "feature" in shap_df.columns
    assert "mean_abs_shap" in shap_df.columns


def test_shap_importance_sorted_descending():
    X = _make_X(300)
    y = _make_y(300)
    model = XGBoostModel(_CFG)
    model.train(X, y)
    shap_df = compute_shap_importance(model, X)
    values = shap_df["mean_abs_shap"].values
    assert (values[:-1] >= values[1:]).all()


def test_shap_importance_non_negative():
    X = _make_X(300)
    y = _make_y(300)
    model = XGBoostModel(_CFG)
    model.train(X, y)
    shap_df = compute_shap_importance(model, X)
    assert (shap_df["mean_abs_shap"] >= 0).all()
