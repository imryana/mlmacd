"""
trainer.py — Walk-forward training orchestration for the XGBoost signal model.

Pipeline
--------
1. Load data/processed/pooled_features.parquet
2. Add regime labels per ticker from the saved HMM
3. Walk-forward cross-validation (sliding window over unique dates):
     train window = config.model.walk_forward_train_bars  unique dates
     test  window = config.model.walk_forward_test_bars   unique dates
     slide forward by test_bars each iteration
4. Evaluate aggregate metrics and per-fold signal-weighted returns
5. Retrain final model on the full dataset
6. Compute SHAP feature importance
7. Save artefacts:
     models/saved/xgboost_final.pkl
     models/saved/wf_results.parquet
     models/saved/shap_importance.parquet
     models/saved/wf_metrics.json

Optional: pass --tune to run 50-trial Optuna search and write best
hyperparameters back to config.yaml.
"""

import argparse
import json
import logging
import pathlib
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from models.xgboost_model import XGBoostModel
from regime.hmm_detector import HMMDetector

log = logging.getLogger(__name__)

_ROOT = pathlib.Path(__file__).resolve().parents[1]

# ── Non-feature columns (excluded from the feature matrix) ───────────────

_OHLCV       = {"open", "high", "low", "close", "volume"}
_META        = {"ticker", "forward_return", "label"}
_NON_FEATURE = _OHLCV | _META


# ── Path helpers ──────────────────────────────────────────────────────────

def _load_config() -> dict:
    """Load and return config.yaml as a dict."""
    with open(_ROOT / "config" / "config.yaml") as fh:
        return yaml.safe_load(fh)


def _save_config(cfg: dict) -> None:
    """Write *cfg* back to config.yaml."""
    with open(_ROOT / "config" / "config.yaml", "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)


def _pooled_path()        -> pathlib.Path: return _ROOT / "data" / "processed" / "pooled_features.parquet"
def _final_model_path()   -> pathlib.Path: return _ROOT / "models" / "saved" / "xgboost_final.pkl"
def _wf_results_path()    -> pathlib.Path: return _ROOT / "models" / "saved" / "wf_results.parquet"
def _shap_path()          -> pathlib.Path: return _ROOT / "models" / "saved" / "shap_importance.parquet"
def _wf_metrics_path()    -> pathlib.Path: return _ROOT / "models" / "saved" / "wf_metrics.json"


# ── Data preparation ──────────────────────────────────────────────────────

def _add_regime_column(pooled: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Predict and attach a ``regime`` column to the pooled DataFrame.

    Each ticker's rows are processed as a separate HMM sequence so that
    inter-ticker boundaries do not corrupt the transition model.

    Parameters
    ----------
    pooled : pd.DataFrame
        Pooled feature dataset (output of features.pipeline).
    cfg : dict
        Full config dict.

    Returns
    -------
    pd.DataFrame
        Copy of *pooled* with a ``regime`` integer column appended.
    """
    detector = HMMDetector.load(cfg)
    pooled   = pooled.copy()

    regime_parts: list[pd.Series] = []
    for ticker, group in pooled.groupby("ticker"):
        regime_parts.append(detector.predict_regime(group.sort_index()))

    pooled["regime"] = pd.concat(regime_parts).reindex(pooled.index)
    return pooled


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return all columns that should be used as XGBoost features.

    Excludes OHLCV, ticker, label, and forward_return.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    list[str]
    """
    return [c for c in df.columns if c not in _NON_FEATURE]


# ── Walk-forward engine ───────────────────────────────────────────────────

def _sharpe(returns: np.ndarray) -> float:
    """
    Annualised Sharpe ratio (risk-free = 0) for a daily return series.

    Returns 0.0 if the series has zero variance.
    """
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252))


def run_walk_forward(
    df: pd.DataFrame,
    cfg: dict,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, dict]:
    """
    Run sliding-window walk-forward cross-validation.

    The window slides over *unique* trading dates so that every ticker
    present on a given date is included in training/testing together.

    Parameters
    ----------
    df : pd.DataFrame
        Full prepared dataset with all feature columns and ``label``.
    cfg : dict
        Full config dict.
    feature_cols : list[str]
        Column names to use as model input.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        ``(wf_results_df, aggregate_metrics)``

        *wf_results_df* has columns: fold, ticker, signal, confidence,
        actual, forward_return, regime, proba_short, proba_flat, proba_long.

        *aggregate_metrics* is a plain dict of scalar summary statistics.
    """
    train_bars = cfg["model"]["walk_forward_train_bars"]
    test_bars  = cfg["model"]["walk_forward_test_bars"]

    unique_dates = np.array(sorted(df.index.unique()))
    n_dates      = len(unique_dates)
    n_folds      = max(0, (n_dates - train_bars) // test_bars)

    if n_folds == 0:
        raise ValueError(
            f"Not enough dates for walk-forward: {n_dates} dates "
            f"but need at least {train_bars + test_bars}."
        )

    log.info(
        "Walk-forward: %d folds, train=%d dates, test=%d dates.",
        n_folds, train_bars, test_bars,
    )

    all_results: list[pd.DataFrame] = []

    for fold in range(n_folds):
        train_start = fold * test_bars
        train_end   = train_start + train_bars
        test_end    = train_end   + test_bars

        if test_end > n_dates:
            break

        train_dates = set(unique_dates[train_start:train_end])
        test_dates  = set(unique_dates[train_end:test_end])

        train_df = df[df.index.isin(train_dates)]
        test_df  = df[df.index.isin(test_dates)]

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        X_train = train_df[feature_cols]
        y_train = train_df["label"]
        X_test  = test_df[feature_cols]
        y_test  = test_df["label"]

        model = XGBoostModel(cfg)
        model.train(X_train, y_train)

        proba           = model.predict_proba(X_test)
        signals, conf   = model.predict_signal(X_test)

        fold_df = pd.DataFrame(
            {
                "fold":           fold,
                "ticker":         test_df["ticker"].values,
                "signal":         signals,
                "confidence":     conf,
                "actual":         y_test.values,
                "forward_return": test_df["forward_return"].values,
                "regime":         test_df["regime"].values if "regime" in test_df else 0,
                "proba_short":    proba[:, 0],
                "proba_flat":     proba[:, 1],
                "proba_long":     proba[:, 2],
            },
            index=test_df.index,
        )
        all_results.append(fold_df)

        # Per-fold logging
        acc = accuracy_score(y_test, signals)
        sig_returns = signals * test_df["forward_return"].values
        log.info("Fold %2d | acc=%.3f | sig_ret=%.4f", fold, acc, sig_returns.mean())

    if not all_results:
        raise RuntimeError("Walk-forward produced zero results.")

    wf_df = pd.concat(all_results).sort_index()

    # ── Aggregate metrics ────────────────────────────────────────────────

    y_true = wf_df["actual"].values
    y_pred = wf_df["signal"].values
    sig_ret = (wf_df["signal"] * wf_df["forward_return"]).values

    # Only evaluate non-flat signals for signal-quality metrics
    active_mask = y_pred != 0
    active_ret  = sig_ret[active_mask]

    metrics = {
        "n_folds":           n_folds,
        "total_test_rows":   int(len(wf_df)),
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "f1_short":          float(f1_score(y_true, y_pred, labels=[-1], average="macro", zero_division=0)),
        "f1_flat":           float(f1_score(y_true, y_pred, labels=[ 0], average="macro", zero_division=0)),
        "f1_long":           float(f1_score(y_true, y_pred, labels=[ 1], average="macro", zero_division=0)),
        "f1_weighted":       float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix":  confusion_matrix(y_true, y_pred, labels=[-1, 0, 1]).tolist(),
        "signal_hit_rate":   float((active_ret > 0).mean()) if len(active_ret) else 0.0,
        "mean_signal_return": float(active_ret.mean()) if len(active_ret) else 0.0,
        "sharpe_signal":     _sharpe(active_ret),
        "pct_signals":       float(active_mask.mean()),
    }

    return wf_df, metrics


# ── SHAP importance ───────────────────────────────────────────────────────

def compute_shap_importance(
    model: XGBoostModel,
    X: pd.DataFrame,
    max_rows: int = 5000,
) -> pd.DataFrame:
    """
    Compute mean |SHAP| per feature using TreeExplainer.

    For multi-class models, SHAP values are averaged across classes.

    Parameters
    ----------
    model : XGBoostModel
    X : pd.DataFrame
        Feature matrix (full dataset or a representative sample).
    max_rows : int
        Maximum number of rows to use for SHAP computation (speed/memory).

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``mean_abs_shap``.  Sorted descending.
    """
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer   = shap.TreeExplainer(model.model)
        shap_values = explainer.shap_values(X)

    # shap_values: list of (n_samples, n_features) arrays  — one per class
    # or a single (n_samples, n_features, n_classes) array in newer shap
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        # shape (n_samples, n_features, n_classes)
        mean_abs = np.abs(shap_values).mean(axis=(0, 2))

    importance = (
        pd.DataFrame({"feature": list(X.columns), "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return importance


# ── Optuna hyperparameter search ──────────────────────────────────────────

def tune_hyperparameters(
    df: pd.DataFrame,
    cfg: dict,
    feature_cols: list[str],
    n_trials: int = 50,
) -> dict:
    """
    Run an Optuna study to maximise walk-forward signal Sharpe ratio.

    Searches over XGBoost hyperparameters.  Best parameters are written
    back to *cfg* (in-memory only; caller should persist to config.yaml).

    Parameters
    ----------
    df : pd.DataFrame
    cfg : dict
    feature_cols : list[str]
    n_trials : int

    Returns
    -------
    dict
        Best hyperparameter values.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        trial_cfg = json.loads(json.dumps(cfg))   # deep copy
        m = trial_cfg["model"]
        m["xgb_max_depth"]        = trial.suggest_int("max_depth",        3, 9)
        m["xgb_learning_rate"]    = trial.suggest_float("learning_rate",  0.01, 0.3, log=True)
        m["xgb_n_estimators"]     = trial.suggest_int("n_estimators",     100, 800)
        m["xgb_subsample"]        = trial.suggest_float("subsample",      0.5, 1.0)
        m["xgb_min_child_weight"] = trial.suggest_int("min_child_weight", 1, 20)

        try:
            _, metrics = run_walk_forward(df, trial_cfg, feature_cols)
            return metrics["sharpe_signal"]
        except Exception:
            return -999.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    log.info("Best Optuna params: %s  (Sharpe=%.4f)", best, study.best_value)

    # Write best params back to cfg
    cfg["model"]["xgb_max_depth"]        = best["max_depth"]
    cfg["model"]["xgb_learning_rate"]    = best["learning_rate"]
    cfg["model"]["xgb_n_estimators"]     = best["n_estimators"]
    cfg["model"]["xgb_subsample"]        = best["subsample"]
    cfg["model"]["xgb_min_child_weight"] = best["min_child_weight"]

    return best


# ── Main training pipeline ────────────────────────────────────────────────

def run_training_pipeline(
    cfg: dict,
    tune: bool = False,
) -> tuple[XGBoostModel, pd.DataFrame, dict]:
    """
    Execute the full model training pipeline end-to-end.

    Parameters
    ----------
    cfg : dict
        Full config dict from config.yaml.
    tune : bool
        If True, run Optuna hyperparameter search first.

    Returns
    -------
    tuple[XGBoostModel, pd.DataFrame, dict]
        ``(final_model, wf_results_df, aggregate_metrics)``
    """
    # ── 1. Load pooled features ─────────────────────────────────────────
    pooled_path = _pooled_path()
    if not pooled_path.exists():
        raise FileNotFoundError(
            "Pooled features not found. "
            "Run `python -m features.pipeline` first."
        )
    pooled = pd.read_parquet(pooled_path)
    log.info("Loaded pooled features: %s rows, %s tickers.", len(pooled), pooled["ticker"].nunique())

    # ── 2. Add HMM regime labels ────────────────────────────────────────
    pooled = _add_regime_column(pooled, cfg)
    log.info("Regime distribution: %s", pooled["regime"].value_counts().to_dict())

    # ── 3. Derive feature list ──────────────────────────────────────────
    feature_cols = _get_feature_columns(pooled)
    log.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    # ── 4. Optional Optuna tuning ────────────────────────────────────────
    if tune:
        log.info("Starting Optuna hyperparameter search (%d trials)...", 50)
        best_params = tune_hyperparameters(pooled, cfg, feature_cols, n_trials=50)
        _save_config(cfg)
        log.info("Best params saved to config.yaml: %s", best_params)

    # ── 5. Walk-forward cross-validation ────────────────────────────────
    wf_df, metrics = run_walk_forward(pooled, cfg, feature_cols)

    # Persist walk-forward results
    _wf_results_path().parent.mkdir(parents=True, exist_ok=True)
    wf_df.to_parquet(_wf_results_path())
    log.info("Walk-forward results saved: %d rows.", len(wf_df))

    # ── 6. Retrain final model on full dataset ───────────────────────────
    X_full = pooled[feature_cols]
    y_full = pooled["label"]

    final_model = XGBoostModel(cfg)
    final_model.train(X_full, y_full)
    final_model.save(_final_model_path())
    log.info("Final model saved to %s.", _final_model_path())

    # ── 7. SHAP feature importance ───────────────────────────────────────
    log.info("Computing SHAP importance...")
    shap_df = compute_shap_importance(final_model, X_full)
    shap_df.to_parquet(_shap_path(), index=False)
    log.info("SHAP importance saved.")

    # ── 8. Persist metrics ───────────────────────────────────────────────
    with open(_wf_metrics_path(), "w") as fh:
        json.dump(metrics, fh, indent=2)
    log.info("Walk-forward metrics saved.")

    return final_model, wf_df, metrics


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    """Train the XGBoost model with walk-forward CV, then save artefacts."""
    parser = argparse.ArgumentParser(description="Train the XGBoost signal model.")
    parser.add_argument(
        "--tune", action="store_true",
        help="Run Optuna hyperparameter search (50 trials) before training.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    cfg = _load_config()

    final_model, wf_df, metrics = run_training_pipeline(cfg, tune=args.tune)

    print("\n--- Walk-Forward Results ---")
    print(f"  Folds          : {metrics['n_folds']}")
    print(f"  Test rows      : {metrics['total_test_rows']:,}")
    print(f"  Accuracy       : {metrics['accuracy']:.3f}")
    print(f"  F1 (weighted)  : {metrics['f1_weighted']:.3f}")
    print(f"  F1 short/flat/long: {metrics['f1_short']:.3f} / {metrics['f1_flat']:.3f} / {metrics['f1_long']:.3f}")
    print(f"  Signal hit rate: {metrics['signal_hit_rate']:.3f}")
    print(f"  Mean sig return: {metrics['mean_signal_return']:.5f}")
    print(f"  Signal Sharpe  : {metrics['sharpe_signal']:.3f}")
    print(f"  % bars signalled: {metrics['pct_signals']:.1%}")

    print("\n  Confusion matrix (rows=actual, cols=pred | short/flat/long):")
    cm = np.array(metrics["confusion_matrix"])
    print(f"  {cm}")

    shap_df = pd.read_parquet(_shap_path())
    print("\n--- Top 10 Features (SHAP) ---")
    print(shap_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
