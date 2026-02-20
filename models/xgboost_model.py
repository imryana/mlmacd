"""
xgboost_model.py — XGBoost classifier wrapper for the ML signal scanner.

Supports NVIDIA GPU acceleration via XGBoost's native CUDA backend.
Set ``model.use_gpu: true`` in config.yaml to enable (requires CUDA runtime).
XGBoost >= 2.0 uses ``device='cuda'``; the wrapper handles this automatically.
"""

import logging
import pathlib
import warnings
from typing import Optional

# XGBoost GPU training is on CUDA; prediction on CPU pandas DataFrames triggers
# an informational "Falling back to DMatrix" advisory.  Training (the bottleneck)
# still runs fully on GPU, so suppress the advisory globally.
warnings.filterwarnings("ignore", message="Falling back to prediction")

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

log = logging.getLogger(__name__)

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_config() -> dict:
    """Load and return config.yaml as a dict."""
    with open(_ROOT / "config" / "config.yaml") as fh:
        return yaml.safe_load(fh)


class XGBoostModel:
    """
    Thin wrapper around XGBClassifier with signal-generation helpers.

    Parameters
    ----------
    cfg : dict
        Full config dict from config.yaml.  Reads ``model.*`` and
        ``model.use_gpu`` keys.
    """

    def __init__(self, cfg: dict):
        m = cfg["model"]

        # GPU: XGBoost >= 2.0 uses device='cuda'; older uses tree_method='gpu_hist'
        use_gpu = m.get("use_gpu", False)
        xgb_version = tuple(int(x) for x in xgb.__version__.split(".")[:2])

        if use_gpu:
            if xgb_version >= (2, 0):
                device_kwargs = {"device": "cuda", "tree_method": "hist"}
            else:
                device_kwargs = {"tree_method": "gpu_hist"}
            log.info("XGBoost GPU training enabled (device=cuda, xgb %s).", xgb.__version__)
        else:
            device_kwargs = {"tree_method": "hist"}
            log.info("XGBoost CPU training (use_gpu=false).")

        self.model = xgb.XGBClassifier(
            max_depth        = m["xgb_max_depth"],
            learning_rate    = m["xgb_learning_rate"],
            n_estimators     = m["xgb_n_estimators"],
            subsample        = m["xgb_subsample"],
            min_child_weight = m["xgb_min_child_weight"],
            early_stopping_rounds = m["xgb_early_stopping_rounds"],
            objective        = "multi:softprob",
            num_class        = 3,
            eval_metric      = "mlogloss",
            verbosity        = 0,   # suppress XGBoost C++ advisory messages
            **device_kwargs,
        )
        self._feature_names: Optional[list[str]] = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series]    = None,
    ) -> "XGBoostModel":
        """
        Fit the XGBoost classifier.

        XGBoost requires labels in {0, 1, 2}.  Internally we map
        {-1 → 0, 0 → 1, 1 → 2} and reverse on output.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Ternary labels in {-1, 0, 1}.
        X_val, y_val : optional
            Validation set for early stopping.  If omitted, a 10% hold-out
            split from (X, y) is used.

        Returns
        -------
        self
        """
        self._feature_names = list(X.columns)
        y_enc = self._encode_labels(y)

        if X_val is None or y_val is None:
            split   = max(1, int(len(X) * 0.9))
            X_tr, X_vl = X.iloc[:split], X.iloc[split:]
            y_tr, y_vl = y_enc.iloc[:split], y_enc.iloc[split:]
        else:
            X_tr, X_vl = X, X_val
            y_tr, y_vl = y_enc, self._encode_labels(y_val)

        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_vl, y_vl)],
            verbose=False,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return class probabilities for *X*.

        Returns
        -------
        np.ndarray, shape (n_samples, 3)
            Columns correspond to classes [-1, 0, +1].
        """
        return self.model.predict_proba(X)

    def predict_signal(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (signal, confidence) for each row in *X*.

        ``signal`` is in {-1, 0, 1}; ``confidence`` is the max class
        probability.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (signals, confidences), each of shape (n_samples,).
        """
        proba     = self.predict_proba(X)
        class_idx = np.argmax(proba, axis=1)
        signals   = self._decode_labels(class_idx)
        confidence = proba[np.arange(len(proba)), class_idx]
        return signals, confidence

    def save(self, path: pathlib.Path) -> None:
        """
        Serialise the fitted model to *path* using joblib.

        Parameters
        ----------
        path : pathlib.Path
        """
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self._feature_names}, path)
        log.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: pathlib.Path, cfg: dict) -> "XGBoostModel":
        """
        Deserialise a saved model from *path*.

        Parameters
        ----------
        path : pathlib.Path
        cfg : dict

        Returns
        -------
        XGBoostModel
        """
        data = joblib.load(path)
        instance = cls.__new__(cls)
        instance.model = data["model"]
        instance._feature_names = data.get("feature_names")
        log.info("Model loaded from %s", path)
        return instance

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _encode_labels(y: pd.Series) -> pd.Series:
        """Map {-1, 0, 1} → {0, 1, 2} for XGBoost multi-class."""
        return y.map({-1: 0, 0: 1, 1: 2})

    @staticmethod
    def _decode_labels(encoded: np.ndarray) -> np.ndarray:
        """Map {0, 1, 2} → {-1, 0, 1}."""
        mapping = np.array([-1, 0, 1])
        return mapping[encoded]


# ── entry point ───────────────────────────────────────────────────────────


def main():
    """Smoke-test: train on synthetic data and print signal output."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    cfg = _load_config()
    rng = np.random.default_rng(42)
    n   = 600

    X = pd.DataFrame(rng.standard_normal((n, 10)), columns=[f"feat_{i}" for i in range(10)])
    y = pd.Series(rng.choice([-1, 0, 1], size=n))

    model = XGBoostModel(cfg)
    model.train(X, y)

    signals, conf = model.predict_signal(X.iloc[-5:])
    print("Signals    :", signals)
    print("Confidence :", conf.round(3))


if __name__ == "__main__":
    main()
