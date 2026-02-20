"""
hmm_detector.py — Gaussian Hidden Markov Model market-regime classifier.

Trains a 3-state (configurable) GaussianHMM on pooled S&P 500 data using
three regime-sensitive features:
  - 5-day rolling return
  - 20-day realised volatility
  - 14-day ATR as % of price

After fitting, states are labelled by their mean 5-day return:
  1 (bull)   — highest mean return state
  2 (bear)   — lowest  mean return state
  0 (choppy) — all remaining states

The fitted model and state mapping are serialised to
``models/saved/hmm_model.pkl`` via joblib.
"""

import logging
import pathlib
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from hmmlearn import hmm

log = logging.getLogger(__name__)

_ROOT = pathlib.Path(__file__).resolve().parents[1]

# ── Regime code constants ─────────────────────────────────────────────────

CHOPPY = 0
BULL   = 1
BEAR   = 2

_REGIME_NAMES = {CHOPPY: "choppy", BULL: "bull", BEAR: "bear"}


# ── path helpers ──────────────────────────────────────────────────────────

def _load_config() -> dict:
    """Load and return config.yaml as a dict."""
    with open(_ROOT / "config" / "config.yaml") as fh:
        return yaml.safe_load(fh)


def _model_save_path() -> pathlib.Path:
    """Return the default path for the serialised HMM."""
    return _ROOT / "models" / "saved" / "hmm_model.pkl"


def _pooled_path() -> pathlib.Path:
    """Return the path to the pooled-features parquet."""
    return _ROOT / "data" / "processed" / "pooled_features.parquet"


# ── feature extraction ────────────────────────────────────────────────────

_HMM_FEATURES = ["rolling_5d_return", "realised_vol", "atr_pct"]


def extract_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the three HMM input features from a processed ticker DataFrame.

    The 5-day rolling return is computed from the ``close`` column.
    ``realised_vol`` and ``atr_pct`` are expected to be pre-computed columns
    (output of ``features.indicators.calculate_indicators``).

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame with at least ``close``, ``realised_vol``,
        ``atr_pct`` columns and a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Columns: ``rolling_5d_return``, ``realised_vol``, ``atr_pct``.
        Rows with any NaN are dropped.
    """
    feat = pd.DataFrame(index=df.index)
    feat["rolling_5d_return"] = df["close"].pct_change(5)
    feat["realised_vol"]      = df["realised_vol"]
    feat["atr_pct"]           = df["atr_pct"]
    return feat.dropna()


# ── HMMDetector class ─────────────────────────────────────────────────────


class HMMDetector:
    """
    Gaussian HMM wrapper for market-regime classification.

    Parameters
    ----------
    cfg : dict
        Full config dict from config.yaml.  Reads
        ``model.hmm_n_states`` and ``model.hmm_n_iter``.
    """

    def __init__(self, cfg: dict):
        m = cfg["model"]
        self.n_states  = m["hmm_n_states"]
        self.n_iter    = m["hmm_n_iter"]
        self._model: Optional[hmm.GaussianHMM] = None
        self._state_map: Optional[dict[int, int]] = None   # HMM state → regime code

    # ── training ──────────────────────────────────────────────────────────

    def fit(self, pooled: pd.DataFrame) -> "HMMDetector":
        """
        Train the HMM on pooled multi-ticker data.

        Each ticker's rows are treated as a separate observation sequence so
        that inter-ticker transitions do not pollute the transition matrix.

        Parameters
        ----------
        pooled : pd.DataFrame
            Output of ``features.pipeline.run_pipeline``.  Must contain a
            ``ticker`` column plus ``close``, ``realised_vol``, ``atr_pct``.

        Returns
        -------
        self
        """
        sequences: list[np.ndarray] = []
        lengths:   list[int]         = []

        for ticker, group in pooled.groupby("ticker"):
            group_sorted = group.sort_index()
            feat = extract_hmm_features(group_sorted)
            if len(feat) < self.n_states * 2:
                log.debug("Skipping %s — too few rows (%d).", ticker, len(feat))
                continue
            sequences.append(feat.values.astype(float))
            lengths.append(len(feat))

        if not sequences:
            raise ValueError("No usable sequences found in pooled data.")

        X = np.concatenate(sequences, axis=0)
        log.info(
            "Training HMM: %d states, %d total observations from %d tickers.",
            self.n_states, len(X), len(lengths),
        )

        self._model = hmm.GaussianHMM(
            n_components    = self.n_states,
            covariance_type = "full",
            n_iter          = self.n_iter,
            random_state    = 42,
        )
        self._model.fit(X, lengths=lengths)
        log.info("HMM converged: %s", self._model.monitor_.converged)

        self._build_state_map()
        return self

    def _build_state_map(self) -> None:
        """
        Map raw HMM state indices to regime codes.

        Sorted by mean 5-day return (feature 0):
          highest → BULL (1), lowest → BEAR (2), rest → CHOPPY (0).
        """
        mean_returns = self._model.means_[:, 0]   # 5-day return column
        rank = np.argsort(mean_returns)            # ascending: rank[0]=lowest

        state_map: dict[int, int] = {}
        for i, state_idx in enumerate(rank):
            if i == len(rank) - 1:
                state_map[state_idx] = BULL
            elif i == 0:
                state_map[state_idx] = BEAR
            else:
                state_map[state_idx] = CHOPPY

        self._state_map = state_map
        log.info(
            "State mapping (HMM idx → regime):  %s",
            {k: _REGIME_NAMES[v] for k, v in state_map.items()},
        )

    # ── inference ─────────────────────────────────────────────────────────

    def predict_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict regime code for every bar in *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Processed ticker DataFrame (same format as pooled input).

        Returns
        -------
        pd.Series
            Regime code (0=choppy, 1=bull, 2=bear) for each row in *df*.
            Warmup rows where features are NaN are filled with CHOPPY (0).
        """
        if self._model is None or self._state_map is None:
            raise RuntimeError("Model not fitted.  Call fit() or load() first.")

        feat = extract_hmm_features(df)

        if feat.empty:
            return pd.Series(CHOPPY, index=df.index, name="regime", dtype=int)

        raw_states = self._model.predict(feat.values.astype(float))
        regime = pd.Series(
            [self._state_map[s] for s in raw_states],
            index=feat.index,
            name="regime",
            dtype=int,
        )
        # Fill warmup rows (NaN features) with CHOPPY
        return regime.reindex(df.index, fill_value=CHOPPY)

    def add_regime_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of *df* with a ``regime`` column appended.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        df = df.copy()
        df["regime"] = self.predict_regime(df)
        return df

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, path: Optional[pathlib.Path] = None) -> pathlib.Path:
        """
        Serialise the fitted model and state mapping to *path*.

        Parameters
        ----------
        path : pathlib.Path, optional
            Defaults to ``models/saved/hmm_model.pkl``.

        Returns
        -------
        pathlib.Path
            Path where the model was saved.
        """
        path = pathlib.Path(path or _model_save_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model":     self._model,
            "state_map": self._state_map,
            "n_states":  self.n_states,
        }
        joblib.dump(payload, path)
        log.info("HMM saved to %s", path)
        return path

    @classmethod
    def load(cls, cfg: dict, path: Optional[pathlib.Path] = None) -> "HMMDetector":
        """
        Deserialise a saved HMM from *path*.

        Parameters
        ----------
        cfg : dict
            Full config dict (used to initialise the wrapper).
        path : pathlib.Path, optional
            Defaults to ``models/saved/hmm_model.pkl``.

        Returns
        -------
        HMMDetector
        """
        path = pathlib.Path(path or _model_save_path())
        payload = joblib.load(path)

        instance = cls.__new__(cls)
        instance.n_states   = payload["n_states"]
        instance.n_iter     = cfg["model"]["hmm_n_iter"]
        instance._model     = payload["model"]
        instance._state_map = payload["state_map"]

        log.info("HMM loaded from %s", path)
        return instance

    # ── diagnostics ───────────────────────────────────────────────────────

    def state_statistics(self, pooled: pd.DataFrame) -> pd.DataFrame:
        """
        Return a summary table of each HMM state's regime label and statistics.

        Parameters
        ----------
        pooled : pd.DataFrame
            Same pooled data used for training.

        Returns
        -------
        pd.DataFrame
            Index: HMM state index.
            Columns: regime, regime_name, mean_5d_return, mean_realised_vol,
                     mean_atr_pct, n_observations, pct_observations.
        """
        if self._model is None or self._state_map is None:
            raise RuntimeError("Model not fitted.")

        sequences, lengths = [], []
        for _, group in pooled.groupby("ticker"):
            feat = extract_hmm_features(group.sort_index())
            if len(feat) >= self.n_states * 2:
                sequences.append(feat.values.astype(float))
                lengths.append(len(feat))

        X = np.concatenate(sequences, axis=0)
        raw_states = self._model.predict(X, lengths=lengths)
        total = len(raw_states)

        rows = []
        for s in range(self.n_states):
            mask  = raw_states == s
            count = mask.sum()
            rows.append(
                {
                    "hmm_state":       s,
                    "regime":          self._state_map[s],
                    "regime_name":     _REGIME_NAMES[self._state_map[s]],
                    "mean_5d_return":  float(X[mask, 0].mean()) if count else np.nan,
                    "mean_realised_vol": float(X[mask, 1].mean()) if count else np.nan,
                    "mean_atr_pct":    float(X[mask, 2].mean()) if count else np.nan,
                    "n_observations":  int(count),
                    "pct_observations": round(count / total * 100, 1),
                }
            )

        return pd.DataFrame(rows).set_index("hmm_state")


# ── entry point ───────────────────────────────────────────────────────────


def main():
    """Train HMM on pooled features, print state statistics, save model."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    cfg     = _load_config()
    pooled_path = _pooled_path()

    if not pooled_path.exists():
        raise FileNotFoundError(
            "Pooled features not found.  Run `python -m features.pipeline` first."
        )

    pooled = pd.read_parquet(pooled_path)
    print(f"Pooled dataset: {len(pooled):,} rows, {pooled['ticker'].nunique()} tickers")

    detector = HMMDetector(cfg)
    detector.fit(pooled)

    stats = detector.state_statistics(pooled)
    print("\nHMM State Statistics:")
    print(stats.to_string())

    save_path = detector.save()
    print(f"\nModel saved to: {save_path}")

    # Verify by adding regime column to first ticker
    first_ticker = pooled["ticker"].iloc[0]
    sample = pooled[pooled["ticker"] == first_ticker].copy()
    sample = detector.add_regime_column(sample)
    dist   = sample["regime"].map(_REGIME_NAMES).value_counts()
    print(f"\nRegime distribution for {first_ticker}:")
    print(dist.to_string())


if __name__ == "__main__":
    main()
