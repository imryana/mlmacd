"""
labels.py — Forward-return label construction for ML training.

Computes an N-bar forward return for each row and assigns a ternary label:
  +1 (long)  — forward return > long_threshold
  -1 (short) — forward return < short_threshold
   0 (flat)  — otherwise

All thresholds come from config.yaml (labels section).
The last *forward_window* rows are dropped because the future close is unknown.
"""

import logging
import pathlib

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_config() -> dict:
    """Load and return config.yaml as a dict."""
    with open(_ROOT / "config" / "config.yaml") as fh:
        return yaml.safe_load(fh)


def add_labels(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Append ``forward_return`` and ``label`` columns to *df*.

    The last ``forward_window`` rows are dropped (future close unavailable).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least a ``close`` column and a DatetimeIndex.
        Usually the output of :func:`features.indicators.calculate_indicators`.
    cfg : dict
        Full config dict from config.yaml.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with two extra columns and the tail rows removed.
    """
    window          = cfg["labels"]["forward_window"]
    long_threshold  = cfg["labels"]["long_threshold"]
    short_threshold = cfg["labels"]["short_threshold"]

    df = df.copy()

    df["forward_return"] = df["close"].shift(-window) / df["close"] - 1

    conditions = [
        df["forward_return"] >  long_threshold,
        df["forward_return"] <  short_threshold,
    ]
    choices = [1, -1]
    df["label"] = np.select(conditions, choices, default=0)

    # Drop the last N rows where the future close is unknown
    df = df.iloc[:-window].copy()

    log.debug(
        "Labels added: %d rows remaining.  Distribution: %s",
        len(df),
        df["label"].value_counts().to_dict(),
    )
    return df


def label_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Return the count and percentage of each label class.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`add_labels`.

    Returns
    -------
    pd.Series
        Index is label value (-1, 0, 1), values are counts.
    """
    counts = df["label"].value_counts().sort_index()
    pcts   = (counts / len(df) * 100).round(1)
    summary = pd.DataFrame({"count": counts, "pct": pcts})
    return summary


# ── entry point ───────────────────────────────────────────────────────────


def main():
    """Compute labels for AAPL and print class distribution."""
    import sys
    sys.path.insert(0, str(_ROOT))

    from features.indicators import calculate_indicators

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    cfg = _load_config()

    parquet_path = _ROOT / "data" / "raw" / "equities" / "AAPL.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            "AAPL.parquet not found — run `python -m ingestion.downloader` first."
        )

    raw       = pd.read_parquet(parquet_path)
    with_inds = calculate_indicators(raw, cfg, sector_code=4)
    labeled   = add_labels(with_inds, cfg)

    print(f"Rows after labeling: {len(labeled)}")
    print(f"\nForward window : {cfg['labels']['forward_window']} bars")
    print(f"Long threshold : > {cfg['labels']['long_threshold']:.1%}")
    print(f"Short threshold: < {cfg['labels']['short_threshold']:.1%}")
    print("\nLabel distribution:")
    print(label_distribution(labeled).to_string())


if __name__ == "__main__":
    main()
