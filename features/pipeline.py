"""
pipeline.py — Orchestrate the full feature-engineering pipeline for all tickers.

For each ticker:
  1. Load raw OHLCV parquet from data/raw/equities/{ticker}.parquet
  2. Run indicators.calculate_indicators()
  3. Run labels.add_labels()
  4. Drop rows with any NaN in feature columns
  5. Save to data/processed/equities/{ticker}.parquet

Also writes a pooled dataset:
  data/processed/pooled_features.parquet
(all tickers concatenated, with a ``ticker`` column retained)
"""

import logging
import pathlib
from typing import Optional

import pandas as pd
import yaml
from tqdm import tqdm

from features.indicators import FEATURE_COLUMNS, calculate_indicators
from features.labels import add_labels
from features.macro import compute_macro_features, download_macro

log = logging.getLogger(__name__)

_ROOT = pathlib.Path(__file__).resolve().parents[1]


# ── path helpers ──────────────────────────────────────────────────────────


def _load_config() -> dict:
    """Load and return config.yaml as a dict."""
    with open(_ROOT / "config" / "config.yaml") as fh:
        return yaml.safe_load(fh)


def _raw_path(ticker: str) -> pathlib.Path:
    """Return the raw parquet path for *ticker*."""
    return _ROOT / "data" / "raw" / "equities" / f"{ticker}.parquet"


def _processed_path(ticker: str) -> pathlib.Path:
    """Return the processed parquet path for *ticker*."""
    return _ROOT / "data" / "processed" / "equities" / f"{ticker}.parquet"


def _pooled_path() -> pathlib.Path:
    """Return the path for the pooled feature parquet."""
    return _ROOT / "data" / "processed" / "pooled_features.parquet"


def _universe_path() -> pathlib.Path:
    """Return the path to the universe CSV."""
    return _ROOT / "data" / "universe" / "sp500_tickers.csv"


# ── sector encoding ───────────────────────────────────────────────────────


def build_sector_map(universe: pd.DataFrame) -> dict[str, int]:
    """
    Build a ticker → sector integer mapping from the universe DataFrame.

    Sector strings are label-encoded alphabetically (stable across runs).

    Parameters
    ----------
    universe : pd.DataFrame
        Output of ingestion.universe.load_universe().

    Returns
    -------
    dict[str, int]
        Mapping from ticker symbol to integer sector code.
    """
    sectors       = universe["sector"].unique()
    sector_codes  = {s: i for i, s in enumerate(sorted(sectors))}
    return {
        row["ticker"]: sector_codes[row["sector"]]
        for _, row in universe.iterrows()
    }


# ── single-ticker pipeline ────────────────────────────────────────────────


def process_ticker(
    ticker: str,
    cfg: dict,
    sector_code: int = 0,
) -> Optional[pd.DataFrame]:
    """
    Run the full feature pipeline for a single ticker.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    cfg : dict
        Full config dict from config.yaml.
    sector_code : int
        Label-encoded GICS sector integer.

    Returns
    -------
    pd.DataFrame or None
        Processed DataFrame, or None if the raw file is missing or processing
        fails.
    """
    raw_path = _raw_path(ticker)
    if not raw_path.exists():
        log.debug("Skipping %s — raw file not found.", ticker)
        return None

    try:
        raw = pd.read_parquet(raw_path)
        df  = calculate_indicators(raw, cfg, sector_code=sector_code)
        df  = add_labels(df, cfg)

        # Drop rows where any feature column is NaN (warmup period).
        # Macro columns are joined at the pooled level so only drop on columns
        # that are already present in the per-ticker DataFrame.
        before = len(df)
        drop_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        df = df.dropna(subset=drop_cols)
        dropped = before - len(df)
        if dropped:
            log.debug("%s: dropped %d NaN rows (warmup).", ticker, dropped)

        if df.empty:
            log.warning("%s: no rows left after NaN drop.", ticker)
            return None

        out_path = _processed_path(ticker)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)

        return df

    except Exception as exc:
        log.warning("Pipeline failed for %s: %s", ticker, exc)
        return None


# ── full pipeline ─────────────────────────────────────────────────────────


def run_pipeline(
    tickers: list[str],
    cfg: dict,
    sector_map: dict[str, int],
) -> tuple[list[str], list[str]]:
    """
    Run the feature pipeline for every ticker in *tickers*.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to process.
    cfg : dict
        Full config dict from config.yaml.
    sector_map : dict[str, int]
        Ticker → sector code mapping.

    Returns
    -------
    tuple[list[str], list[str]]
        (succeeded, failed) lists of ticker symbols.
    """
    succeeded: list[str] = []
    failed:    list[str] = []
    frames:    list[pd.DataFrame] = []

    for ticker in tqdm(tickers, desc="Feature pipeline", unit="ticker"):
        code = sector_map.get(ticker, 0)
        df   = process_ticker(ticker, cfg, sector_code=code)

        if df is None:
            failed.append(ticker)
            continue

        df["ticker"] = ticker
        frames.append(df)
        succeeded.append(ticker)

    if frames:
        pooled = pd.concat(frames, axis=0)
        pooled.sort_values(["ticker", pooled.index.name or "date"], inplace=True)

        # ── Merge macro features by date ──────────────────────────────────
        macro_df = compute_macro_features(cfg)
        if not macro_df.empty:
            macro_cols = list(macro_df.columns)
            pooled = pooled.join(macro_df, how="left")
            pooled[macro_cols] = pooled[macro_cols].ffill()
            log.info("Macro features merged: %s", macro_cols)
        else:
            log.warning("Macro features unavailable — run download_macro() first.")

        pooled_path = _pooled_path()
        pooled_path.parent.mkdir(parents=True, exist_ok=True)
        pooled.to_parquet(pooled_path)
        log.info("Pooled dataset saved: %d rows, %d tickers.", len(pooled), len(succeeded))

    return succeeded, failed


# ── entry point ───────────────────────────────────────────────────────────


def main():
    """Run the feature pipeline for all tickers with a raw parquet file."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    cfg      = _load_config()

    # Fetch / update VIX and SPY macro data first
    log.info("Downloading macro data (VIX + SPY)...")
    download_macro(cfg)

    universe = pd.read_csv(_universe_path())
    sector_map = build_sector_map(universe)

    # Only process tickers that have a downloaded raw file
    raw_dir = _ROOT / "data" / "raw" / "equities"
    available = [p.stem for p in sorted(raw_dir.glob("*.parquet"))]

    ticker_filter = cfg["data"].get("tickers")
    if ticker_filter:
        available = [t for t in available if t in ticker_filter]
        log.info("Ticker filter active: %s", available)

    if not available:
        raise FileNotFoundError(
            "No raw parquet files found. Run `python -m ingestion.downloader` first."
        )

    print(f"Processing {len(available)} tickers...")
    succeeded, failed = run_pipeline(available, cfg, sector_map)

    print(f"\n  Succeeded : {len(succeeded)}")
    print(f"  Failed    : {len(failed)}")
    if failed:
        print(f"  Failed tickers: {', '.join(failed)}")

    if succeeded:
        pooled = pd.read_parquet(_pooled_path())
        print(f"\n  Pooled dataset: {len(pooled):,} rows x {len(pooled.columns)} columns")
        print(f"  Label distribution:")
        print(pooled["label"].value_counts().sort_index().rename({-1: "short", 0: "flat", 1: "long"}).to_string())


if __name__ == "__main__":
    main()
