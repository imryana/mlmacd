"""
downloader.py — Download and validate OHLCV data for S&P 500 tickers.

Uses yfinance for data retrieval.  Supports incremental updates: if a
parquet file already exists for a ticker the downloader only fetches bars
since the last stored date and appends them.

Each ticker is saved to data/raw/equities/{ticker}.parquet.
Failed tickers are logged to data/raw/failed_tickers.txt.
"""

import logging
import pathlib
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
import yaml
from tqdm import tqdm

log = logging.getLogger(__name__)

_ROOT = pathlib.Path(__file__).resolve().parents[1]

# ── path helpers ──────────────────────────────────────────────────────────


def _load_config() -> dict:
    """Load and return config.yaml as a dict."""
    with open(_ROOT / "config" / "config.yaml") as fh:
        return yaml.safe_load(fh)


def _raw_dir() -> pathlib.Path:
    """Return the directory where raw parquet files are stored."""
    return _ROOT / "data" / "raw" / "equities"


def _universe_path() -> pathlib.Path:
    """Return the path to the S&P 500 universe CSV."""
    return _ROOT / "data" / "universe" / "sp500_tickers.csv"


def _failed_path() -> pathlib.Path:
    """Return the path to the failed-tickers log."""
    return _ROOT / "data" / "raw" / "failed_tickers.txt"


# ── download ──────────────────────────────────────────────────────────────


def download_ticker(
    ticker: str,
    start: str,
    end: str,
    existing: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """
    Download OHLCV data for a single ticker via yfinance.

    If *existing* is provided, only bars after its last date are fetched and
    the result is appended, giving incremental update behaviour.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
    start : str
        ISO date string (YYYY-MM-DD) for the earliest requested bar.
    end : str
        ISO date string or ``"today"``.
    existing : pd.DataFrame, optional
        Previously stored OHLCV data for this ticker.

    Returns
    -------
    pd.DataFrame or None
        OHLCV DataFrame with a DatetimeIndex named ``"date"``, or ``None``
        if the download or parse failed.
    """
    if end == "today":
        end = date.today().isoformat()

    fetch_start = start
    if existing is not None and not existing.empty:
        last_date = existing.index.max()
        fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        if fetch_start >= end:
            log.debug("%s is already up to date.", ticker)
            return existing

    try:
        raw = yf.download(
            ticker,
            start=fetch_start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        log.warning("Download failed for %s: %s", ticker, exc)
        return None

    if raw is None or raw.empty:
        log.warning("No data returned for %s.", ticker)
        return None

    # yfinance ≥ 0.2 returns a MultiIndex; flatten to simple column names.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.index = pd.to_datetime(raw.index)
    raw.index.name = "date"
    raw.columns = [c.lower() for c in raw.columns]

    # Keep only standard OHLCV columns (yfinance may include extras).
    ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
    raw = raw[ohlcv_cols]

    if existing is not None and not existing.empty:
        raw = pd.concat([existing, raw])
        raw = raw[~raw.index.duplicated(keep="last")]
        raw.sort_index(inplace=True)

    return raw


# ── validation ────────────────────────────────────────────────────────────


def validate_ticker(df: pd.DataFrame, cfg: dict) -> tuple[bool, str]:
    """
    Validate a downloaded ticker DataFrame against configured quality rules.

    Rules (all sourced from config.yaml):
    - Minimum ``data.min_history_days`` bars.
    - No gap of more than 5 consecutive missing trading days
      (approximated as > 7 calendar days between adjacent rows).
    - All closing prices above ``data.min_price``.
    - All volume values strictly positive.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with a DatetimeIndex.
    cfg : dict
        Full config dict loaded from config.yaml.

    Returns
    -------
    tuple[bool, str]
        ``(True, "")`` if valid; ``(False, reason)`` otherwise.
    """
    min_days = cfg["data"]["min_history_days"]
    min_price = cfg["data"]["min_price"]

    if len(df) < min_days:
        return False, f"Only {len(df)} bars (need {min_days})"

    # Detect runs of > 7 calendar-day gaps between adjacent rows.
    date_idx = pd.DatetimeIndex(df.index)
    if len(date_idx) > 1:
        gaps = (date_idx[1:] - date_idx[:-1]).days
        if (gaps > 7).any():
            return False, "Gap of more than 5 consecutive missing trading days"

    if (df["close"] < min_price).any():
        return False, f"Price dropped below min_price={min_price}"

    if (df["volume"] <= 0).any():
        return False, "Zero or negative volume on at least one bar"

    return True, ""


# ── persistence ───────────────────────────────────────────────────────────


def load_existing(ticker: str) -> Optional[pd.DataFrame]:
    """
    Load the existing parquet file for *ticker*, if present.

    Parameters
    ----------
    ticker : str
        Ticker symbol (used as filename stem).

    Returns
    -------
    pd.DataFrame or None
    """
    path = _raw_dir() / f"{ticker}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


def save_ticker(ticker: str, df: pd.DataFrame) -> pathlib.Path:
    """
    Save a ticker's OHLCV DataFrame as a parquet file.

    Parameters
    ----------
    ticker : str
        Ticker symbol (used as filename stem).
    df : pd.DataFrame
        Validated OHLCV data.

    Returns
    -------
    pathlib.Path
        Path to the saved file.
    """
    path = _raw_dir() / f"{ticker}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


# ── orchestration ─────────────────────────────────────────────────────────


def download_all(
    tickers: list[str], cfg: dict
) -> tuple[list[str], list[str]]:
    """
    Download and validate OHLCV data for a list of tickers.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to process.
    cfg : dict
        Full config dict from config.yaml.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(succeeded, failed)`` — lists of ticker symbols.
    """
    _raw_dir().mkdir(parents=True, exist_ok=True)

    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"]
    succeeded: list[str] = []
    failed: list[str] = []

    for ticker in tqdm(tickers, desc="Downloading tickers", unit="ticker"):
        existing = load_existing(ticker)
        df = download_ticker(ticker, start, end, existing=existing)

        if df is None:
            failed.append(ticker)
            continue

        ok, reason = validate_ticker(df, cfg)
        if not ok:
            log.warning("Validation failed for %s: %s", ticker, reason)
            failed.append(ticker)
            continue

        save_ticker(ticker, df)
        succeeded.append(ticker)

    failed_path = _failed_path()
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(failed_path, "w") as fh:
        fh.write("\n".join(failed))

    return succeeded, failed


# ── entry point ───────────────────────────────────────────────────────────


def main():
    """Download OHLCV data for all S&P 500 tickers and report results."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    cfg = _load_config()

    universe_path = _universe_path()
    if not universe_path.exists():
        raise FileNotFoundError(
            "Universe CSV not found. Run `python -m ingestion.universe` first."
        )

    universe = pd.read_csv(universe_path)
    tickers = universe["ticker"].tolist()

    print(f"Starting download for {len(tickers)} tickers...")
    succeeded, failed = download_all(tickers, cfg)

    print(f"\n  Succeeded : {len(succeeded)}")
    print(f"  Failed    : {len(failed)}")
    if failed:
        preview = ", ".join(failed[:20])
        suffix = " ..." if len(failed) > 20 else ""
        print(f"  Failed tickers: {preview}{suffix}")
        print(f"  Full list written to: {_failed_path()}")


if __name__ == "__main__":
    main()
