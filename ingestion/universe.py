"""
universe.py — Fetch and cache the S&P 500 constituent list.

Scrapes the Wikipedia table at:
  https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

Cleans tickers for Yahoo Finance compatibility (dots → dashes) and saves
the result to data/universe/sp500_tickers.csv.
"""

import io
import logging
import pathlib

import pandas as pd
import requests
import yaml

log = logging.getLogger(__name__)

# ── path helpers ──────────────────────────────────────────────────────────

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_config() -> dict:
    """Load and return config.yaml as a dict."""
    with open(_ROOT / "config" / "config.yaml") as fh:
        return yaml.safe_load(fh)


def _universe_path() -> pathlib.Path:
    """Return the absolute path to the universe CSV file."""
    return _ROOT / "data" / "universe" / "sp500_tickers.csv"


# ── core functions ────────────────────────────────────────────────────────


def fetch_sp500_tickers() -> pd.DataFrame:
    """
    Scrape the S&P 500 constituent table from Wikipedia.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, company, sector, sub_industry, date_added.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    log.info("Fetching S&P 500 tickers from %s", url)

    # Wikipedia returns 403 to the default urllib user-agent; spoof a browser.
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    tables = pd.read_html(io.StringIO(response.text), attrs={"id": "constituents"})
    if not tables:
        raise ValueError("Could not find constituents table on Wikipedia page.")

    raw = tables[0]
    raw.columns = [str(c).strip() for c in raw.columns]

    # Wikipedia column names can vary; map the ones we need.
    col_map = {
        "Symbol": "ticker",
        "Security": "company",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry",
        "Date added": "date_added",
    }
    raw = raw.rename(columns=col_map)

    df = raw[["ticker", "company", "sector", "sub_industry", "date_added"]].copy()

    # Replace '.' with '-' so tickers work with yfinance (e.g. BRK.B → BRK-B).
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False).str.strip()

    log.info("Fetched %d tickers.", len(df))
    return df


def save_universe(df: pd.DataFrame) -> pathlib.Path:
    """
    Save the universe DataFrame to the configured CSV path.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`fetch_sp500_tickers`.

    Returns
    -------
    pathlib.Path
        Path to the saved file.
    """
    path = _universe_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Saved universe to %s", path)
    return path


def load_universe() -> pd.DataFrame:
    """
    Load the cached S&P 500 universe CSV.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, company, sector, sub_industry, date_added.

    Raises
    ------
    FileNotFoundError
        If the CSV has not been generated yet.
    """
    path = _universe_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Universe file not found at {path}. "
            "Run `python -m ingestion.universe` first."
        )
    return pd.read_csv(path)


# ── entry point ───────────────────────────────────────────────────────────


def main():
    """Fetch S&P 500 tickers, save to CSV, and print a sector breakdown."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    df = fetch_sp500_tickers()
    save_universe(df)

    print(f"\nOK — {len(df)} tickers saved to {_universe_path()}")
    print("\nSector breakdown:")
    breakdown = (
        df.groupby("sector")["ticker"]
        .count()
        .sort_values(ascending=False)
        .rename("count")
    )
    print(breakdown.to_string())


if __name__ == "__main__":
    main()
