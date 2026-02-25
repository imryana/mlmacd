"""
macro.py — Market-wide (macro) feature engineering.

Downloads VIX and SPY via yfinance (incremental, like the main downloader)
and computes 6 market-context features indexed by date:

  vix_level       — raw VIX close
  vix_pct_rank    — rolling 252-day percentile of VIX (0.0–1.0)
  vix_high        — binary: VIX > 25 (fear threshold)
  spy_above_200   — binary: SPY close > SPY EMA(200)
  spy_rsi         — RSI(14) of SPY
  spy_return_20   — SPY 20-day rolling return

These columns are later joined into the pooled feature dataset by date so that
every stock on a given day sees the same macro environment.
"""

import logging
import pathlib

import numpy as np
import pandas as pd
import yfinance as yf
import yaml

log = logging.getLogger(__name__)

_ROOT = pathlib.Path(__file__).resolve().parents[1]

# ── Path helpers ──────────────────────────────────────────────────────────


def _macro_dir() -> pathlib.Path:
    """Return the directory for macro raw parquets."""
    return _ROOT / "data" / "raw" / "macro"


def _vix_path() -> pathlib.Path:
    return _macro_dir() / "vix.parquet"


def _spy_path() -> pathlib.Path:
    return _macro_dir() / "spy.parquet"


# ── EMA / RSI helpers (self-contained, no cfg dependency) ─────────────────


def _ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (span-based)."""
    return series.ewm(span=period, min_periods=period, adjust=False).mean()


def _rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder smoothed MA (alpha = 1/period), used by RSI."""
    return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    """RSI(period) for a price series."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── Download ──────────────────────────────────────────────────────────────


def download_macro(cfg: dict) -> None:
    """
    Download (or incrementally update) VIX and SPY raw parquets.

    Both tickers bypass normal downloader validation:
    - ^VIX has no trading volume.
    - They are market indices, not individual equities.

    Parameters
    ----------
    cfg : dict
        Full config dict from config.yaml.  Reads ``macro.vix_ticker``,
        ``macro.spy_ticker``, ``data.start_date``, and ``data.end_date``.
    """
    macro_cfg  = cfg["macro"]
    start_date = cfg["data"]["start_date"]
    end_date   = cfg["data"].get("end_date", "today")
    if end_date == "today":
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    _macro_dir().mkdir(parents=True, exist_ok=True)

    for ticker, path in [
        (macro_cfg["vix_ticker"], _vix_path()),
        (macro_cfg["spy_ticker"], _spy_path()),
    ]:
        if path.exists():
            existing = pd.read_parquet(path)
            last_date = existing.index.max()
            fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if fetch_start >= end_date:
                log.info("%s is up to date (last=%s).", ticker, last_date.date())
                continue
        else:
            existing    = None
            fetch_start = start_date

        log.info("Downloading %s from %s to %s...", ticker, fetch_start, end_date)
        try:
            raw = yf.download(
                ticker,
                start=fetch_start,
                end=end_date,
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                log.warning("%s: no data returned.", ticker)
                continue

            # Flatten multi-level columns if present (yfinance >= 0.2)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0).str.lower()
            else:
                raw.columns = raw.columns.str.lower()

            raw.index = pd.to_datetime(raw.index).tz_localize(None)
            raw = raw[["close"]].copy()

            if existing is not None:
                raw = pd.concat([existing, raw])
                raw = raw[~raw.index.duplicated(keep="last")]
                raw.sort_index(inplace=True)

            raw.to_parquet(path)
            log.info("%s saved: %d rows.", ticker, len(raw))

        except Exception as exc:
            log.error("Failed to download %s: %s", ticker, exc)


# ── Feature computation ───────────────────────────────────────────────────


def compute_macro_features(cfg: dict) -> pd.DataFrame:
    """
    Compute 6 market-wide features from the stored VIX and SPY parquets.

    Parameters
    ----------
    cfg : dict
        Full config dict from config.yaml.  Reads ``macro.*`` keys.

    Returns
    -------
    pd.DataFrame
        Date-indexed DataFrame with columns:
        vix_level, vix_pct_rank, vix_high,
        spy_above_200, spy_rsi, spy_return_20.

        Returns an empty DataFrame if the raw files are missing.
    """
    if not _vix_path().exists() or not _spy_path().exists():
        log.warning(
            "Macro parquets not found — run download_macro() first. "
            "Macro features will be absent."
        )
        return pd.DataFrame()

    macro_cfg = cfg.get("macro", {})

    vix = pd.read_parquet(_vix_path())["close"].rename("vix_close")
    spy = pd.read_parquet(_spy_path())["close"].rename("spy_close")

    rank_window  = macro_cfg.get("vix_pct_rank_window", 252)
    rsi_period   = macro_cfg.get("spy_rsi_period",        14)
    ema_period   = macro_cfg.get("spy_ema_period",        200)
    ret_period   = macro_cfg.get("spy_return_period",     20)

    # ── VIX features ──────────────────────────────────────────────────────
    vix_level = vix

    # Rolling percentile rank (0–1): fraction of past year where VIX was lower
    def _pct_rank(arr: np.ndarray) -> float:
        if len(arr) == 0:
            return np.nan
        current = arr[-1]
        return float((arr[:-1] < current).sum() / max(len(arr) - 1, 1))

    vix_pct_rank = (
        vix
        .rolling(rank_window, min_periods=rank_window // 2)
        .apply(_pct_rank, raw=True)
    )

    vix_high = (vix > 25).astype(int)

    # ── SPY features ──────────────────────────────────────────────────────
    spy_ema200     = _ema(spy, ema_period)
    spy_above_200  = (spy > spy_ema200).astype(int)
    spy_rsi        = _rsi(spy, rsi_period)
    spy_return_20  = spy / spy.shift(ret_period) - 1

    # ── Combine on a shared date index ───────────────────────────────────
    macro_df = pd.concat(
        [
            vix_level.rename("vix_level"),
            vix_pct_rank.rename("vix_pct_rank"),
            vix_high.rename("vix_high"),
            spy_above_200.rename("spy_above_200"),
            spy_rsi.rename("spy_rsi"),
            spy_return_20.rename("spy_return_20"),
        ],
        axis=1,
    )
    macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None)
    macro_df.sort_index(inplace=True)
    return macro_df


# ── Entry point ───────────────────────────────────────────────────────────


def main():
    """Download macro data and print a sample of computed features."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    with open(_ROOT / "config" / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    download_macro(cfg)
    macro_df = compute_macro_features(cfg)

    if macro_df.empty:
        print("No macro data available.")
        return

    print(f"Macro features: {len(macro_df)} rows")
    print(macro_df.tail(5).to_string())


if __name__ == "__main__":
    main()
