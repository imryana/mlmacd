"""
indicators.py — Technical indicator calculation for a single OHLCV DataFrame.

All indicators are implemented natively with numpy / pandas (no pandas-ta
dependency) so the code runs on any Python 3.11+ environment.

Input:  OHLCV DataFrame with columns [open, high, low, close, volume]
        and a DatetimeIndex.
Output: Same DataFrame with all feature columns appended in-place (copy
        returned — original is not mutated).
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── EMA helpers ───────────────────────────────────────────────────────────


def _ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential moving average using pandas ewm (span-based, min_periods=period).

    Parameters
    ----------
    series : pd.Series
    period : int

    Returns
    -------
    pd.Series
    """
    return series.ewm(span=period, min_periods=period, adjust=False).mean()


def _rma(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder's smoothed moving average (RMA / SMMA), used by RSI and ATR.

    Equivalent to EMA with alpha = 1/period.
    """
    return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


# ── MACD ─────────────────────────────────────────────────────────────────


def _add_macd(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Add MACD features to *df*.

    New columns
    -----------
    macd_line, macd_signal, macd_histogram,
    macd_hist_slope_1, macd_hist_slope_3,
    macd_crossover  (-1 / 0 / 1),
    macd_divergence (-1 / 0 / 1)
    """
    fast = cfg["indicators"]["macd_fast"]
    slow = cfg["indicators"]["macd_slow"]
    sig  = cfg["indicators"]["macd_signal"]

    close = df["close"]
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)

    macd_line   = ema_fast - ema_slow
    macd_signal = _ema(macd_line, sig)
    macd_hist   = macd_line - macd_signal

    df["macd_line"]      = macd_line
    df["macd_signal"]    = macd_signal
    df["macd_histogram"] = macd_hist

    df["macd_hist_slope_1"] = macd_hist.diff(1)
    df["macd_hist_slope_3"] = macd_hist.diff(3)

    # Crossover: 1 if bullish cross in last 3 bars, -1 bearish, 0 none
    cross = pd.Series(0, index=df.index)
    bull_cross = (macd_line > macd_signal) & (macd_line.shift(1) <= macd_signal.shift(1))
    bear_cross = (macd_line < macd_signal) & (macd_line.shift(1) >= macd_signal.shift(1))
    cross[bull_cross] = 1
    cross[bear_cross] = -1
    df["macd_crossover"] = (
        cross.rolling(3, min_periods=1)
        .apply(lambda x: x[x != 0].iloc[-1] if (x != 0).any() else 0, raw=False)
        .astype(int)
    )

    # Divergence: price at rolling 10-bar low but MACD not → bullish, vice versa
    window = 10
    roll_close_min = close.rolling(window, min_periods=window).min()
    roll_macd_min  = macd_line.rolling(window, min_periods=window).min()
    roll_close_max = close.rolling(window, min_periods=window).max()
    roll_macd_max  = macd_line.rolling(window, min_periods=window).max()

    div = pd.Series(0, index=df.index)
    div[close <= roll_close_min * 1.001] = np.where(
        macd_line[close <= roll_close_min * 1.001] > roll_macd_min[close <= roll_close_min * 1.001],
        1, 0
    )
    bear_div_mask = close >= roll_close_max * 0.999
    div[bear_div_mask] = np.where(
        macd_line[bear_div_mask] < roll_macd_max[bear_div_mask],
        -1, div[bear_div_mask]
    )
    df["macd_divergence"] = div

    # ── Crossover quality features ─────────────────────────────────────────

    # Bars since last raw crossover event (0 = crossover happened this bar)
    bar_idx = pd.Series(np.arange(len(df), dtype=float), index=df.index)
    last_cross_idx = bar_idx.where(cross != 0).ffill()
    df["macd_bars_since_cross"] = (bar_idx - last_cross_idx).fillna(bar_idx)

    # Absolute gap between MACD line and signal (crossover strength, continuous)
    df["macd_cross_strength"] = (macd_line - macd_signal).abs()

    # MACD line position relative to zero (bull/bear territory)
    df["macd_above_zero"] = (macd_line > 0).astype(int)

    # Histogram acceleration: second derivative (is momentum expanding or fading?)
    df["macd_hist_acceleration"] = df["macd_hist_slope_1"].diff(1)

    # Zero-line crossover: MACD line crossing zero (trend-change confirmation)
    zero_cross = pd.Series(0, index=df.index)
    bull_zero = (macd_line > 0) & (macd_line.shift(1) <= 0)
    bear_zero = (macd_line < 0) & (macd_line.shift(1) >= 0)
    zero_cross[bull_zero] = 1
    zero_cross[bear_zero] = -1
    df["macd_zero_cross"] = (
        zero_cross.rolling(3, min_periods=1)
        .apply(lambda x: x[x != 0].iloc[-1] if (x != 0).any() else 0, raw=False)
        .astype(int)
    )

    # Weekly MACD trend: is weekly MACD line above its signal line?
    # Resampled to Friday-close weeks so dates align with trading days.
    weekly_close = close.resample("W-FRI").last()
    w_ema_fast = _ema(weekly_close, fast)
    w_ema_slow = _ema(weekly_close, slow)
    w_macd     = w_ema_fast - w_ema_slow
    w_signal   = _ema(w_macd, sig)
    w_bull     = (w_macd > w_signal).astype(int)
    df["macd_weekly_bull"] = (
        w_bull.reindex(df.index, method="ffill").fillna(0).astype(int)
    )

    return df


# ── RSI ───────────────────────────────────────────────────────────────────


def _add_rsi(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Add RSI features to *df*.

    New columns: rsi, rsi_slope, rsi_overbought, rsi_oversold
    """
    period    = cfg["indicators"]["rsi_period"]
    overbought = cfg["indicators"]["rsi_overbought"]
    oversold   = cfg["indicators"]["rsi_oversold"]

    delta  = df["close"].diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)

    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    df["rsi"]          = rsi
    df["rsi_slope"]    = rsi.diff(3)
    df["rsi_overbought"] = (rsi > overbought).astype(int)
    df["rsi_oversold"]   = (rsi < oversold).astype(int)

    return df


# ── ADX ───────────────────────────────────────────────────────────────────


def _add_adx(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Add ADX / +DI / -DI features to *df*.

    New columns: adx, adx_plus_di, adx_minus_di, adx_trending
    """
    period    = cfg["indicators"]["adx_period"]
    min_trend = cfg["indicators"]["adx_min_trend"]

    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move   = high.diff()
    down_move = -low.diff()

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s  = _rma(pd.Series(plus_dm,  index=df.index), period)
    minus_dm_s = _rma(pd.Series(minus_dm, index=df.index), period)
    atr_s      = _rma(tr, period)

    plus_di  = 100 * plus_dm_s  / atr_s.replace(0, np.nan)
    minus_di = 100 * minus_dm_s / atr_s.replace(0, np.nan)

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _rma(dx, period)

    df["adx"]          = adx
    df["adx_plus_di"]  = plus_di
    df["adx_minus_di"] = minus_di
    df["adx_trending"] = (adx > min_trend).astype(int)

    return df


# ── EMA ───────────────────────────────────────────────────────────────────


def _add_ema(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Add EMA-based features to *df*.

    New columns: ema_fast, ema_slow, price_ema_slow_ratio, price_ema_slow_pct,
                 golden_cross, death_cross, above_ema_slow
    """
    fast = cfg["indicators"]["ema_fast"]   # 50
    slow = cfg["indicators"]["ema_slow"]   # 200

    close    = df["close"]
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)

    df["ema_fast"] = ema_fast
    df["ema_slow"] = ema_slow
    df["price_ema_slow_ratio"] = close / ema_slow
    df["price_ema_slow_pct"]   = (close - ema_slow) / ema_slow * 100

    # Golden / death cross: EMA-50 crossed EMA-200 within last 5 bars
    cross_above = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
    cross_below = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))

    df["golden_cross"] = cross_above.rolling(5, min_periods=1).max().astype(int)
    df["death_cross"]  = cross_below.rolling(5, min_periods=1).max().astype(int)
    df["above_ema_slow"] = (close > ema_slow).astype(int)

    return df


# ── Volatility ────────────────────────────────────────────────────────────


def _add_volatility(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Add ATR, Bollinger Band width, and realised volatility features.

    New columns: atr, atr_pct, bb_width, realised_vol
    """
    atr_period  = cfg["indicators"]["atr_period"]
    bb_period   = cfg["indicators"]["bb_period"]
    bb_std_mult = cfg["indicators"]["bb_std"]
    rv_period   = cfg["indicators"]["realised_vol_period"]

    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    # ATR
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = _rma(tr, atr_period)

    df["atr"]     = atr
    df["atr_pct"] = atr / close * 100

    # Bollinger Bands
    bb_mid   = close.rolling(bb_period, min_periods=bb_period).mean()
    bb_std   = close.rolling(bb_period, min_periods=bb_period).std(ddof=0)
    bb_upper = bb_mid + bb_std_mult * bb_std
    bb_lower = bb_mid - bb_std_mult * bb_std
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid

    # Realised volatility (annualised)
    log_ret = np.log(close / close.shift(1))
    df["realised_vol"] = log_ret.rolling(rv_period, min_periods=rv_period).std() * np.sqrt(252)

    return df


# ── Volume ────────────────────────────────────────────────────────────────


def _add_volume(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Add volume-based features to *df*.

    New columns: volume_ratio, obv_slope, volume_confirmation
    """
    vol_ma_period  = cfg["indicators"]["volume_ma_period"]
    obv_slope_per  = cfg["indicators"]["obv_slope_period"]

    volume = df["volume"].astype(float)
    close  = df["close"]

    vol_ma = volume.rolling(vol_ma_period, min_periods=vol_ma_period).mean()
    df["volume_ratio"] = volume / vol_ma.replace(0, np.nan)

    # OBV
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    df["obv_slope"] = obv.diff(obv_slope_per)

    # Volume confirmation: high volume on a MACD crossover day
    has_crossover = df["macd_crossover"].abs() > 0
    df["volume_confirmation"] = ((df["volume_ratio"] > 1.2) & has_crossover).astype(int)

    return df


# ── Market context ────────────────────────────────────────────────────────


def _add_market_context(df: pd.DataFrame, cfg: dict, sector_code: int) -> pd.DataFrame:
    """
    Add market-context and sector features.

    New columns: rolling_return_60, return_autocorr_20, sector
    """
    rr_period    = cfg["indicators"]["rolling_return_period"]  # 60
    ac_period    = cfg["indicators"]["autocorr_period"]        # 20

    close   = df["close"]
    log_ret = np.log(close / close.shift(1))

    df["rolling_return_60"]  = close / close.shift(rr_period) - 1
    df["return_autocorr_20"] = (
        log_ret
        .rolling(ac_period, min_periods=ac_period)
        .apply(lambda x: x.autocorr(lag=1), raw=False)
    )
    df["sector"] = sector_code

    return df


# ── Public API ────────────────────────────────────────────────────────────

#: Ordered list of all feature column names produced by calculate_indicators.
FEATURE_COLUMNS = [
    # MACD
    "macd_line", "macd_signal", "macd_histogram",
    "macd_hist_slope_1", "macd_hist_slope_3",
    "macd_crossover", "macd_divergence",
    # MACD quality / multi-timeframe
    "macd_bars_since_cross", "macd_cross_strength",
    "macd_above_zero", "macd_hist_acceleration",
    "macd_zero_cross", "macd_weekly_bull",
    # RSI
    "rsi", "rsi_slope", "rsi_overbought", "rsi_oversold",
    # ADX
    "adx", "adx_plus_di", "adx_minus_di", "adx_trending",
    # EMA
    "ema_fast", "ema_slow",
    "price_ema_slow_ratio", "price_ema_slow_pct",
    "golden_cross", "death_cross", "above_ema_slow",
    # Volatility
    "atr", "atr_pct", "bb_width", "realised_vol",
    # Volume
    "volume_ratio", "obv_slope", "volume_confirmation",
    # Market context
    "rolling_return_60", "return_autocorr_20", "sector",
    # Macro context (6 new — joined from features.macro)
    "vix_level", "vix_pct_rank", "vix_high",
    "spy_above_200", "spy_rsi", "spy_return_20",
]


def calculate_indicators(
    df: pd.DataFrame,
    cfg: dict,
    sector_code: int = 0,
) -> pd.DataFrame:
    """
    Calculate all technical indicators for a single ticker OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns [open, high, low, close, volume] and a
        DatetimeIndex.  The original is not mutated.
    cfg : dict
        Full config dict loaded from config.yaml.
    sector_code : int, optional
        Label-encoded GICS sector integer (assigned by pipeline).  Default 0.

    Returns
    -------
    pd.DataFrame
        Original columns plus all feature columns listed in FEATURE_COLUMNS.
    """
    df = df.copy()

    # MACD must come before volume (volume_confirmation references macd_crossover)
    df = _add_macd(df, cfg)
    df = _add_rsi(df, cfg)
    df = _add_adx(df, cfg)
    df = _add_ema(df, cfg)
    df = _add_volatility(df, cfg)
    df = _add_volume(df, cfg)
    df = _add_market_context(df, cfg, sector_code)

    log.debug(
        "Indicators calculated: %d rows, %d feature columns.",
        len(df),
        len(FEATURE_COLUMNS),
    )
    return df


# ── entry point ───────────────────────────────────────────────────────────


def main():
    """Load AAPL raw data, compute all indicators, print a summary."""
    import pathlib
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    root = pathlib.Path(__file__).resolve().parents[1]
    with open(root / "config" / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    parquet_path = root / "data" / "raw" / "equities" / "AAPL.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            "AAPL.parquet not found — run `python -m ingestion.downloader` first."
        )

    raw = pd.read_parquet(parquet_path)
    result = calculate_indicators(raw, cfg, sector_code=4)  # 4 = Info Tech

    print(f"Input rows : {len(raw)}")
    print(f"Output rows: {len(result)}")
    print(f"Columns    : {len(result.columns)}")
    print("\nLast row feature values:")
    print(result[FEATURE_COLUMNS].iloc[-1].to_string())

    nan_counts = result[FEATURE_COLUMNS].isna().sum()
    non_zero_nans = nan_counts[nan_counts > 0]
    if not non_zero_nans.empty:
        print("\nNaN counts (warmup period expected):")
        print(non_zero_nans.to_string())


if __name__ == "__main__":
    main()
