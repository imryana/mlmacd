"""
stock_detail.py — Page 2: Individual Stock Deep Dive

Shows a full interactive analysis of a single ticker including:
  - Key metrics bar
  - Candlestick + EMA + trade-level overlay chart
  - MACD chart
  - RSI chart (with overbought/oversold bands)
  - ADX chart with regime colour bands
  - Adjustable trade setup panel
  - SHAP top-5 drivers bar chart
  - Historical signal table
"""

import ast
import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from app.app import load_scan_results, load_ticker_ohlc, load_trade_cards, load_wf_results

_REGIME_COLOURS = {
    0: "rgba(108,117,125,0.15)",
    1: "rgba(40,167,69,0.15)",
    2: "rgba(220,53,69,0.15)",
}


# ── Chart builders ────────────────────────────────────────────────────────


def _price_chart(
    df: pd.DataFrame,
    entry: float | None = None,
    stop: float | None = None,
    target: float | None = None,
    trail_act: float | None = None,
    lookback: int = 120,
) -> go.Figure:
    """
    Build a candlestick chart with EMA overlays and optional trade levels.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV + indicator DataFrame indexed by date.
    entry, stop, target, trail_act : float, optional
        Trade levels drawn as horizontal dashed lines.
    lookback : int
        Number of bars to display.

    Returns
    -------
    go.Figure
    """
    df = df.tail(lookback)
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="Price",
        increasing_line_color="#28a745",
        decreasing_line_color="#dc3545",
    ))

    if "ema_fast" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["ema_fast"], name="EMA-50",
                                 line=dict(color="#007bff", width=1.5)))
    if "ema_slow" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["ema_slow"], name="EMA-200",
                                 line=dict(color="#fd7e14", width=1.5)))

    def _hline(y: float | None, color: str, dash: str, label: str) -> None:
        if y is not None and pd.notna(y):
            fig.add_hline(
                y=y, line_color=color, line_dash=dash, line_width=1.5,
                annotation_text=label, annotation_position="right",
            )

    _hline(entry,    "#007bff", "dash",    f"Entry {entry:.2f}"     if entry    else "")
    _hline(stop,     "#dc3545", "solid",   f"Stop {stop:.2f}"       if stop     else "")
    _hline(target,   "#28a745", "dot",     f"Target {target:.2f}"   if target   else "")
    _hline(trail_act,"#6f42c1", "dashdot", f"Trail {trail_act:.2f}" if trail_act else "")

    # Regime background shading
    if "regime" in df.columns:
        prev  = None
        start = df.index[0]
        for dt, row in df.iterrows():
            reg = int(row["regime"])
            if prev is not None and reg != prev:
                fig.add_vrect(
                    x0=str(start), x1=str(dt),
                    fillcolor=_REGIME_COLOURS.get(prev, "rgba(0,0,0,0)"),
                    opacity=1, layer="below", line_width=0,
                )
                start = dt
            prev = reg
        if prev is not None:
            fig.add_vrect(
                x0=str(start), x1=str(df.index[-1]),
                fillcolor=_REGIME_COLOURS.get(prev, "rgba(0,0,0,0)"),
                opacity=1, layer="below", line_width=0,
            )

    fig.update_layout(
        title="Price", xaxis_rangeslider_visible=False,
        height=420, margin=dict(t=40, b=10),
        legend=dict(orientation="h", y=1.02),
    )
    return fig


def _macd_chart(df: pd.DataFrame, lookback: int = 120) -> go.Figure:
    """Build MACD histogram + lines chart."""
    df  = df.tail(lookback)
    fig = go.Figure()

    if "macd_histogram" in df.columns:
        colours = ["#28a745" if v >= 0 else "#dc3545" for v in df["macd_histogram"]]
        fig.add_trace(go.Bar(x=df.index, y=df["macd_histogram"],
                             name="Histogram", marker_color=colours))
    if "macd_line" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["macd_line"], name="MACD",
                                 line=dict(color="#007bff", width=1.5)))
    # Signal line may be stored as "macd_signal" or "macd_signal_line"
    for col in ("macd_signal_line", "macd_signal"):
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name="Signal",
                                     line=dict(color="#fd7e14", width=1.5)))
            break

    fig.update_layout(title="MACD", height=220, margin=dict(t=40, b=10),
                      legend=dict(orientation="h", y=1.05))
    return fig


def _rsi_chart(df: pd.DataFrame, cfg: dict, lookback: int = 120) -> go.Figure:
    """Build RSI chart with overbought/oversold reference bands."""
    df  = df.tail(lookback)
    ob  = cfg.get("indicators", {}).get("rsi_overbought", 70)
    os_ = cfg.get("indicators", {}).get("rsi_oversold", 30)
    fig = go.Figure()

    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI",
                                 line=dict(color="#007bff", width=1.5),
                                 fill="tozeroy", fillcolor="rgba(0,123,255,0.05)"))
        fig.add_hrect(y0=ob,  y1=100, fillcolor="rgba(220,53,69,0.10)",  line_width=0)
        fig.add_hrect(y0=0,   y1=os_, fillcolor="rgba(40,167,69,0.10)",  line_width=0)
        fig.add_hline(y=ob,  line_color="#dc3545", line_dash="dash", line_width=1)
        fig.add_hline(y=os_, line_color="#28a745", line_dash="dash", line_width=1)

    fig.update_layout(title="RSI", height=220, margin=dict(t=40, b=10),
                      yaxis=dict(range=[0, 100]))
    return fig


def _adx_chart(df: pd.DataFrame, cfg: dict, lookback: int = 120) -> go.Figure:
    """Build ADX + DI chart."""
    df      = df.tail(lookback)
    adx_min = cfg.get("indicators", {}).get("adx_min_trend", 20)
    fig     = go.Figure()

    if "adx" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["adx"], name="ADX",
                                 line=dict(color="#6f42c1", width=2)))
    if "adx_plus_di" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["adx_plus_di"], name="+DI",
                                 line=dict(color="#28a745", width=1.2, dash="dot")))
    if "adx_minus_di" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["adx_minus_di"], name="-DI",
                                 line=dict(color="#dc3545", width=1.2, dash="dot")))

    fig.add_hline(y=adx_min, line_color="gray", line_dash="dash", line_width=1,
                  annotation_text=f"Trend threshold ({adx_min})",
                  annotation_position="right")

    fig.update_layout(title="ADX / DI", height=220, margin=dict(t=40, b=10),
                      legend=dict(orientation="h", y=1.05))
    return fig


def _shap_chart(shap_top5: list[dict]) -> go.Figure | None:
    """Build a horizontal bar chart of top SHAP drivers."""
    if not shap_top5:
        return None
    features = [d.get("feature", "") for d in shap_top5]
    values   = [float(d.get("shap_value", 0.0)) for d in shap_top5]
    colours  = ["#28a745" if v >= 0 else "#dc3545" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=features, orientation="h", marker_color=colours,
    ))
    fig.update_layout(
        title="Top Feature Contributions (SHAP)",
        height=250, margin=dict(t=40, b=10), xaxis_title="SHAP value",
    )
    return fig


# ── Trade card calculator ─────────────────────────────────────────────────


def _calc_trade_card(
    price: float,
    atr: float,
    signal: int,
    cfg: dict,
    stop_mult: float,
    rr: float,
) -> dict:
    """
    Dynamically calculate trade card levels given adjustable parameters.

    Parameters
    ----------
    price, atr : float
        Latest close and ATR.
    signal : int
        1 (long) or -1 (short).
    cfg : dict
    stop_mult, rr : float
        User-adjusted stop multiplier and risk:reward ratio.

    Returns
    -------
    dict
        Computed trade levels and position sizing.
    """
    s  = cfg.get("signals", {})
    pf = cfg.get("portfolio", {})

    buf    = atr * s.get("entry_buffer_atr", 0.25)
    entry  = price - buf if signal == 1 else price + buf
    risk   = atr * stop_mult
    stop   = entry - risk if signal == 1 else entry + risk
    target = entry + risk * rr if signal == 1 else entry - risk * rr

    stop_pct   = (stop   - entry) / entry * 100
    target_pct = (target - entry) / entry * 100

    port_size   = pf.get("size", 50_000)
    risk_amount = port_size * pf.get("risk_per_trade", 0.01)
    units       = risk_amount / risk if risk > 0 else 0.0
    value       = units * entry
    pct         = value / port_size * 100 if port_size else 0.0

    return {
        "entry": entry, "stop": stop, "target": target,
        "stop_pct": stop_pct, "target_pct": target_pct,
        "rr_ratio": abs(target_pct / stop_pct) if stop_pct else 0.0,
        "units": units, "value": value, "pct": pct,
    }


# ── Render ────────────────────────────────────────────────────────────────


def render(cfg: dict) -> None:
    """
    Render the Stock Detail page.

    Parameters
    ----------
    cfg : dict
        Full config dict from config.yaml.
    """
    st.title("Stock Detail")

    scan  = load_scan_results()
    cards = load_trade_cards()

    # Ticker selection
    available = sorted(scan["ticker"].unique().tolist()) if (
        not scan.empty and "ticker" in scan.columns) else []
    default   = st.session_state.get("selected_ticker")
    if default not in available and available:
        default = available[0]

    if available:
        idx    = available.index(default) if default in available else 0
        ticker = st.selectbox("Select Ticker", available, index=idx)
    else:
        ticker = st.text_input("Ticker symbol (manual entry)", value=default or "AAPL").upper()

    if not ticker:
        st.info("Enter or select a ticker to begin.")
        return

    st.session_state["selected_ticker"] = ticker

    # Data loading
    ohlc = load_ticker_ohlc(ticker)
    if ohlc.empty:
        st.warning(
            f"No processed data found for **{ticker}**. "
            "Run `python -m features.pipeline` first."
        )
        return

    latest = ohlc.iloc[-1]

    # Find signal row and card row
    sig_row  = None
    card_row = None
    if not scan.empty and "ticker" in scan.columns:
        m = scan["ticker"] == ticker
        if m.any():
            sig_row = scan[m].iloc[-1]
    if not cards.empty and "ticker" in cards.columns:
        m = cards["ticker"] == ticker
        if m.any():
            card_row = cards[m].iloc[-1]

    signal_val = int(sig_row["signal"]) if sig_row is not None else 0

    # ── Metrics bar ───────────────────────────────────────────────────────
    cols = st.columns(7)
    cols[0].metric("Price",      f"{float(latest.get('close', 0)):.2f}")
    sig_name = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(signal_val, "—")
    cols[1].metric("Signal",     sig_name)
    if sig_row is not None:
        cols[2].metric("Confidence", f"{float(sig_row.get('confidence', 0)):.1%}")
        cols[3].metric("Regime",     str(sig_row.get("regime_name", "—")))
    else:
        cols[2].metric("Confidence", "—")
        cols[3].metric("Regime",     "—")
    cols[4].metric("ADX",       f"{float(latest.get('adx', 0)):.1f}")
    cols[5].metric("RSI",       f"{float(latest.get('rsi', 0)):.1f}")
    cols[6].metric("MACD Hist", f"{float(latest.get('macd_histogram', 0)):.3f}")

    st.markdown("---")

    # ── Adjustable trade parameters ───────────────────────────────────────
    with st.expander("Adjust Trade Parameters", expanded=False):
        c1, c2 = st.columns(2)
        stop_mult = c1.slider(
            "Stop Multiplier (× ATR)", 0.5, 5.0,
            float(cfg.get("signals", {}).get("stop_multiplier", 2.0)), 0.25,
        )
        rr_ratio = c2.slider(
            "Risk:Reward Ratio", 1.0, 5.0,
            float(cfg.get("signals", {}).get("rr_ratio", 2.0)), 0.25,
        )

    price_val = float(latest.get("close", 100))
    atr_val   = float(latest.get("atr", 1))
    sig_for_card = signal_val if signal_val != 0 else 1
    tc = _calc_trade_card(price_val, atr_val, sig_for_card, cfg, stop_mult, rr_ratio)

    # ── Price chart ───────────────────────────────────────────────────────
    trail_act = None
    if signal_val != 0:
        trail_mult = cfg.get("signals", {}).get("trailing_stop_activation_atr", 1.0)
        trail_act  = tc["entry"] + atr_val * trail_mult * (1 if signal_val == 1 else -1)

    st.plotly_chart(
        _price_chart(
            ohlc,
            entry    = tc["entry"]  if signal_val != 0 else None,
            stop     = tc["stop"]   if signal_val != 0 else None,
            target   = tc["target"] if signal_val != 0 else None,
            trail_act= trail_act,
        ),
        use_container_width=True,
    )

    # ── Indicator charts ──────────────────────────────────────────────────
    col_macd, col_rsi = st.columns(2)
    with col_macd:
        st.plotly_chart(_macd_chart(ohlc), use_container_width=True)
    with col_rsi:
        st.plotly_chart(_rsi_chart(ohlc, cfg), use_container_width=True)

    st.plotly_chart(_adx_chart(ohlc, cfg), use_container_width=True)

    st.markdown("---")

    # ── Trade setup panel ─────────────────────────────────────────────────
    st.subheader("Trade Setup")
    if signal_val != 0:
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Direction:** {'LONG' if signal_val == 1 else 'SHORT'}")
        c1.markdown(f"**Limit Entry:** `{tc['entry']:.2f}`")
        c1.markdown(f"**Stop Loss:** `{tc['stop']:.2f}` ({tc['stop_pct']:.1f}%)")
        c1.markdown(f"**Take Profit:** `{tc['target']:.2f}` ({tc['target_pct']:.1f}%)")
        c1.markdown(f"**R:R Ratio:** `{tc['rr_ratio']:.2f}:1`")

        trail_dist = atr_val * cfg.get("signals", {}).get("trailing_stop_distance_atr", 1.5)
        c2.markdown(f"**Trailing Distance:** `{trail_dist:.2f}`")
        c2.markdown(f"**Units:** `{tc['units']:.1f}`")
        c2.markdown(f"**Position Value:** `£{tc['value']:,.0f}`")
        c2.markdown(f"**Portfolio %:** `{tc['pct']:.1f}%`")

        risk_amt = cfg.get("portfolio", {}).get("size", 50_000) * cfg.get("portfolio", {}).get("risk_per_trade", 0.01)
        exp_bars = cfg.get("signals", {}).get("entry_expiry_bars", 2)
        time_bars = cfg.get("signals", {}).get("time_exit_bars", 5)
        c3.markdown(f"**Risk Amount:** `£{risk_amt:.0f}`")
        c3.markdown(f"**ATR:** `{atr_val:.2f}`")
        c3.markdown(f"**Entry Expiry:** `{exp_bars} bars`")
        c3.markdown(f"**Time Exit:** `{time_bars} bars`")
    else:
        st.info("No active signal for this ticker.")

    st.markdown("---")

    # ── SHAP chart ────────────────────────────────────────────────────────
    st.subheader("Model Drivers (SHAP)")
    shap_top5: list[dict] = []
    if card_row is not None and "shap_top5" in card_row.index:
        raw = card_row["shap_top5"]
        try:
            shap_top5 = ast.literal_eval(str(raw)) if raw and str(raw) != "[]" else []
        except Exception:
            shap_top5 = []

    if shap_top5:
        fig_shap = _shap_chart(shap_top5)
        if fig_shap:
            st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.info("SHAP values not available. Run `python -m signals.trade_setup` to compute them.")

    st.markdown("---")

    # ── Signal history ────────────────────────────────────────────────────
    st.subheader("Signal History (last 20 walk-forward bars)")
    wf = load_wf_results()
    if not wf.empty and "ticker" in wf.columns:
        hist = wf[wf["ticker"] == ticker].sort_index(ascending=False).head(20)
        if not hist.empty:
            disp_cols = [c for c in ("signal", "confidence", "actual", "regime", "forward_return")
                         if c in hist.columns]
            hd = hist[disp_cols].copy()
            hd.index = pd.to_datetime(hd.index).strftime("%Y-%m-%d")
            hd.index.name = "Date"
            hd.columns = [c.replace("_", " ").title() for c in hd.columns]
            st.dataframe(hd, use_container_width=True)
        else:
            st.info(f"No walk-forward history found for {ticker}.")
    else:
        st.info("Walk-forward results not found. Run `python -m models.trainer` first.")
