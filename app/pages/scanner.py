"""
scanner.py — Page 1: S&P 500 Scanner

Displays the latest scan results with filtering, conditional formatting,
summary metrics, and CSV / Excel export.
"""

import io
import pathlib
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from app.app import load_scan_results, load_trade_cards


# ── Colour helpers ────────────────────────────────────────────────────────

_SIGNAL_COLOUR = {"LONG": "#28a745", "SHORT": "#dc3545", "FLAT": "#6c757d"}
_REGIME_COLOUR = {"bull": "#28a745", "bear": "#dc3545", "choppy": "#6c757d"}


def _style_table(df: pd.DataFrame) -> "pd.DataFrame.style":
    """Apply green/red row background based on signal direction."""

    def row_colour(row):
        sig = str(row.get("signal_name", row.get("signal", "")))
        if sig in ("LONG", "1"):
            bg = "background-color: rgba(40,167,69,0.12)"
        elif sig in ("SHORT", "-1"):
            bg = "background-color: rgba(220,53,69,0.12)"
        else:
            bg = ""
        return [bg] * len(row)

    return df.style.apply(row_colour, axis=1)


# ── Render ────────────────────────────────────────────────────────────────


def render(cfg: dict) -> None:
    """
    Render the Scanner page.

    Parameters
    ----------
    cfg : dict
        Full config dict from config.yaml.
    """
    st.title("S&P 500 ML Signal Scanner")

    scan = load_scan_results()

    if scan.empty:
        st.warning(
            "No scan results found. Click **Run Scan Now** in the sidebar "
            "or run `python -m signals.scanner` from the command line."
        )
        return

    # ── Header metrics ────────────────────────────────────────────────────
    n_signals = len(scan)
    n_long    = int((scan["signal"] == 1).sum())
    n_short   = int((scan["signal"] == -1).sum())
    avg_conf  = float(scan["confidence"].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Signals", n_signals)
    c2.metric("Long", n_long, delta=None)
    c3.metric("Short", n_short, delta=None)
    c4.metric("Avg Confidence", f"{avg_conf:.1%}")

    st.markdown("---")

    # ── Filter controls ───────────────────────────────────────────────────
    with st.expander("Filters", expanded=True):
        col1, col2, col3, col4, col5 = st.columns(5)

        all_sectors = sorted(scan["sector"].dropna().unique().tolist()) if "sector" in scan.columns else []
        sel_sectors = col1.multiselect("Sector", all_sectors, default=all_sectors)

        direction_options = ["All", "Long", "Short"]
        sel_direction = col2.selectbox("Direction", direction_options)

        conf_min = float(scan["confidence"].min())
        conf_max = float(scan["confidence"].max())
        sel_conf = col3.slider(
            "Min Confidence",
            min_value=round(conf_min, 2),
            max_value=1.0,
            value=round(conf_min, 2),
            step=0.01,
            format="%.2f",
        )

        adx_min_default = cfg.get("indicators", {}).get("adx_min_trend", 20)
        sel_adx = col4.slider("Min ADX", 10, 60, int(adx_min_default))

        regime_options = ["All", "bull", "bear", "choppy"]
        sel_regime = col5.selectbox("Regime", regime_options)

    # ── Apply filters ─────────────────────────────────────────────────────
    filtered = scan.copy()
    if sel_sectors and "sector" in filtered.columns:
        filtered = filtered[filtered["sector"].isin(sel_sectors)]
    if sel_direction == "Long":
        filtered = filtered[filtered["signal"] == 1]
    elif sel_direction == "Short":
        filtered = filtered[filtered["signal"] == -1]
    if "confidence" in filtered.columns:
        filtered = filtered[filtered["confidence"] >= sel_conf]
    if "adx" in filtered.columns:
        filtered = filtered[filtered["adx"] >= sel_adx]
    if sel_regime != "All" and "regime_name" in filtered.columns:
        filtered = filtered[filtered["regime_name"] == sel_regime]

    st.caption(f"Showing {len(filtered)} of {n_signals} signals")

    # ── Load trade cards for entry/stop/target ────────────────────────────
    cards = load_trade_cards()
    if not cards.empty and "ticker" in cards.columns:
        cards_map = cards.set_index("ticker")[
            [c for c in ("limit_entry", "stop_loss", "take_profit", "rr_ratio") if c in cards.columns]
        ]
        filtered = filtered.join(cards_map, on="ticker", how="left")

    # ── Build display table ───────────────────────────────────────────────
    display_cols = [c for c in (
        "ticker", "company", "sector", "signal_name", "confidence",
        "regime_name", "adx", "rsi", "macd_histogram", "price",
        "limit_entry", "stop_loss", "take_profit", "rr_ratio",
    ) if c in filtered.columns]

    display_df = filtered[display_cols].copy()

    # Format numbers
    pct_cols  = ("confidence",)
    num2_cols = ("adx", "rsi", "macd_histogram", "price",
                 "limit_entry", "stop_loss", "take_profit", "rr_ratio")
    for col in pct_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"{x:.1%}")
    for col in num2_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(
                lambda x: f"{x:.2f}" if pd.notna(x) else "—"
            )

    # Rename for display
    col_rename = {
        "signal_name":    "Signal",
        "confidence":     "Confidence",
        "regime_name":    "Regime",
        "macd_histogram": "MACD Hist",
        "limit_entry":    "Entry",
        "stop_loss":      "Stop",
        "take_profit":    "Target",
        "rr_ratio":       "R:R",
    }
    display_df = display_df.rename(columns={k: v for k, v in col_rename.items() if k in display_df.columns})
    display_df.columns = [c.upper() if c in ("ticker", "company", "sector", "adx", "rsi", "price") else c
                          for c in display_df.columns]

    styled = _style_table(display_df)
    st.dataframe(styled, use_container_width=True, height=420)

    # ── Ticker selector for detail page ──────────────────────────────────
    if not filtered.empty:
        tickers = filtered["ticker"].tolist() if "ticker" in filtered.columns else []
        if tickers:
            sel_ticker = st.selectbox("Open ticker in Stock Detail →", ["(select)"] + tickers)
            if sel_ticker != "(select)":
                st.session_state["selected_ticker"] = sel_ticker
                st.session_state["page"] = "Stock Detail"
                st.rerun()

    st.markdown("---")

    # ── Summary: sector distribution bar chart ────────────────────────────
    if "sector" in filtered.columns and not filtered.empty:
        st.subheader("Sector Distribution")
        sector_counts = (
            filtered.groupby(["sector", "signal_name"])
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            sector_counts,
            x="sector",
            y="count",
            color="signal_name",
            color_discrete_map={"LONG": "#28a745", "SHORT": "#dc3545"},
            labels={"sector": "Sector", "count": "Signals", "signal_name": "Direction"},
            title="Signals by Sector",
        )
        fig.update_layout(height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # ── Export buttons ────────────────────────────────────────────────────
    st.markdown("---")
    col_csv, col_xlsx, _ = st.columns([1, 1, 4])

    csv_bytes = filtered.to_csv(index=False).encode()
    col_csv.download_button(
        "Export CSV",
        data=csv_bytes,
        file_name="scanner_signals.csv",
        mime="text/csv",
        use_container_width=True,
    )

    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
        filtered.to_excel(writer, index=False, sheet_name="Signals")
    col_xlsx.download_button(
        "Export Excel",
        data=xlsx_buf.getvalue(),
        file_name="scanner_signals.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
