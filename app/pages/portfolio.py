"""
portfolio.py — Page 3: Portfolio View

Allows the user to enter their current holdings and see:
  - Portfolio summary: total value, unrealised P&L, sector exposure pie chart
  - Model signal overlay for each holding
  - Suggested actions (exit candidates, new opportunities, sector warnings)
  - Regime exposure breakdown
"""

import pathlib
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from app.app import load_scan_results, load_ticker_ohlc, load_universe

_SIGNAL_NAMES = {1: "LONG", -1: "SHORT", 0: "FLAT"}
_REGIME_NAMES = {0: "choppy", 1: "bull", 2: "bear"}


# ── Holdings parser ───────────────────────────────────────────────────────


def _parse_holdings(text: str) -> pd.DataFrame:
    """
    Parse a multi-line holdings string into a DataFrame.

    Expected format (one per line):
        TICKER,SHARES,AVG_COST
    e.g.
        AAPL,100,145.50
        MSFT,50,320.00

    Lines that cannot be parsed are silently skipped.

    Parameters
    ----------
    text : str

    Returns
    -------
    pd.DataFrame
        Columns: ticker, shares, avg_cost
    """
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            ticker   = parts[0].upper()
            shares   = float(parts[1])
            avg_cost = float(parts[2])
            rows.append({"ticker": ticker, "shares": shares, "avg_cost": avg_cost})
        except ValueError:
            continue
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ticker", "shares", "avg_cost"])


# ── Render ────────────────────────────────────────────────────────────────


def render(cfg: dict) -> None:
    """
    Render the Portfolio page.

    Parameters
    ----------
    cfg : dict
        Full config dict from config.yaml.
    """
    st.title("Portfolio View")

    max_sector_exp = cfg.get("portfolio", {}).get("max_sector_exposure", 0.30)

    # ── Holdings input ────────────────────────────────────────────────────
    st.subheader("Enter Holdings")
    st.caption("Format: TICKER, SHARES, AVG_COST  (one per line)")

    default_text = st.session_state.get("portfolio_text", "")
    holdings_text = st.text_area(
        "Holdings",
        value=default_text,
        height=160,
        placeholder="AAPL,100,145.50\nMSFT,50,320.00\nGOOGL,20,175.00",
        label_visibility="collapsed",
    )
    st.session_state["portfolio_text"] = holdings_text

    holdings = _parse_holdings(holdings_text)
    if holdings.empty:
        st.info("Enter your holdings above to see the portfolio view.")
        return

    # ── Enrich with current prices ────────────────────────────────────────
    scan      = load_scan_results()
    universe  = load_universe()

    # Build sector map from universe
    sector_map: dict[str, str] = {}
    if not universe.empty and "ticker" in universe.columns and "sector" in universe.columns:
        sector_map = dict(zip(universe["ticker"], universe["sector"]))

    enriched_rows = []
    for _, row in holdings.iterrows():
        ticker   = row["ticker"]
        shares   = row["shares"]
        avg_cost = row["avg_cost"]

        # Current price from scan or processed OHLC
        cur_price = None
        if not scan.empty and "ticker" in scan.columns:
            m = scan["ticker"] == ticker
            if m.any():
                cur_price = float(scan[m].iloc[-1]["price"])

        if cur_price is None:
            ohlc = load_ticker_ohlc(ticker)
            if not ohlc.empty and "close" in ohlc.columns:
                cur_price = float(ohlc.iloc[-1]["close"])

        if cur_price is None:
            cur_price = avg_cost  # fallback

        cost_basis   = shares * avg_cost
        market_value = shares * cur_price
        unreal_pnl   = market_value - cost_basis
        unreal_pnl_pct = (unreal_pnl / cost_basis * 100) if cost_basis > 0 else 0.0

        # Model signal
        model_signal = 0
        model_conf   = None
        if not scan.empty and "ticker" in scan.columns:
            m = scan["ticker"] == ticker
            if m.any():
                model_signal = int(scan[m].iloc[-1]["signal"])
                model_conf   = float(scan[m].iloc[-1]["confidence"])

        sector = sector_map.get(ticker, "Unknown")
        enriched_rows.append({
            "ticker":         ticker,
            "shares":         shares,
            "avg_cost":       avg_cost,
            "current_price":  cur_price,
            "cost_basis":     cost_basis,
            "market_value":   market_value,
            "unrealised_pnl": unreal_pnl,
            "pnl_pct":        unreal_pnl_pct,
            "signal":         model_signal,
            "signal_name":    _SIGNAL_NAMES.get(model_signal, "—"),
            "confidence":     model_conf,
            "sector":         sector,
        })

    port_df = pd.DataFrame(enriched_rows)

    # ── Summary metrics ───────────────────────────────────────────────────
    total_value  = port_df["market_value"].sum()
    total_cost   = port_df["cost_basis"].sum()
    total_pnl    = port_df["unrealised_pnl"].sum()
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Value",       f"£{total_value:,.0f}")
    c2.metric("Total Cost",        f"£{total_cost:,.0f}")
    c3.metric("Unrealised P&L",    f"£{total_pnl:,.0f}",
              delta=f"{total_pnl_pct:.1f}%")
    c4.metric("Holdings",          len(port_df))

    st.markdown("---")

    col_tbl, col_pie = st.columns([3, 2])

    # ── Holdings table with model overlay ─────────────────────────────────
    with col_tbl:
        st.subheader("Holdings")

        def _row_colour(row):
            sig = row.get("signal", 0)
            if sig == 1:
                bg = "background-color: rgba(40,167,69,0.12)"
            elif sig == -1:
                bg = "background-color: rgba(220,53,69,0.12)"
            else:
                bg = ""
            return [bg] * len(row)

        disp = port_df[[
            "ticker", "shares", "avg_cost", "current_price",
            "market_value", "pnl_pct", "signal_name", "confidence", "sector",
        ]].copy()
        disp = disp.rename(columns={
            "avg_cost":     "Avg Cost",
            "current_price":"Price",
            "market_value": "Value",
            "pnl_pct":      "P&L %",
            "signal_name":  "Signal",
            "confidence":   "Conf",
        })
        for col in ("Avg Cost", "Price", "Value"):
            disp[col] = disp[col].map(lambda x: f"£{x:,.2f}")
        disp["P&L %"] = disp["P&L %"].map(lambda x: f"{x:+.1f}%")
        disp["Conf"]  = disp["Conf"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")

        styled = disp.style.apply(_row_colour, axis=1)
        st.dataframe(styled, use_container_width=True)

    # ── Sector exposure pie ───────────────────────────────────────────────
    with col_pie:
        st.subheader("Sector Exposure")
        sector_vals = (
            port_df.groupby("sector")["market_value"]
            .sum()
            .reset_index()
            .rename(columns={"market_value": "Value"})
        )
        sector_vals["pct"] = sector_vals["Value"] / total_value
        fig_pie = px.pie(
            sector_vals, values="Value", names="sector",
            title="Portfolio by Sector", hole=0.35,
        )
        fig_pie.update_layout(height=340, margin=dict(t=40, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # ── Suggested actions ─────────────────────────────────────────────────
    st.subheader("Suggested Actions")

    exit_cands  = port_df[port_df["signal"] <= 0]   # flat or short while long
    new_opps    = []
    if not scan.empty and "ticker" in scan.columns:
        held_tickers = set(port_df["ticker"])
        new_opps_df  = scan[~scan["ticker"].isin(held_tickers) & (scan["signal"] != 0)]
        new_opps     = new_opps_df.head(5).to_dict("records")

    # Sector overweights
    overweight = sector_vals[sector_vals["pct"] > max_sector_exp]["sector"].tolist()

    col_exit, col_new = st.columns(2)

    with col_exit:
        st.markdown("**Consider Exiting (model signals flat/short):**")
        if exit_cands.empty:
            st.success("All holdings aligned with model signals.")
        else:
            for _, r in exit_cands.iterrows():
                conf_str = "—" if pd.isna(r["confidence"]) else f"{r['confidence']:.0%}"
                st.warning(
                    f"**{r['ticker']}** — Model: {r['signal_name']} ({conf_str})"
                )

    with col_new:
        st.markdown("**New Opportunities (not in portfolio):**")
        if new_opps:
            for opp in new_opps:
                sig = _SIGNAL_NAMES.get(opp.get("signal", 0), "—")
                st.info(
                    f"**{opp['ticker']}** — {sig} "
                    f"({opp.get('confidence', 0):.0%} conf)"
                )
        else:
            st.info("No new signals outside current portfolio.")

    if overweight:
        st.error(
            f"Sector overweight warning: **{', '.join(overweight)}** "
            f"exceed {max_sector_exp:.0%} of portfolio."
        )

    st.markdown("---")

    # ── Regime exposure ───────────────────────────────────────────────────
    st.subheader("Portfolio Regime Exposure")

    regime_vals: dict[str, float] = {"bull": 0.0, "bear": 0.0, "choppy": 0.0}
    if not scan.empty and "ticker" in scan.columns and "regime_name" in scan.columns:
        for _, row in port_df.iterrows():
            m = scan["ticker"] == row["ticker"]
            if m.any():
                reg_name = str(scan[m].iloc[-1].get("regime_name", "choppy"))
                regime_vals[reg_name] = regime_vals.get(reg_name, 0.0) + row["market_value"]

    total_w_regime = sum(regime_vals.values())
    if total_w_regime > 0:
        regime_pct = {k: v / total_value for k, v in regime_vals.items()}
        colour_map = {"bull": "#28a745", "bear": "#dc3545", "choppy": "#6c757d"}
        fig_reg = go.Figure(go.Bar(
            x=list(regime_pct.keys()),
            y=[v * 100 for v in regime_pct.values()],
            marker_color=[colour_map.get(k, "#aaa") for k in regime_pct],
            text=[f"{v:.1%}" for v in regime_pct.values()],
            textposition="auto",
        ))
        fig_reg.update_layout(
            title="Holdings by Regime (% of portfolio)",
            yaxis_title="%", height=280, margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_reg, use_container_width=True)
    else:
        st.info("Regime data unavailable — run the scanner first.")
