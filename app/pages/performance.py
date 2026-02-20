"""
performance.py — Page 4: Model Performance & Backtest Results

Displays:
  - Key metric cards (Sharpe, Sortino, Max DD, Win Rate, Profit Factor, Calmar)
  - Equity curve
  - Monthly returns heatmap
  - Breakdown tabs: by sector, regime, direction, exit type
  - Trade distribution histograms + MAE/MFE scatter
  - SHAP feature importance bar chart
  - Walk-forward Sharpe by fold
  - Model metadata
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from app.app import (
    load_backtest_metrics,
    load_shap_importance,
    load_trade_log,
    load_wf_metrics,
    load_wf_results,
)

_REGIME_NAMES = {0: "choppy", 1: "bull", 2: "bear"}


# ── Chart builders ────────────────────────────────────────────────────────


def _equity_curve_chart(equity_curve: list[dict]) -> go.Figure:
    """Plot cumulative equity curve."""
    dates  = [p["date"]   for p in equity_curve if p["date"]]
    values = [p["equity"] for p in equity_curve if p["date"]]

    fig = go.Figure(go.Scatter(
        x=dates, y=values, mode="lines",
        line=dict(color="#007bff", width=2),
        fill="tozeroy", fillcolor="rgba(0,123,255,0.08)",
        name="Equity",
    ))
    fig.update_layout(
        title="Equity Curve", height=320,
        xaxis_title="Date", yaxis_title="Portfolio Value (£)",
        margin=dict(t=40, b=20),
    )
    return fig


def _monthly_heatmap(monthly_returns: dict[str, float]) -> go.Figure:
    """
    Build a year × month heatmap from a dict of {"YYYY-MM": pnl_pct}.

    Returns
    -------
    go.Figure
    """
    if not monthly_returns:
        return go.Figure()

    records = [
        {"year": int(k[:4]), "month": int(k[5:7]), "pnl": v}
        for k, v in monthly_returns.items()
    ]
    df_m = pd.DataFrame(records)

    years  = sorted(df_m["year"].unique())
    months = list(range(1, 13))
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    z = []
    for yr in years:
        row = []
        for mo in months:
            m = df_m[(df_m["year"] == yr) & (df_m["month"] == mo)]
            row.append(float(m["pnl"].sum()) if not m.empty else None)
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=month_labels,
        y=[str(y) for y in years],
        colorscale=[[0, "#dc3545"], [0.5, "#ffffff"], [1, "#28a745"]],
        zmid=0,
        text=[[f"{v:.1f}%" if v is not None else "" for v in row] for row in z],
        texttemplate="%{text}",
        colorbar=dict(title="P&L %"),
    ))
    fig.update_layout(
        title="Monthly Returns Heatmap", height=max(200, len(years) * 35 + 80),
        margin=dict(t=40, b=20),
    )
    return fig


def _bar_breakdown(data_wr: dict, data_pnl: dict, title: str) -> go.Figure:
    """Grouped bar chart for win-rate and mean P&L by category."""
    labels = list(data_wr.keys())
    wr_vals  = [data_wr[k]  * 100 for k in labels]
    pnl_vals = [data_pnl[k]       for k in labels]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=wr_vals, name="Win Rate %",
        marker_color="#007bff", yaxis="y",
    ))
    fig.add_trace(go.Bar(
        x=labels, y=pnl_vals, name="Mean P&L %",
        marker_color=["#28a745" if v >= 0 else "#dc3545" for v in pnl_vals],
        yaxis="y2",
    ))
    fig.update_layout(
        title=title,
        barmode="group",
        yaxis=dict(title="Win Rate %", range=[0, 110]),
        yaxis2=dict(title="Mean P&L %", overlaying="y", side="right"),
        height=320, margin=dict(t=40, b=20),
        legend=dict(orientation="h"),
    )
    return fig


def _wf_sharpe_chart(wf_df: pd.DataFrame) -> go.Figure:
    """Rolling Sharpe per walk-forward fold."""
    if wf_df.empty or "fold" not in wf_df.columns:
        return go.Figure()

    rows = []
    for fold, grp in wf_df.groupby("fold"):
        rets = (grp["signal"] * grp["forward_return"]).values
        rets = rets[grp["signal"].values != 0]
        if len(rets) >= 2 and rets.std() > 0:
            sharpe = float(rets.mean() / rets.std() * np.sqrt(252))
        else:
            sharpe = 0.0
        rows.append({"fold": int(fold), "sharpe": sharpe})

    df_s = pd.DataFrame(rows).sort_values("fold")
    colours = ["#28a745" if s >= 0 else "#dc3545" for s in df_s["sharpe"]]

    fig = go.Figure(go.Bar(
        x=df_s["fold"].astype(str), y=df_s["sharpe"],
        marker_color=colours, name="Signal Sharpe",
    ))
    fig.add_hline(y=0, line_color="gray", line_dash="dash", line_width=1)
    fig.update_layout(
        title="Walk-Forward Sharpe by Fold",
        xaxis_title="Fold", yaxis_title="Annualised Sharpe",
        height=280, margin=dict(t=40, b=20),
    )
    return fig


# ── Render ────────────────────────────────────────────────────────────────


def render(cfg: dict) -> None:
    """
    Render the Performance page.

    Parameters
    ----------
    cfg : dict
        Full config dict from config.yaml.
    """
    st.title("Model Performance")

    metrics    = load_backtest_metrics()
    wf_metrics = load_wf_metrics()
    shap_df    = load_shap_importance()
    trade_log  = load_trade_log()
    wf_df      = load_wf_results()

    if not metrics or "error" in metrics:
        st.warning(
            "No backtest results found. "
            "Run `python -m backtest.engine` to generate results."
        )
        if metrics.get("error"):
            st.info(metrics["error"])
    else:
        # ── Key metric cards ──────────────────────────────────────────────
        st.subheader("Key Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c5, c6, c7, c8 = st.columns(4)

        c1.metric("Win Rate",        f"{metrics.get('win_rate', 0):.1%}")
        c2.metric("Profit Factor",   f"{metrics.get('profit_factor', 0):.2f}")
        c3.metric("Sharpe Ratio",    f"{metrics.get('sharpe_ratio', 0):.2f}")
        c4.metric("Sortino Ratio",   f"{metrics.get('sortino_ratio', 0):.2f}")
        c5.metric("Max Drawdown",    f"{metrics.get('max_drawdown_pct', 0):.1f}%")
        c6.metric("Calmar Ratio",    f"{metrics.get('calmar_ratio', 0):.2f}")
        c7.metric("Total Trades",    f"{metrics.get('filled_trades', 0):,}")
        c8.metric("Trades / Year",   f"{metrics.get('trades_per_year', 0):.0f}")

        st.markdown("---")

        # ── Equity curve ──────────────────────────────────────────────────
        eq_curve = metrics.get("equity_curve", [])
        if eq_curve:
            st.plotly_chart(_equity_curve_chart(eq_curve), use_container_width=True)

        # ── Monthly heatmap ───────────────────────────────────────────────
        monthly = metrics.get("monthly_returns", {})
        if monthly:
            st.plotly_chart(_monthly_heatmap(monthly), use_container_width=True)

        st.markdown("---")

        # ── Breakdown tabs ────────────────────────────────────────────────
        st.subheader("Performance Breakdown")
        tabs = st.tabs(["By Sector", "By Regime", "By Direction", "By Exit Type"])

        with tabs[0]:
            wr_s  = metrics.get("sector_win_rates", {})
            pnl_s = metrics.get("sector_mean_pnl", {})
            if wr_s:
                st.plotly_chart(_bar_breakdown(wr_s, pnl_s, "Performance by Sector"),
                                use_container_width=True)
            else:
                st.info("No sector breakdown available.")

        with tabs[1]:
            wr_r  = metrics.get("regime_win_rates", {})
            pnl_r = metrics.get("regime_mean_pnl", {})
            if wr_r:
                st.plotly_chart(_bar_breakdown(wr_r, pnl_r, "Performance by Regime"),
                                use_container_width=True)
            else:
                st.info("No regime breakdown available.")

        with tabs[2]:
            dir_wr  = {
                "Long":  metrics.get("long_win_rate", 0),
                "Short": metrics.get("short_win_rate", 0),
            }
            dir_pnl = {
                "Long":  metrics.get("long_mean_pnl", 0),
                "Short": metrics.get("short_mean_pnl", 0),
            }
            st.plotly_chart(_bar_breakdown(dir_wr, dir_pnl, "Long vs Short"),
                            use_container_width=True)

        with tabs[3]:
            ex_wr   = metrics.get("exit_type_win_rates", {})
            ex_cnt  = metrics.get("exit_type_counts", {})
            ex_pnl  = {k: 0.0 for k in ex_wr}  # no per-exit-type mean P&L in metrics
            if ex_wr:
                # Show win rate + count as bar charts
                fig_ex = go.Figure()
                fig_ex.add_trace(go.Bar(
                    x=list(ex_wr.keys()),
                    y=[v * 100 for v in ex_wr.values()],
                    name="Win Rate %",
                    marker_color="#007bff",
                    text=[f"n={ex_cnt.get(k,0)}" for k in ex_wr],
                    textposition="outside",
                ))
                fig_ex.update_layout(
                    title="Exit Type Win Rates", yaxis_title="%",
                    height=300, margin=dict(t=40, b=20),
                )
                st.plotly_chart(fig_ex, use_container_width=True)
            else:
                st.info("No exit type breakdown available.")

        st.markdown("---")

        # ── Trade distributions ───────────────────────────────────────────
        if not trade_log.empty:
            st.subheader("Trade Distributions")
            filled = trade_log[trade_log["exit_type"] != "missed"]

            col_h, col_pnl = st.columns(2)
            with col_h:
                fig_h = px.histogram(filled, x="hold_bars", nbins=20,
                                     title="Hold Duration Distribution",
                                     color_discrete_sequence=["#007bff"])
                fig_h.update_layout(height=280, margin=dict(t=40, b=10))
                st.plotly_chart(fig_h, use_container_width=True)

            with col_pnl:
                fig_p = px.histogram(filled, x="pnl_pct", nbins=40,
                                     title="P&L Distribution (%)",
                                     color_discrete_sequence=["#6f42c1"])
                fig_p.update_layout(height=280, margin=dict(t=40, b=10))
                st.plotly_chart(fig_p, use_container_width=True)

            # MAE vs MFE scatter
            fig_scatter = px.scatter(
                filled, x="mae_pct", y="mfe_pct", color="exit_type",
                title="MAE vs MFE by Exit Type",
                labels={"mae_pct": "MAE %", "mfe_pct": "MFE %"},
                opacity=0.6,
            )
            fig_scatter.update_layout(height=320, margin=dict(t=40, b=20))
            st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # ── SHAP feature importance ───────────────────────────────────────────
    st.subheader("Feature Importance (SHAP)")
    if not shap_df.empty:
        top20 = shap_df.head(20)
        fig_shap = px.bar(
            top20, x="mean_abs_shap", y="feature", orientation="h",
            title="Mean |SHAP| — Top 20 Features",
            color="mean_abs_shap",
            color_continuous_scale=[[0, "#cce5ff"], [1, "#007bff"]],
        )
        fig_shap.update_layout(
            height=500, margin=dict(t=40, b=10),
            yaxis=dict(autorange="reversed"),
            showlegend=False, coloraxis_showscale=False,
        )
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.info("SHAP importance not found. Run `python -m models.trainer` first.")

    st.markdown("---")

    # ── Walk-forward fold Sharpe ──────────────────────────────────────────
    st.subheader("Walk-Forward Consistency")
    if not wf_df.empty:
        st.plotly_chart(_wf_sharpe_chart(wf_df), use_container_width=True)

    # ── Model metadata ────────────────────────────────────────────────────
    if wf_metrics:
        st.subheader("Model Metadata")
        meta_cols = st.columns(4)
        meta_cols[0].metric("WF Folds",       wf_metrics.get("n_folds", "—"))
        meta_cols[1].metric("Test Rows",       f"{wf_metrics.get('total_test_rows', 0):,}")
        meta_cols[2].metric("Accuracy",        f"{wf_metrics.get('accuracy', 0):.1%}")
        meta_cols[3].metric("Signal Sharpe",   f"{wf_metrics.get('sharpe_signal', 0):.2f}")

        with st.expander("XGBoost Hyperparameters"):
            xgb_keys = [k for k in cfg.get("model", {}) if k.startswith("xgb")]
            for k in xgb_keys:
                st.write(f"**{k}**: {cfg['model'][k]}")

        with st.expander("Full Walk-Forward Metrics JSON"):
            st.json(wf_metrics)
