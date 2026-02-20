"""
app.py — Streamlit entry point for the S&P 500 ML Signal Scanner.

Launch with:
    streamlit run app/app.py

Sidebar provides:
  - Navigation between the 5 pages
  - Last scan timestamp
  - Portfolio size input (persisted in session state)
  - "Run Scan Now" button that re-runs the full signal pipeline

Shared data-loading helpers are defined here with @st.cache_data so all
pages share the same cached copies.
"""

import json
import logging
import pathlib
import subprocess
import sys
from datetime import datetime

import pandas as pd
import streamlit as st
import yaml

# ── Ensure ml_scanner/ is on the path ────────────────────────────────────
_APP_DIR    = pathlib.Path(__file__).resolve().parent
_ROOT       = _APP_DIR.parent
sys.path.insert(0, str(_ROOT))

log = logging.getLogger(__name__)

# ── Path constants ────────────────────────────────────────────────────────

_SCAN_PATH      = _ROOT / "data" / "processed" / "latest_scan.parquet"
_CARDS_PATH     = _ROOT / "data" / "processed" / "trade_cards.parquet"
_METRICS_PATH   = _ROOT / "models" / "saved" / "backtest_metrics.json"
_SHAP_PATH      = _ROOT / "models" / "saved" / "shap_importance.parquet"
_WF_PATH        = _ROOT / "models" / "saved" / "wf_results.parquet"
_TRADE_LOG_PATH = _ROOT / "models" / "saved" / "trade_log.parquet"
_WF_METRICS_PATH= _ROOT / "models" / "saved" / "wf_metrics.json"
_CONFIG_PATH    = _ROOT / "config" / "config.yaml"
_UNIVERSE_PATH  = _ROOT / "data" / "universe" / "sp500_tickers.csv"
_PROCESSED_DIR  = _ROOT / "data" / "processed" / "equities"


# ── Shared data loaders ───────────────────────────────────────────────────


@st.cache_data(ttl=300)
def load_scan_results() -> pd.DataFrame:
    """Load the latest scanner output, or an empty DataFrame if not found."""
    if not _SCAN_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(_SCAN_PATH)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_trade_cards() -> pd.DataFrame:
    """Load trade cards, or empty DataFrame if not found."""
    if not _CARDS_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(_CARDS_PATH)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_backtest_metrics() -> dict:
    """Load backtest metrics JSON, or empty dict if not found."""
    if not _METRICS_PATH.exists():
        return {}
    try:
        with open(_METRICS_PATH) as fh:
            return json.load(fh)
    except Exception:
        return {}


@st.cache_data(ttl=300)
def load_wf_metrics() -> dict:
    """Load walk-forward metrics JSON, or empty dict if not found."""
    if not _WF_METRICS_PATH.exists():
        return {}
    try:
        with open(_WF_METRICS_PATH) as fh:
            return json.load(fh)
    except Exception:
        return {}


@st.cache_data(ttl=300)
def load_shap_importance() -> pd.DataFrame:
    """Load SHAP feature importance parquet, or empty DataFrame."""
    if not _SHAP_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(_SHAP_PATH)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_wf_results() -> pd.DataFrame:
    """Load walk-forward results parquet, or empty DataFrame."""
    if not _WF_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(_WF_PATH)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_trade_log() -> pd.DataFrame:
    """Load backtest trade log, or empty DataFrame."""
    if not _TRADE_LOG_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(_TRADE_LOG_PATH)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_config() -> dict:
    """Load config.yaml."""
    try:
        with open(_CONFIG_PATH) as fh:
            return yaml.safe_load(fh)
    except Exception:
        return {}


@st.cache_data(ttl=300)
def load_ticker_ohlc(ticker: str) -> pd.DataFrame:
    """Load processed OHLC + features for a single ticker."""
    path = _PROCESSED_DIR / f"{ticker}.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_universe() -> pd.DataFrame:
    """Load the S&P 500 universe CSV."""
    if not _UNIVERSE_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(_UNIVERSE_PATH)
    except Exception:
        return pd.DataFrame()


# ── Session state defaults ────────────────────────────────────────────────


def _init_session_state(cfg: dict) -> None:
    """Initialise session-state keys that pages rely on."""
    defaults = {
        "page":           "Scanner",
        "selected_ticker": None,
        "portfolio_size": cfg.get("portfolio", {}).get("size", 50_000),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Scan runner ───────────────────────────────────────────────────────────


def _run_scan_pipeline() -> tuple[bool, str]:
    """
    Run the scanner pipeline in a subprocess and return (success, message).

    Runs: downloader (incremental) → features → scanner → trade_setup
    """
    steps = [
        ("Updating data…",       [sys.executable, "-m", "ingestion.downloader"]),
        ("Building features…",   [sys.executable, "-m", "features.pipeline"]),
        ("Running scanner…",     [sys.executable, "-m", "signals.scanner"]),
        ("Building trade cards…",[sys.executable, "-m", "signals.trade_setup"]),
    ]

    for label, cmd in steps:
        result = subprocess.run(
            cmd,
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            return False, f"{label} failed:\n{result.stderr[-500:]}"

    # Invalidate caches
    load_scan_results.clear()
    load_trade_cards.clear()

    return True, "Scan complete."


# ── Sidebar ───────────────────────────────────────────────────────────────


def _build_sidebar(cfg: dict) -> str:
    """
    Render the sidebar and return the selected page name.

    Parameters
    ----------
    cfg : dict
        Full config dict (used for portfolio size default).

    Returns
    -------
    str
        Selected page name.
    """
    with st.sidebar:
        st.markdown("## ML Signal Scanner")
        st.markdown("---")

        page = st.radio(
            "Navigate",
            ["Scanner", "Stock Detail", "Portfolio", "Performance", "Settings"],
            index=["Scanner", "Stock Detail", "Portfolio", "Performance", "Settings"]
            .index(st.session_state.get("page", "Scanner")),
            key="_nav_radio",
        )
        st.session_state["page"] = page

        st.markdown("---")

        # Last scan timestamp
        if _SCAN_PATH.exists():
            mtime = datetime.fromtimestamp(_SCAN_PATH.stat().st_mtime)
            st.caption(f"Last scan: {mtime.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.caption("No scan data found.")

        # Portfolio size
        st.session_state["portfolio_size"] = st.number_input(
            "Portfolio size (£)",
            min_value=1_000,
            max_value=10_000_000,
            value=int(st.session_state["portfolio_size"]),
            step=1_000,
        )

        st.markdown("---")

        # Run scan button
        if st.button("Run Scan Now", use_container_width=True):
            with st.spinner("Running scan pipeline…"):
                ok, msg = _run_scan_pipeline()
            if ok:
                st.success(msg)
            else:
                st.error(msg)
            st.rerun()

    return page


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    """Streamlit app entry point."""
    st.set_page_config(
        page_title="S&P 500 ML Signal Scanner",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    cfg = load_config()
    _init_session_state(cfg)
    page = _build_sidebar(cfg)

    # ── Page routing ──────────────────────────────────────────────────────
    if page == "Scanner":
        from app.pages import scanner as scanner_page
        scanner_page.render(cfg)

    elif page == "Stock Detail":
        from app.pages import stock_detail as detail_page
        detail_page.render(cfg)

    elif page == "Portfolio":
        from app.pages import portfolio as portfolio_page
        portfolio_page.render(cfg)

    elif page == "Performance":
        from app.pages import performance as perf_page
        perf_page.render(cfg)

    elif page == "Settings":
        from app.pages import settings as settings_page
        settings_page.render(cfg)


if __name__ == "__main__":
    main()
