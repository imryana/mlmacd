"""
settings.py — Page 5: Settings

Allows the user to view and edit configuration parameters and save
changes back to config.yaml.  Organises settings into four sections:
  - Portfolio
  - Signal
  - Alerts
  - Scheduler

Also exposes manual pipeline controls (Run Scan Now, Retrain Model Now)
and a raw config.yaml viewer.
"""

import pathlib
import subprocess
import sys

import streamlit as st
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

_CONFIG_PATH = _ROOT / "config" / "config.yaml"


# ── Config I/O ────────────────────────────────────────────────────────────


def _load_cfg() -> dict:
    """Load config.yaml as a dict."""
    try:
        with open(_CONFIG_PATH) as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:
        st.error(f"Could not read config.yaml: {exc}")
        return {}


def _save_cfg(cfg: dict) -> bool:
    """
    Write *cfg* back to config.yaml.

    Returns
    -------
    bool
        True on success.
    """
    try:
        with open(_CONFIG_PATH, "w") as fh:
            yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)
        return True
    except Exception as exc:
        st.error(f"Could not write config.yaml: {exc}")
        return False


# ── Render ────────────────────────────────────────────────────────────────


def render(cfg: dict) -> None:
    """
    Render the Settings page.

    Parameters
    ----------
    cfg : dict
        Full config dict (may be stale; we reload fresh copy for editing).
    """
    st.title("Settings")
    st.caption("Changes are saved to `config/config.yaml`.")

    # Always reload a fresh copy for editing
    fresh_cfg = _load_cfg()
    if not fresh_cfg:
        st.error("Cannot load configuration. Check that config/config.yaml exists.")
        return

    portfolio = fresh_cfg.get("portfolio", {})
    signals   = fresh_cfg.get("signals", {})
    alerts    = fresh_cfg.get("alerts", {})
    scheduler = fresh_cfg.get("scheduler", {})

    # ── Portfolio settings ────────────────────────────────────────────────
    st.subheader("Portfolio")
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    new_size = c1.number_input(
        "Portfolio Size (£)",
        min_value=1_000, max_value=10_000_000,
        value=int(portfolio.get("size", 50_000)), step=1_000,
    )
    new_risk = c2.number_input(
        "Risk per Trade (%)",
        min_value=0.1, max_value=10.0,
        value=float(portfolio.get("risk_per_trade", 0.01)) * 100,
        step=0.1, format="%.1f",
    )
    new_max_pos = c3.number_input(
        "Max Open Positions",
        min_value=1, max_value=100,
        value=int(portfolio.get("max_open_positions", 10)),
    )
    new_max_sector = c4.slider(
        "Max Sector Exposure (%)",
        min_value=5, max_value=100,
        value=int(float(portfolio.get("max_sector_exposure", 0.30)) * 100),
        step=5,
    )

    # ── Signal settings ───────────────────────────────────────────────────
    st.subheader("Signals")
    s1, s2, s3 = st.columns(3)
    s4, s5, s6 = st.columns(3)

    new_conf_thresh = s1.slider(
        "Confidence Threshold",
        min_value=0.50, max_value=1.0,
        value=float(fresh_cfg.get("model", {}).get("confidence_threshold", 0.60)),
        step=0.01, format="%.2f",
    )
    new_entry_method = s2.selectbox(
        "Entry Method",
        ["limit", "next_open", "confirmation"],
        index=["limit", "next_open", "confirmation"].index(
            signals.get("entry_method", "limit")
        ),
    )
    new_stop_mult = s3.slider(
        "Stop Multiplier (× ATR)",
        min_value=0.5, max_value=5.0,
        value=float(signals.get("stop_multiplier", 2.0)), step=0.25,
    )
    new_rr = s4.slider(
        "R:R Ratio",
        min_value=1.0, max_value=5.0,
        value=float(signals.get("rr_ratio", 2.0)), step=0.25,
    )
    new_trailing = s5.toggle(
        "Trailing Stop",
        value=float(signals.get("trailing_stop_activation_atr", 1.0)) > 0,
    )
    new_time_exit = s6.number_input(
        "Time Exit (bars)",
        min_value=1, max_value=50,
        value=int(signals.get("time_exit_bars", 5)),
    )

    # ── Alert settings ────────────────────────────────────────────────────
    st.subheader("Alerts")
    a1, a2 = st.columns(2)

    email_on = a1.toggle("Email Alerts", value=bool(alerts.get("email_enabled", False)))
    tg_on    = a2.toggle("Telegram Alerts", value=bool(alerts.get("telegram_enabled", False)))

    new_min_conf_alert = st.slider(
        "Min Confidence to Alert",
        min_value=0.50, max_value=1.0,
        value=float(alerts.get("min_confidence_to_alert", 0.65)),
        step=0.01, format="%.2f",
    )

    if email_on:
        with st.expander("Email Configuration", expanded=True):
            st.text_input("SMTP Server",
                          value=alerts.get("email_smtp_server", "smtp.gmail.com"))
            st.number_input("SMTP Port", value=int(alerts.get("email_port", 587)))
            st.info("Set EMAIL_USER and EMAIL_PASS in your `.env` file.")

    if tg_on:
        with st.expander("Telegram Configuration", expanded=True):
            st.info("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your `.env` file.")

    # Test alert button
    if st.button("Send Test Alert"):
        from alerts.notifier import NotificationManager
        import pandas as pd
        cards_path = _ROOT / "data" / "processed" / "trade_cards.parquet"
        if cards_path.exists():
            cards = pd.read_parquet(cards_path)
            mgr   = NotificationManager(fresh_cfg)
            result = mgr.send_alerts(cards)
            st.success(
                f"Alerts dispatched — {result['sent']} sent, "
                f"{result['skipped_conf']} skipped (conf), "
                f"{result['skipped_dedup']} skipped (dedup)."
            )
        else:
            st.warning("No trade cards found. Run the scanner first.")

    # ── Scheduler settings ────────────────────────────────────────────────
    st.subheader("Scheduler")
    sch1, sch2 = st.columns(2)

    new_scan_time = sch1.text_input(
        "Daily Scan Time (HH:MM)",
        value=scheduler.get("scan_time", "21:00"),
    )
    new_retrain_day = sch2.selectbox(
        "Weekly Retrain Day",
        ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"],
        index=["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
        .index(scheduler.get("retrain_day", "sunday")),
    )

    st.markdown("---")

    # ── Manual controls ───────────────────────────────────────────────────
    st.subheader("Manual Pipeline Controls")
    col_scan, col_retrain, _ = st.columns([1, 1, 2])

    if col_scan.button("Run Scan Now", use_container_width=True):
        with st.spinner("Running scanner pipeline…"):
            result = subprocess.run(
                [sys.executable, "-m", "signals.scanner"],
                cwd=str(_ROOT), capture_output=True, text=True, timeout=300,
            )
        if result.returncode == 0:
            st.success("Scan complete.")
        else:
            st.error(f"Scan failed:\n{result.stderr[-400:]}")

    if col_retrain.button("Retrain Model Now", use_container_width=True):
        with st.spinner("Retraining model — this may take several minutes…"):
            result = subprocess.run(
                [sys.executable, "-m", "models.trainer"],
                cwd=str(_ROOT), capture_output=True, text=True, timeout=1800,
            )
        if result.returncode == 0:
            st.success("Model retrained successfully.")
        else:
            st.error(f"Retraining failed:\n{result.stderr[-400:]}")

    st.markdown("---")

    # ── Save settings ─────────────────────────────────────────────────────
    if st.button("Save Settings", type="primary", use_container_width=False):
        fresh_cfg["portfolio"]["size"]                  = int(new_size)
        fresh_cfg["portfolio"]["risk_per_trade"]        = round(new_risk / 100, 4)
        fresh_cfg["portfolio"]["max_open_positions"]    = int(new_max_pos)
        fresh_cfg["portfolio"]["max_sector_exposure"]   = round(new_max_sector / 100, 2)
        fresh_cfg["model"]["confidence_threshold"]      = round(new_conf_thresh, 2)
        fresh_cfg["signals"]["entry_method"]            = new_entry_method
        fresh_cfg["signals"]["stop_multiplier"]         = round(new_stop_mult, 2)
        fresh_cfg["signals"]["rr_ratio"]                = round(new_rr, 2)
        fresh_cfg["signals"]["time_exit_bars"]          = int(new_time_exit)
        fresh_cfg["alerts"]["email_enabled"]            = email_on
        fresh_cfg["alerts"]["telegram_enabled"]         = tg_on
        fresh_cfg["alerts"]["min_confidence_to_alert"]  = round(new_min_conf_alert, 2)
        fresh_cfg["scheduler"]["scan_time"]             = new_scan_time
        fresh_cfg["scheduler"]["retrain_day"]           = new_retrain_day

        if _save_cfg(fresh_cfg):
            st.success("Settings saved to config.yaml.")
            st.rerun()
        else:
            st.error("Failed to save settings.")

    st.markdown("---")

    # ── Raw config viewer ─────────────────────────────────────────────────
    with st.expander("Raw config.yaml"):
        try:
            raw = _CONFIG_PATH.read_text()
        except Exception:
            raw = "Could not read config.yaml"
        st.code(raw, language="yaml")
