"""
scheduler.py — APScheduler-based automation for the ML Signal Scanner.

Jobs
----
run_daily_scan(cfg)
    Runs the full scan pipeline after US market close:
      1. Incremental data download
      2. Feature engineering
      3. Signal generation
      4. Trade card construction
      5. Alert dispatch
    Logs outcome to scheduler/scan_log.csv.

run_weekly_retrain(cfg)
    Retrains the HMM regime detector and XGBoost model weekly:
      1. Backs up existing models
      2. Retrains HMM, then XGBoost (walk-forward)
      3. Compares new Sharpe vs old Sharpe (90 % threshold)
      4. Rolls back if quality degrades
    Logs outcome to scheduler/retrain_log.csv.

Usage
-----
    # Start scheduler daemon (runs until Ctrl-C)
    python -m scheduler.scheduler

    # One-shot manual trigger (for Streamlit / testing)
    python -m scheduler.scheduler --run-now-scan
    python -m scheduler.scheduler --run-now-retrain
"""

import argparse
import csv
import json
import logging
import pathlib
import shutil
import sys
import time
from datetime import datetime, timezone

import yaml

log = logging.getLogger(__name__)

_ROOT        = pathlib.Path(__file__).resolve().parents[1]
_MODELS_DIR  = _ROOT / "models" / "saved"
_BACKUP_DIR  = _ROOT / "models" / "saved_backup"
_METRICS_PATH = _MODELS_DIR / "backtest_metrics.json"
_SCAN_LOG    = _ROOT / "scheduler" / "scan_log.csv"
_RETRAIN_LOG = _ROOT / "scheduler" / "retrain_log.csv"

_SCAN_LOG_COLS    = ["datetime", "status", "signals_found", "alerts_sent", "error"]
_RETRAIN_LOG_COLS = ["datetime", "old_sharpe", "new_sharpe", "model_accepted", "error"]


# ── Config ────────────────────────────────────────────────────────────────


def _load_config() -> dict:
    """Load and return config/config.yaml as a dict."""
    cfg_path = _ROOT / "config" / "config.yaml"
    with open(cfg_path) as fh:
        return yaml.safe_load(fh)


# ── CSV logger ────────────────────────────────────────────────────────────


def _log(log_path: pathlib.Path, row: dict) -> None:
    """
    Append one row dict to a CSV log file.

    Creates the file (with header) on first write.  The column order matches
    the key order in *row*.

    Parameters
    ----------
    log_path : pathlib.Path
    row      : dict
        Keys become column headers on first write.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    try:
        with open(log_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception as exc:
        log.warning("Could not write %s: %s", log_path, exc)


# ── Daily scan job ────────────────────────────────────────────────────────


def run_daily_scan(cfg: dict) -> None:
    """
    Execute the full scan pipeline and dispatch alerts.

    Steps
    -----
    1. Incremental data download  (ingestion.downloader.main)
    2. Feature engineering        (features.pipeline.main)
    3. Signal generation          (signals.scanner.run_scan)
    4. Trade card construction    (signals.trade_setup.build_all_trade_cards)
    5. Alert dispatch             (alerts.notifier.NotificationManager.send_alerts)
    6. Log result to scan_log.csv

    Parameters
    ----------
    cfg : dict
        Full config dict.
    """
    ts         = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    error_msg  = ""
    signals_n  = 0
    alerts_n   = 0
    status     = "ok"

    try:
        # ── Step 1: download incremental bars ────────────────────────────
        log.info("[scan] Downloading latest price data…")
        import ingestion.downloader as _dl
        _dl.main()

        # ── Step 2: rebuild feature parquets ─────────────────────────────
        log.info("[scan] Rebuilding feature parquets…")
        import features.pipeline as _fp
        _fp.main()

        # ── Step 3: run scanner ───────────────────────────────────────────
        log.info("[scan] Running signal scanner…")
        from signals.scanner import run_scan
        signals_df = run_scan(cfg)
        signals_n  = len(signals_df)
        log.info("[scan] %d signal(s) found.", signals_n)

        # ── Step 4: build trade cards ─────────────────────────────────────
        trade_cards = None
        if signals_n > 0:
            log.info("[scan] Building trade cards…")
            from signals.trade_setup import build_all_trade_cards
            trade_cards = build_all_trade_cards(signals_df, cfg)

        # ── Step 5: dispatch alerts ───────────────────────────────────────
        if trade_cards is not None and not trade_cards.empty:
            log.info("[scan] Dispatching alerts…")
            from alerts.notifier import NotificationManager
            result   = NotificationManager(cfg).send_alerts(trade_cards)
            alerts_n = result.get("sent", 0)
        else:
            log.info("[scan] No trade cards — skipping alerts.")

    except Exception as exc:
        status    = "error"
        error_msg = str(exc)
        log.error("[scan] Pipeline error: %s", exc, exc_info=True)

    _log(_SCAN_LOG, {
        "datetime":       ts,
        "status":         status,
        "signals_found":  signals_n,
        "alerts_sent":    alerts_n,
        "error":          error_msg,
    })
    log.info("[scan] Done — status=%s signals=%d alerts=%d", status, signals_n, alerts_n)


# ── Weekly retrain job ────────────────────────────────────────────────────


def run_weekly_retrain(cfg: dict) -> None:
    """
    Retrain HMM and XGBoost models, keeping new models only if Sharpe holds up.

    Steps
    -----
    1. Load existing backtest_metrics.json → record old Sharpe
    2. Back up current model artefacts to models/saved_backup/
    3. Retrain HMM regime detector
    4. Retrain XGBoost (walk-forward)
    5. Load new backtest_metrics.json → record new Sharpe
    6. If new_sharpe >= 0.9 * old_sharpe: keep new models
       Else: restore backup
    7. Log result to retrain_log.csv

    Parameters
    ----------
    cfg : dict
        Full config dict.
    """
    ts           = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    error_msg    = ""
    old_sharpe   = float("nan")
    new_sharpe   = float("nan")
    accepted     = False

    try:
        # ── Step 1: record existing Sharpe ───────────────────────────────
        if _METRICS_PATH.exists():
            with open(_METRICS_PATH) as fh:
                metrics   = json.load(fh)
            old_sharpe = float(metrics.get("sharpe_ratio", float("nan")))
        log.info("[retrain] Old Sharpe ratio: %.4f", old_sharpe)

        # ── Step 2: back up existing models ──────────────────────────────
        if _MODELS_DIR.exists():
            if _BACKUP_DIR.exists():
                shutil.rmtree(_BACKUP_DIR)
            shutil.copytree(_MODELS_DIR, _BACKUP_DIR)
            log.info("[retrain] Models backed up to %s", _BACKUP_DIR)

        # ── Step 3: retrain HMM ───────────────────────────────────────────
        log.info("[retrain] Retraining HMM regime detector…")
        import regime.hmm_detector as _hmm
        _hmm.main()

        # ── Step 4: retrain XGBoost ───────────────────────────────────────
        log.info("[retrain] Retraining XGBoost (walk-forward)…")
        from models.trainer import run_walk_forward
        run_walk_forward(cfg)

        # ── Step 5: load new Sharpe ───────────────────────────────────────
        if _METRICS_PATH.exists():
            with open(_METRICS_PATH) as fh:
                new_metrics = json.load(fh)
            new_sharpe = float(new_metrics.get("sharpe_ratio", float("nan")))
        log.info("[retrain] New Sharpe ratio: %.4f", new_sharpe)

        # ── Step 6: accept or roll back ───────────────────────────────────
        import math
        if math.isnan(old_sharpe) or math.isnan(new_sharpe):
            # Cannot compare — keep new models but log the anomaly
            accepted  = True
            log.warning("[retrain] Sharpe comparison skipped (NaN). Keeping new models.")
        elif new_sharpe >= 0.9 * old_sharpe:
            accepted = True
            log.info("[retrain] New models accepted (%.4f >= 90%% of %.4f).",
                     new_sharpe, old_sharpe)
        else:
            accepted = False
            log.warning(
                "[retrain] New Sharpe %.4f < 90%% of old %.4f — rolling back.",
                new_sharpe, old_sharpe,
            )
            if _BACKUP_DIR.exists():
                shutil.rmtree(_MODELS_DIR)
                shutil.copytree(_BACKUP_DIR, _MODELS_DIR)
                log.info("[retrain] Models restored from backup.")

    except Exception as exc:
        error_msg = str(exc)
        log.error("[retrain] Retrain error: %s", exc, exc_info=True)

    _log(_RETRAIN_LOG, {
        "datetime":       ts,
        "old_sharpe":     f"{old_sharpe:.4f}" if not __import__("math").isnan(old_sharpe) else "",
        "new_sharpe":     f"{new_sharpe:.4f}" if not __import__("math").isnan(new_sharpe) else "",
        "model_accepted": accepted,
        "error":          error_msg,
    })
    log.info("[retrain] Done — accepted=%s", accepted)


# ── Scheduler entry point ─────────────────────────────────────────────────


def main() -> None:
    """
    Start the APScheduler daemon or execute a one-shot job.

    Flags
    -----
    --run-now-scan      Run the daily scan immediately then exit.
    --run-now-retrain   Run the weekly retrain immediately then exit.
    (no flags)          Start the background scheduler and block until Ctrl-C.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="ML Scanner Scheduler")
    parser.add_argument("--run-now-scan",    action="store_true",
                        help="Run the daily scan pipeline immediately and exit.")
    parser.add_argument("--run-now-retrain", action="store_true",
                        help="Run the weekly retrain pipeline immediately and exit.")
    args = parser.parse_args()

    cfg = _load_config()

    # ── One-shot manual triggers ──────────────────────────────────────────
    if args.run_now_scan:
        log.info("Manual trigger: daily scan")
        run_daily_scan(cfg)
        return

    if args.run_now_retrain:
        log.info("Manual trigger: weekly retrain")
        run_weekly_retrain(cfg)
        return

    # ── APScheduler daemon ────────────────────────────────────────────────
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
        import pytz
    except ImportError as exc:
        log.error("APScheduler or pytz not installed: %s", exc)
        sys.exit(1)

    sched_cfg  = cfg.get("scheduler", {})
    tz_name    = sched_cfg.get("scan_timezone", "Europe/London")
    scan_tz    = pytz.timezone(tz_name)

    # Parse scan time "HH:MM"
    scan_time    = sched_cfg.get("scan_time", "21:00")
    scan_h, scan_m = (int(x) for x in scan_time.split(":"))

    # Parse retrain time "HH:MM"
    retrain_time = sched_cfg.get("retrain_time", "08:00")
    ret_h, ret_m = (int(x) for x in retrain_time.split(":"))

    # APScheduler day-of-week (mon=0 … sun=6)
    day_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    retrain_day  = sched_cfg.get("retrain_day", "sunday")
    retrain_dow  = day_map.get(retrain_day.lower(), 6)

    scheduler = BackgroundScheduler(timezone=scan_tz)

    scheduler.add_job(
        func=run_daily_scan,
        trigger=CronTrigger(hour=scan_h, minute=scan_m, timezone=scan_tz),
        args=[cfg],
        id="daily_scan",
        name="Daily Scan",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    log.info("Daily scan scheduled at %02d:%02d %s.", scan_h, scan_m, tz_name)

    scheduler.add_job(
        func=run_weekly_retrain,
        trigger=CronTrigger(
            day_of_week=retrain_dow,
            hour=ret_h,
            minute=ret_m,
            timezone=scan_tz,
        ),
        args=[cfg],
        id="weekly_retrain",
        name="Weekly Retrain",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    log.info("Weekly retrain scheduled on %s at %02d:%02d %s.",
             retrain_day, ret_h, ret_m, tz_name)

    scheduler.start()
    log.info("Scheduler started.  Press Ctrl-C to stop.")

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        log.info("Shutting down scheduler…")
        scheduler.shutdown()
        log.info("Scheduler stopped.")


if __name__ == "__main__":
    main()
