# CLAUDE.md — S&P 500 ML Signal Scanner

## !! ACTION REQUIRED BEFORE PRODUCTION USE !!
The pipeline has only been run on 5 test tickers (AAPL, MSFT, GOOGL, JPM, JNJ).
To train on the full S&P 500 universe run these commands in order:

    python -m ingestion.downloader          # ~20-30 min — downloads all 503 tickers
    python -m features.pipeline             # ~5-10 min  — builds processed parquets
    python -m regime.hmm_detector           # ~1 min     — retrains HMM on full data
    python -m models.trainer                # ~5-15 min  — retrains XGBoost (GPU)
    python -m models.trainer --tune         # optional: Optuna hyperparameter search

---

## 1. Project Purpose

A fully automated end-to-end pipeline that scans the S&P 500 daily, classifies
each stock's market regime with an HMM, generates directional signals with a
pooled XGBoost classifier, constructs ATR-based trade cards with position sizing,
and dispatches email/Telegram alerts — all surfaced in a five-page Streamlit app.

---

## 2. Architecture Overview

```
Raw OHLCV (yfinance)
       │
       ▼
┌──────────────────┐
│ ingestion/       │  downloader.py — incremental per-ticker parquet
│ downloader.py    │  universe.py   — S&P 500 ticker list (CSV)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ features/        │  indicators.py — 20+ TA features (MACD, RSI, ATR, …)
│ pipeline.py      │  labels.py     — forward-return labels (long/flat/short)
└──────┬───────────┘
       │
       ├─────────────────────────────┐
       ▼                             ▼
┌──────────────────┐    ┌────────────────────┐
│ regime/          │    │ models/             │
│ hmm_detector.py  │    │ trainer.py          │
│ 3-state HMM      │    │ XGBoost walk-fwd CV │
└──────┬───────────┘    └──────────┬─────────┘
       └─────────┬─────────────────┘
                 │
                 ▼
       ┌──────────────────┐
       │ signals/         │
       │ scanner.py       │  run_scan(cfg) → signal rows
       │ trade_setup.py   │  build_all_trade_cards() → trade cards parquet
       └──────┬───────────┘
              │
     ┌────────┴────────┐
     ▼                 ▼
┌──────────┐   ┌────────────────┐
│ alerts/  │   │ app/           │
│notifier  │   │ Streamlit app  │
│ Email +  │   │ 5-page UI      │
│ Telegram │   │                │
└──────────┘   └────────────────┘
              ▲
              │
┌─────────────────┐
│ scheduler/      │
│ scheduler.py    │  APScheduler daily scan + weekly retrain
└─────────────────┘
```

---

## 3. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Pooled XGBoost** | Single model trained on all tickers — regularises against ticker-specific noise and avoids 500 separate models |
| **Walk-forward CV** | 504-bar train / 63-bar test sliding window gives realistic out-of-sample metrics and prevents lookahead bias |
| **3-state GaussianHMM** | Captures bull / bear / choppy regimes as an unsupervised feature; regime is fed into XGBoost as an input |
| **ATR-based sizing** | Stop distance = ATR × stop_multiplier; position units = 1 % risk ÷ ATR risk, giving volatility-adjusted size |
| **Limit-order entry** | Entry placed 0.25× ATR below close (longs) / above close (shorts) to avoid chasing; expires after 2 bars |
| **Dedup alerts** | Alerts log tracks (ticker, signal, date) — same signal not re-sent on the same day even if scheduler re-fires |

---

## 4. Module Reference

| Module | Entry point | Purpose |
|--------|-------------|---------|
| `ingestion/downloader.py` | `python -m ingestion.downloader` | Download OHLCV from yfinance; saves per-ticker raw parquets |
| `ingestion/universe.py` | `python -m ingestion.universe` | Build S&P 500 ticker list CSV |
| `features/pipeline.py` | `python -m features.pipeline` | Run indicators + labels on all tickers; save pooled_features.parquet |
| `features/indicators.py` | — | Compute 20+ technical indicators via pandas-ta |
| `features/labels.py` | — | Forward-return multi-class labels (long / flat / short) |
| `regime/hmm_detector.py` | `python -m regime.hmm_detector` | Train 3-state GaussianHMM; save model pickle |
| `models/trainer.py` | `python -m models.trainer` | Walk-forward XGBoost training; save xgboost_final.pkl + metrics |
| `models/xgboost_model.py` | — | XGBoostModel wrapper (fit / predict / load / save) |
| `signals/scanner.py` | `python -m signals.scanner` | `run_scan(cfg)` → ranked signal DataFrame |
| `signals/trade_setup.py` | `python -m signals.trade_setup` | `build_all_trade_cards()` → trade cards parquet |
| `backtest/engine.py` | — | Event-driven bar-by-bar backtest simulation |
| `backtest/metrics.py` | `python -m backtest.metrics` | Sharpe, Sortino, max-DD, annualised return |
| `alerts/notifier.py` | `python -m alerts.notifier` | EmailNotifier + TelegramNotifier + NotificationManager |
| `scheduler/scheduler.py` | `python -m scheduler.scheduler` | APScheduler daemon: daily scan + weekly retrain |
| `app/app.py` | `streamlit run app/app.py` | Five-page Streamlit application entry point |

---

## 5. Running the Pipeline

Run from `ml_scanner/` with the project venv activated.

```
# 1. (First run only) Download universe list
python -m ingestion.universe             # writes data/universe.csv

# 2. Download price data for all tickers
python -m ingestion.downloader           # writes data/raw/<ticker>.parquet

# 3. Build feature and label parquets
python -m features.pipeline              # writes data/processed/

# 4. Train HMM regime model
python -m regime.hmm_detector            # writes models/saved/hmm_detector.pkl

# 5. Train XGBoost (walk-forward)
python -m models.trainer                 # writes models/saved/xgboost_final.pkl
                                         #        models/saved/wf_metrics.json

# 6. Optional: Optuna hyperparameter search (~50 trials, ~30 min)
python -m models.trainer --tune

# 7. Run signal scanner
python -m signals.scanner               # writes data/processed/signals.parquet

# 8. Build trade cards
python -m signals.trade_setup           # writes data/processed/trade_cards.parquet

# 9. (Optional) Send alerts manually
python -m alerts.notifier

# 10. Launch Streamlit app
streamlit run app/app.py

# 11. (Optional) Start automated scheduler daemon
python -m scheduler.scheduler

# 12. (Optional) One-shot scan trigger from CLI
python -m scheduler.scheduler --run-now-scan
```

---

## 6. Environment Variables

Set in a `.env` file at the project root (copy `.env.example` to `.env`).

| Variable | Required | Description |
|----------|----------|-------------|
| `EMAIL_USER` | No | Gmail address used as both sender and recipient |
| `EMAIL_PASS` | No | Gmail App Password (not your account password) |
| `TELEGRAM_BOT_TOKEN` | No | Bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | No | Your Telegram chat/user ID |
| `POLYGON_API_KEY` | No | Optional — only needed if switching data source from yfinance |

Enable alerts in `config/config.yaml`:
```yaml
alerts:
  email_enabled: true
  telegram_enabled: true
  min_confidence_to_alert: 0.65
```

---

## 7. Current Status

| Phase | Status | Description |
|-------|--------|-------------|
| 1 — Scaffold | ✓ Complete | Directory structure, config.yaml, logging |
| 2 — Config | ✓ Complete | YAML config with all parameters |
| 3 — Ingestion | ✓ Complete | yfinance downloader, universe CSV |
| 4 — Features | ✓ Complete | 20+ indicators, forward-return labels |
| 5 — HMM | ✓ Complete | 3-state GaussianHMM regime detector |
| 6 — XGBoost | ✓ Complete | Walk-forward CV, pooled model, SHAP |
| 7 — Scanner | ✓ Complete | Signal scanner + trade card builder |
| 8 — Backtest | ✓ Complete | Event-driven engine + Sharpe/Sortino/DD metrics |
| 9 — App | ✓ Complete | Five-page Streamlit UI |
| 10 — Alerts | ✓ Complete | Email + Telegram notifiers, dedup log |
| 11 — Scheduler | ✓ Complete | APScheduler daily scan + weekly retrain |
| 12 — Docs | ✓ Complete | This file |
| 13 — Packaging | ✓ Complete | requirements.txt, .env.example, .gitignore |

**Pending / known limitations:**
- Full S&P 500 data not yet downloaded (only 5 test tickers)
- GPU training requires CUDA; set `use_gpu: false` in config.yaml for CPU-only machines
- LSTM model (`models/lstm_model.py`) is a stub — XGBoost is the active classifier

---

## 8. Development Notes

### Python 3.11 — no backslashes inside f-strings
This is a syntax error in Python 3.11:
```python
f"path: {str(path).replace('\\', '/')}"   # ✗ SyntaxError
```
Extract to a local variable first:
```python
path_str = str(path).replace('\\', '/')
f"path: {path_str}"                        # ✓
```

### Confidence threshold for demo
With only 5 tickers the model rarely exceeds 0.60 confidence.
To see signals in the app during development, lower the threshold:
```yaml
model:
  confidence_threshold: 0.40
```

### Streamlit / PowerShell false error
Streamlit writes startup logs to stderr.  PowerShell interprets any stderr
output as a non-zero exit code.  This is cosmetic — the app runs correctly.

### Walk-forward metrics interpretation
- `wf_metrics.json`      → XGBoost CV accuracy, F1, signal hit-rate
- `backtest_metrics.json` → full backtest Sharpe, Sortino, max drawdown, CAGR
  The scheduler's retrain guard uses `backtest_metrics.json["sharpe_ratio"]`.

### Adding a new ticker
```
# 1. Add ticker to data/universe.csv
# 2. Re-run the pipeline from step 2 (downloader) onwards
```
