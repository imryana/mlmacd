# ML Scanner — S&P 500 Machine Learning Signal Scanner

> An end-to-end automated pipeline that scans the S&P 500 daily, classifies market regimes, generates directional trade signals using XGBoost, and delivers trade cards with ATR-based position sizing via a Streamlit dashboard.

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://imryana-mlmacd-xencoptzxfzymp5uvglpw6.streamlit.app/)

> **[Launch the app →](https://imryana-mlmacd-xencoptzxfzymp5uvglpw6.streamlit.app/)**

---

## What Does It Do?

Every trading day the pipeline:

1. **Downloads** the latest OHLCV price bars for 427 S&P 500 stocks (via yfinance)
2. **Computes** 33 technical features per stock (MACD, RSI, ATR, EMA, ADX, Bollinger Bands, OBV, volume, volatility)
3. **Classifies** the market regime for each stock using a 3-state Hidden Markov Model (bull / bear / choppy)
4. **Scores** each stock with a pooled XGBoost classifier → outputs a Long / Flat / Short signal with a confidence score
5. **Filters** signals by confidence threshold and ADX trend strength
6. **Builds trade cards** with limit entry prices, stop-loss, take-profit, and position sizing
7. **Sends alerts** via Email or Telegram

---

## How the Model Works

### Stage 1 — Hidden Markov Model (Regime Detection)

Before any directional signal is generated, a **3-state Gaussian HMM** classifies the current market regime for each stock:

| Regime | Characteristics | Frequency |
|--------|----------------|-----------|
| **Bull** | Low volatility, upward drift | 42.8% |
| **Choppy** | Medium volatility, no clear direction | 40.8% |
| **Bear** | High volatility, downward drift | 16.4% |

The HMM is trained unsupervised on realised volatility, ATR%, and 5-day forward returns. The regime label is fed directly into XGBoost as a feature — this lets the model apply different signal weights depending on whether conditions are trending or volatile.

### Stage 2 — XGBoost Signal Classifier

A single **pooled XGBoost model** is trained across all 427 tickers simultaneously (rather than one model per stock). This:
- Prevents overfitting to any one ticker's history
- Lets the model learn cross-sectional patterns (e.g. what "oversold in a bear regime" means across many stocks)
- Produces a 3-class probability: P(Long), P(Flat), P(Short)

**Training method:** Walk-forward cross-validation — the model is never trained on future data. A 504-bar (~2 year) training window slides forward in 63-bar (~3 month) steps, producing 53 out-of-sample folds.

### Input Features (33 total)

| Category | Features |
|----------|----------|
| **MACD** | Line, signal, histogram, histogram slope (1-bar, 3-bar), crossover flag, divergence |
| **RSI** | Level, slope, overbought/oversold flags |
| **Trend** | EMA fast (50), EMA slow (200), price/EMA ratio, golden cross, death cross |
| **Momentum** | ADX, +DI, -DI, trend flag |
| **Volatility** | ATR, ATR%, Bollinger Band width, realised vol, 60-day rolling return, autocorrelation |
| **Volume** | Volume/MA ratio, OBV slope, volume confirmation flag |
| **Context** | Sector (one-hot), HMM regime |

**Key insight — MACD:** The model uses continuous MACD histogram slope features rather than relying on the binary crossover event. This allows it to detect a crossover *building* before it fires — the histogram converging toward zero is a leading signal, not a lagging one.

### Signal Generation

A signal passes all filters when:
- `P(predicted class) ≥ 0.60` (confidence threshold)
- `ADX ≥ 20` (trend is strong enough to trade)
- Ticker has at least 504 bars of history

### Trade Card Construction

For every signal that passes filters, the pipeline computes:

| Parameter | Formula |
|-----------|---------|
| **Limit entry** | Close ± 0.25 × ATR (better fill than chasing market) |
| **Stop loss** | Entry ± 2.0 × ATR |
| **Take profit** | Entry ± 2 × risk (R:R = 2:1) |
| **Position size** | 1% portfolio risk ÷ ATR risk per unit |
| **Entry expiry** | Cancelled after 2 bars if unfilled |

---

## What Was It Trained On?

| Item | Detail |
|------|--------|
| **Universe** | 427 S&P 500 stocks (76 excluded — insufficient history or data quality) |
| **History** | 2010–present (~15 years per ticker) |
| **Total rows** | 1,551,959 bar-level observations |
| **Walk-forward folds** | 53 (504-bar train / 63-bar test, sliding) |
| **Test set size** | 1,361,326 out-of-sample predictions |
| **Labels** | 5-bar forward return: Long (>+1%), Short (<-1%), Flat (between) |

### Out-of-Sample Performance

| Metric | Value |
|--------|-------|
| Accuracy | 40.6% (3-class, random = 33%) |
| Signal hit rate | 53.4% |
| Mean signal return | +0.23% per 5-bar hold |
| Signal Sharpe | 0.82 |
| Weighted F1 | 0.370 |

### Top 10 Features by SHAP Importance

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | ATR % (volatility-adjusted range) | 0.123 |
| 2 | HMM Regime | 0.051 |
| 3 | EMA 200 (slow trend) | 0.036 |
| 4 | RSI | 0.025 |
| 5 | EMA 50 (fast trend) | 0.023 |
| 6 | Realised volatility | 0.023 |
| 7 | Volume ratio | 0.023 |
| 8 | Sector | 0.018 |
| 9 | Price / EMA 200 ratio | 0.015 |
| 10 | ADX +DI | 0.014 |

---

## Architecture

```
Raw OHLCV (yfinance)
       │
       ▼
┌─────────────────┐
│   Downloader    │  427 tickers · daily incremental
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Feature Pipeline│  33 features · labels · pooled parquet
└──────┬──────────┘
       │
       ├──────────────────────┐
       ▼                      ▼
┌─────────────┐    ┌──────────────────┐
│  HMM Model  │    │  XGBoost Model   │
│  3 regimes  │───▶│  Walk-forward CV │
└─────────────┘    └──────────┬───────┘
                              │
                              ▼
                   ┌──────────────────┐
                   │     Scanner      │  Confidence filter · ADX filter
                   └──────────┬───────┘
                              │
                              ▼
                   ┌──────────────────┐
                   │   Trade Setup    │  Entry · Stop · Target · Sizing
                   └──────┬─────┬────┘
                          │     │
                    ┌─────▼┐   ┌▼──────────┐
                    │Alerts│   │ Streamlit  │
                    │Email/│   │ Dashboard  │
                    │  TG  │   │  5 pages   │
                    └──────┘   └────────────┘
```

---

## Streamlit App — 5 Pages

| Page | Description |
|------|-------------|
| **Scanner** | Live signal table with confidence, regime, sector filters and CSV export |
| **Stock Detail** | Price chart, MACD/RSI indicators, trade setup panel, SHAP feature drivers |
| **Portfolio** | Holdings input, model signal overlay, sector exposure pie chart |
| **Performance** | Backtest equity curve, Sharpe/Sortino/drawdown metrics, monthly return heatmap |
| **Settings** | Edit config, set alert credentials, trigger manual scan or retrain |

---

## Automation

The scheduler runs two jobs automatically:

| Job | When | What |
|-----|------|------|
| **Daily Scan** | 21:00 London time | Download → Features → Scan → Alerts |
| **Weekly Retrain** | Sunday 08:00 | Retrain HMM + XGBoost, auto-rollback if Sharpe degrades |

```bash
python -m scheduler.scheduler          # start daemon
python -m scheduler.scheduler --run-now-scan     # manual trigger
python -m scheduler.scheduler --run-now-retrain  # manual retrain
```

---

## Quick Start

```bash
git clone https://github.com/imryana/mlmacd.git
cd mlmacd

pip install -r requirements.txt
cp .env.example .env          # add email/Telegram credentials

# Run the pipeline
python -m ingestion.downloader
python -m features.pipeline
python -m regime.hmm_detector
python -m models.trainer
python -m signals.scanner

# Launch the app
streamlit run app/app.py
```

---

## Tech Stack

`Python 3.11` · `XGBoost` · `hmmlearn` · `pandas-ta` · `SHAP` · `Streamlit` · `Plotly` · `APScheduler` · `yfinance`
