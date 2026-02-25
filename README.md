# ML Scanner — S&P 500 Machine Learning Signal Scanner

> An end-to-end automated pipeline that scans the S&P 500 daily, classifies market regimes, generates directional trade signals using a dual XGBoost architecture, and delivers trade cards with ATR-based position sizing via a Streamlit dashboard.

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://imryana-mlmacd-xencoptzxfzymp5uvglpw6.streamlit.app/)

> **[Launch the app →](https://imryana-mlmacd-xencoptzxfzymp5uvglpw6.streamlit.app/)**

---

## What Does It Do?

Every trading day the pipeline:

1. **Downloads** the latest OHLCV price bars for 496 S&P 500 stocks (via yfinance)
2. **Computes** 45 technical features per stock — 13 MACD-family, RSI, ATR, EMA, ADX, Bollinger Bands, OBV, volume, volatility, and 6 macro-market features (VIX, SPY)
3. **Classifies** the market regime for each stock using a 3-state Hidden Markov Model (bull / bear / choppy)
4. **Scores** each stock with two independent binary XGBoost classifiers — one for longs, one for shorts
5. **Filters** signals by confidence threshold, ADX trend strength, and regime:
   - Choppy regime: all signals suppressed
   - Short signals: bear regime only
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

The HMM is trained unsupervised on realised volatility, ATR%, and 5-day forward returns. The regime label is fed directly into XGBoost as a feature — this lets the model learn different signal weights for each market condition. Signals in choppy regimes are fully suppressed; short signals are restricted to bear regimes only.

### Stage 2 — Dual Binary XGBoost Classifiers

Rather than a single 3-class model, the pipeline trains **two independent binary models**:

| Model | Target | Output |
|-------|--------|--------|
| **Long model** (`long_model.pkl`) | P(21-bar return > +3%) | Confidence for long entry |
| **Short model** (`short_model.pkl`) | P(21-bar return < −3%) | Confidence for short entry |

Both models are trained on the full S&P 500 universe simultaneously (pooled training), preventing overfitting to any single ticker's history. The long model fires when its confidence exceeds the threshold; the short model fires independently.

**Training method:** Walk-forward cross-validation — the model is never trained on future data. A 504-bar (~2 year) training window slides forward in 63-bar (~3 month) steps, producing 52 out-of-sample folds.

**MACD conviction weighting:** Training rows where a MACD crossover occurred are weighted 4× higher than non-crossover bars. This focuses model capacity on the exact setups we care most about, without discarding the context that other bars provide.

### Stage 3 — Regime-Gated Signal Logic

```
long_conf  ≥ threshold  →  LONG  (all regimes except choppy)
short_conf ≥ threshold  →  SHORT (bear regime only)
otherwise               →  FLAT  (no trade)
```

The bear-regime short gate emerged directly from backtested evidence: short signals in bull and choppy regimes had a negative mean P&L (−0.99% per trade before gating). Restricting shorts to bear regime alone lifted mean P&L from 0.12% → 0.58% per trade and pushed profit factor from 1.047 → 1.211.

---

## Input Features (46 total)

### MACD Features (13)

| Feature | Description |
|---------|-------------|
| `macd_line` | EMA(12) − EMA(26) |
| `macd_signal` | EMA(9) of MACD line |
| `macd_histogram` | MACD line − signal |
| `macd_hist_slope_1` | 1-bar histogram change |
| `macd_hist_slope_3` | 3-bar histogram change |
| `macd_crossover` | Bullish/bearish line-cross (last 3 bars) |
| `macd_divergence` | Price/MACD divergence |
| `macd_bars_since_cross` | Bars elapsed since last crossover — distinguishes fresh vs stale signals |
| `macd_cross_strength` | \|MACD line − signal\| — continuous conviction gauge |
| `macd_above_zero` | MACD line above zero (bull territory) |
| `macd_hist_acceleration` | Second derivative of histogram — expanding or fading momentum |
| `macd_zero_cross` | MACD line crossed zero (trend-change confirmation) |
| `macd_weekly_bull` | Weekly MACD above its signal line — higher-timeframe trend alignment |

The histogram slope features detect a crossover *building* before it fires — the histogram converging toward zero is a leading signal. `macd_regime_alignment` (a derived feature computed at training time) fires when a MACD crossover and HMM regime point in the same direction simultaneously.

### Other Technical Features (26)

| Category | Features |
|----------|----------|
| **RSI** | Level, slope, overbought/oversold flags |
| **Trend** | EMA fast (50), EMA slow (200), price/EMA ratio, price/EMA%, golden cross, death cross, above EMA flag |
| **ADX** | Level, +DI, −DI, trend flag |
| **Volatility** | ATR, ATR%, Bollinger Band width, realised vol, 60-day rolling return, return autocorrelation |
| **Volume** | Volume/MA ratio, OBV slope, volume-on-crossover confirmation flag |
| **Context** | Sector (label-encoded), HMM regime |

### Macro Features (6)

Six market-wide features are computed daily from VIX and SPY data and injected into every ticker's feature vector:

| Feature | Description |
|---------|-------------|
| `vix_level` | Absolute VIX close |
| `vix_pct_rank` | VIX rolling 252-day percentile rank |
| `vix_high` | VIX above its 252-day median |
| `spy_above_200` | SPY above its 200-day EMA |
| `spy_rsi` | SPY 14-day RSI |
| `spy_return_20` | SPY 20-day return |

---

## Signal Filters

A signal is emitted only when all of the following pass:

| Filter | Condition |
|--------|-----------|
| Confidence | `P(predicted class) ≥ 0.60` |
| Trend strength | `ADX ≥ 20` |
| Regime | Not choppy |
| Direction gate | Shorts: bear regime only |

---

## What Was It Trained On?

| Item | Detail |
|------|--------|
| **Universe** | 496 S&P 500 stocks (7 excluded — insufficient history) |
| **History** | 2010–present (~15 years per ticker) |
| **Total rows** | 1,801,872 bar-level observations |
| **Label horizon** | 21-bar (~1 month) forward return |
| **Label thresholds** | Long > +3%, Short < −3%, Flat otherwise |
| **Walk-forward folds** | 52 (504-bar train / 63-bar test, sliding) |
| **Out-of-sample rows** | 1,555,182 predictions |

### Walk-Forward Performance (out-of-sample)

| Metric | Value |
|--------|-------|
| Signal Sharpe (cost-adjusted) | **2.67** |
| Signal hit rate | **58.7%** |
| Mean signal return | **+1.74%** per 21-bar hold |
| % bars with active signal | 34.0% |
| Long accuracy | 56.7% |
| Short accuracy | 67.4% |

### Backtested Trade Statistics

Full bar-by-bar simulation using limit entries, ATR-based stops/targets, signal-reversal exits, and round-trip commissions:

| Metric | Value |
|--------|-------|
| Filled trades | 424,064 |
| Win rate | **46.5%** |
| Avg win | +7.20% |
| Avg loss | −5.17% |
| Win/loss ratio | 1.39 |
| Profit factor | **1.211** |
| Mean P&L / trade | **+0.58%** |
| Avg hold | 11.5 bars |
| Backtest Sharpe | 11.56 |
| Sortino | 31.02 |

**By direction:**

| Direction | Win rate | Mean P&L |
|-----------|----------|----------|
| Long (all regimes) | 46.9% | +0.74% |
| Short (bear only) | 41.0% | −1.31% |

**By regime:**

| Regime | Win rate | Mean P&L |
|--------|----------|----------|
| Bear | 49.7% | +1.05% |
| Bull | 46.3% | +0.44% |
| Choppy | 44.5% | +0.36% |

### Top 10 Features by SHAP Importance

| Rank | Feature | Mean \|SHAP\| |
|------|---------|--------------|
| 1 | `vix_level` — absolute fear gauge | 0.161 |
| 2 | `atr_pct` — volatility-adjusted range | 0.151 |
| 3 | `vix_pct_rank` — VIX percentile rank | 0.137 |
| 4 | `spy_rsi` — broad market momentum | 0.125 |
| 5 | `spy_return_20` — recent market trend | 0.113 |
| 6 | `sector` — cross-sectional context | 0.053 |
| 7 | `spy_above_200` — market regime anchor | 0.046 |
| 8 | `macd_signal` — MACD EMA signal line | 0.042 |
| 9 | `atr` — absolute volatility | 0.040 |
| 10 | `rolling_return_60` — momentum | 0.032 |

Macro features dominate the top 7 — market-wide conditions (VIX, SPY state) are the strongest predictors of whether any individual stock signal will follow through. `macd_signal` entered the top 10 (rank 8) after MACD crossover bars were upweighted 4× during training.

---

## Architecture

```
Raw OHLCV (yfinance)  +  VIX / SPY macro
            │
            ▼
┌───────────────────────┐
│    Feature Pipeline   │  45 tech features · 6 macro · labels
│    496 tickers        │  1.8M rows · pooled parquet
└──────────┬────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌──────────┐  ┌────────────────────────────────┐
│   HMM    │  │   Dual Binary XGBoost          │
│ 3-state  │  │   long_model.pkl               │
│ regime   │  │   short_model.pkl              │
└────┬─────┘  │   Walk-forward CV (52 folds)   │
     │        │   MACD crossover 4× weighting  │
     └───────▶│   macd_regime_alignment feature│
              └──────────────┬─────────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Signal Gate        │
                  │   · ADX ≥ 20         │
                  │   · Conf ≥ 0.60      │
                  │   · No choppy signals│
                  │   · Shorts: bear only│
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │    Trade Setup       │  Entry · Stop · Target · Sizing
                  └──────┬──────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        ┌──────────┐      ┌─────────────────┐
        │  Alerts  │      │   Streamlit      │
        │ Email/TG │      │   Dashboard      │
        └──────────┘      │   5 pages        │
                          └─────────────────┘
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
python -m scheduler.scheduler                    # start daemon
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

`Python 3.11` · `XGBoost` · `hmmlearn` · `SHAP` · `Streamlit` · `Plotly` · `APScheduler` · `yfinance` · `pandas` · `numpy`
