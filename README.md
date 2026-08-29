# Sentilyze — Systematic Sentiment & Momentum Trading Research Platform

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Code Style: Black](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)
[![Tests: Passing](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](tests/)
[![FastAPI Engine](https://img.shields.io/badge/Microservice-FastAPI%20REST-009688.svg)](api.py)
[![Streamlit Interface](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](app.py)
[![Autonomous CI/CD](https://img.shields.io/badge/Automation-GitHub%20Actions-2088FF.svg)](.github/workflows/)

<p align="center">
  <b>A research and algorithmic trading platform combining Transformer NLP (FinBERT), Walk-Forward Machine Learning (XGBoost), and Asymmetric Risk Management across US equity markets.</b>
</p>

</div>

---

## 🔬 Project Overview & Core Thesis

**Sentilyze** is an open-source quantitative trading research system and end-to-end MLOps showcase. The project investigates whether combining **deep NLP financial sentiment** with **multi-timeframe technical momentum** can generate positive risk-adjusted expectancy when paired with **strict asymmetric trade management**.

### 💡 The Core Finding & Attribution
Financial direction forecasting on daily equity returns is notoriously non-stationary (out-of-sample directional accuracy hovers between **50% to 53%**). Sentilyze's empirical backtests show that **the primary driver of positive expectancy is not raw predictive accuracy, but asymmetric risk/reward architecture**:
* **Asymmetric Payoff Structure**: Taking 50% profit at `+2.5 ATR` (`TP1`) and letting runners target `+4.5 ATR` (`TP2`), while strictly limiting downside risk to `-1.5 ATR` (`SL`).
* **Breakeven Ratchets & Peak Profit Floors**: Moving stop-losses to breakeven (`+0.2%`) once TP1 is reached and locking in $\ge 75\%$ of maximum unrealized gains once peak profits exceed $+1.5\%$.
* **Walk-Forward Validation**: Validated over 2,014+ out-of-sample trading days (~8 years) with zero look-ahead bias, benchmarked against logistic regression baselines and Monte Carlo permutation significance tests.

---

## ⚙️ System Architecture

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │               SENTILYZE QUANTITATIVE PIPELINE          │
                                  └────────────────────────────────────────────────────────┘
                                                              │
             ┌────────────────────────────────────────────────┼────────────────────────────────────────────────┐
             │                                                │                                                │
┌────────────▼────────────┐                      ┌────────────▼────────────┐                      ┌────────────▼────────────┐
│   DATA & NLP INGESTION  │                      │ 4-AGENT COMMITTEE FLOOR │                      │ EXECUTION & RISK ENGINE │
├─────────────────────────┤                      ├─────────────────────────┤                      ├─────────────────────────┤
│ • NewsAPI & RSS Feeds   │                      │ 1. 📈 Technical Alpha   │                      │ • 15-Min Opening Shield │
│ • HuggingFace FinBERT   │ ───────────────────> │ 2. 📰 Sentiment Catalyst│ ───────────────────> │ • Mathematical Kelly    │
│ • Daily Price Tensors   │                      │ 3. 🏛️ Forensic DCF      │                      │ • 3-Stage Scale-Out     │
│ • VIX Macro Indicator   │                      │ 4. 🛡️ Chief Risk Officer│                      │ • Virtual Paper Broker  │
└─────────────────────────┘                      └─────────────────────────┘                      └─────────────────────────┘
```

---

## 👥 The 4-Agent Deliberation Council

All potential trades are evaluated through a structured multi-factor committee grounded strictly in real verifiable data:

| Agent Specialist | Analytical Domain | Data Sources & Methodology |
| :--- | :--- | :--- |
| **1. 📈 Technical Alpha Specialist** | Market structure, moving average regimes, and momentum oscillators. | Real historical price series, `RSI(14)`, `SMA(200)`, `EMA(21)`, and 5-day momentum. |
| **2. 📰 Sentiment Catalyst Specialist** | Breaking news semantic polarity and media tone. | Real live headlines scored via HuggingFace `ProsusAI/finbert` Transformer pipeline. |
| **3. 🏛️ Forensic & Valuation Auditor** | Balance sheet health, operational momentum, and intrinsic valuation. | Live balance sheet filings from yfinance, calculating **Piotroski F-Score (0–9)**, **Altman Z-Score**, and **2-Stage DCF Margin of Safety**. |
| **4. 🛡️ Chief Risk Officer (CRO)** | Risk budgeting, position sizing, and volatility governance. | **Mathematical Fractional Kelly Criterion** ($f^* = \frac{p \cdot b - q}{b}$), **Macro VIX Volatility Gate** ($VIX > 26.0$), and dynamic ATR risk brackets. |

---

## 📐 Mathematical Fractional Kelly Sizing

Position sizing is dynamically calculated from measured strategy win rate ($p$) and win/loss payoff ratio ($b = \frac{\text{avg win}}{\text{avg loss}}$) using a conservative **Quarter-Kelly** fraction:

$$f^* = \frac{p \cdot b - (1 - p)}{b}$$

$$\text{Position Allocation \%} = \min\left(15.0\%, \max\left(0, 0.25 \times f^* \times 100\right)\right)$$

*If expected edge is negative or consensus fails, allocation dynamically drops to **0.0%**.*

---

## ⚡ Asymmetric Trade Execution Mechanics

```
                  ┌────────────────────────────────────────────────────────────────────────┐
                  │                 3-STAGE STAGED PROFIT SCALE-OUT ENGINE                 │
                  └────────────────────────────────────────────────────────────────────────┘
                                                      │
         ┌────────────────────────────────────────────┼────────────────────────────────────────────┐
         │                                            │                                            │
┌────────▼───────────────────────────┐ ┌──────────────▼───────────────────────────┐ ┌──────────────▼───────────────────────────┐
│ STAGE 1: TP0 (+1.0 ATR / ~+3.0%)   │ │ STAGE 2: TP1 (+2.5 ATR / ~+7.5%)         │ │ STAGE 3: TP2 (+4.5 ATR / ~+13.5%)        │
├────────────────────────────────────┤ ├──────────────────────────────────────────┤ ├──────────────────────────────────────────┤
│ • Micro-harvests early gains       │ │ • Slices 50% cash profit to the bank     │ │ • Rides remaining runner shares          │
│ • De-risks trade at first expansion│ │ • Stop-Loss ratchets to Breakeven (+0.2%)│ │ • Monitored for volume exhaustion tops   │
│ • Recycles cash to liquidity pool  │ │ • Trade becomes 100% Risk-Free in ledger │ │ • Hard stop floor: 75% of peak gains     │
└────────────────────────────────────┘ └──────────────────────────────────────────┘ └──────────────────────────────────────────┘
```

1. **15-Minute Opening Volatility Shield (`09:30 - 09:45 EDT`)**: Pauses market orders during opening bell whiplash, waiting for the initial 15-minute range to establish before seeking pullbacks.
2. **High-Watermark Peak Profit Ratchet (75% Floor)**: Once unrealized position profit exceeds $+1.5\%$, a trailing floor is locked at $75\%$ of peak profit, preventing winners from turning into losses.

---

## 📊 Empirical Walk-Forward Backtest Results

Evaluated under strict **Walk-Forward Optimization (WFO)** without look-ahead bias across 2,014+ out-of-sample trading days (~8 years) with realistic market frictions:
* **Broker Commission**: $0.10\%$ per trade
* **Execution Slippage**: $0.05\%$ per trade
* **Margin Borrowing Cost**: $5.0\%$ annualized interest
* **Maintenance Margin**: Reg T $25\%$ liquidation safeguard

### 🧪 4-Year Sizing & Management Benchmark ($100,000 Capital)

| Model Configuration | Ending Equity ($) | Net Profit ($) | Total Return (%) | Win Rate (%) | Sharpe Ratio | Max Drawdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline (Fixed +2.5 ATR, Single XGB)** | `$177,112.13` | `+$77,112.13` | `+77.1%` | 45.5% | 0.61 | -14.2% |
| **2. High Target (+3.5 ATR Runner)** | `$185,054.68` | `+$85,054.68` | `+85.1%` | 38.9% | 0.71 | -16.8% |
| **3. 50/50 Scale-Out & Breakeven Ratchet** | `$169,784.92` | `+$69,784.92` | `+69.8%` | 45.1% | 0.72 | -9.4% |
| **4. Concentrated Top-2 + Scale-Out (1.25x Lev)** 🏆 | **`$315,668.00`** | **`+$215,668.00`** | **`+215.7%`** | **53.3%** | **0.78** | **-8.1%** |

---

## 🛠️ Technology Stack & Engineering Standards

* **Language**: Python 3.10+ (Python ONLY)
* **Machine Learning**: XGBoost (`.json` native serialization — zero joblib/pickle), Scikit-Learn, SHAP
* **Deep NLP**: Transformers (`ProsusAI/finbert`), HuggingFace PyTorch
* **Frontend**: Streamlit (Dark glassmorphic layout)
* **Backend Microservice**: FastAPI & Uvicorn ASGI Server
* **Parallel Engine**: `concurrent.futures.ThreadPoolExecutor` (24-worker concurrency for sub-second universe scanning)
* **Quality & Testing**: `black==25.1.0` formatting, `pytest` unit test suite (100% passing), Bandit security scans

---

## 🚀 Quickstart Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yash6810/Sentilyze.git
cd Sentilyze

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration (`.env`)

```env
# Telemetry & Dispatchers (Optional)
DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
TELEGRAM_BOT_TOKEN="123456:ABC-DEF..."
TELEGRAM_CHAT_ID="-100123456789"

# Market Data API Keys
NEWS_API_KEY="your_newsapi_key"
FINNHUB_API_KEY="your_finnhub_key"
```

### 3. Running the Streamlit Dashboard

```bash
streamlit run app.py --server.fileWatcherType none
```
Open **`http://localhost:8501`** in your browser.

### 4. Running the FastAPI REST Microservice

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
* **Interactive Swagger UI**: `http://localhost:8000/docs`
* **Inference Endpoint**:
  ```bash
  curl -X GET "http://localhost:8000/predict?ticker=NVDA"
  ```

---

## ⚠️ Limitations & Disclosures

1. **Research & Proof of Concept**: Sentilyze is an experimental algorithmic quantitative research system developed for academic modeling, backtesting, and MLOps demonstration. It is not a registered investment advisor or broker-dealer.
2. **Market Non-Stationarity**: Past performance and backtested simulation results do not guarantee future returns. Financial markets exhibit regime shifts, liquidity shocks, and non-linear dynamics.
3. **Execution Modeling**: Virtual paper trading accounts incorporate commissions ($0.10\%$), slippage ($0.05\%$), and margin interest ($5.0\%$), but real-world execution remains subject to exchange liquidity and order-book queue depth.

---

<div align="center">
  <sub>Open-source quantitative research platform. Built with Python & Streamlit.</sub>
</div>
