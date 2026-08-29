# Sentilyze — Systematic Sentiment & Momentum Trading Research Platform

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Code Style: Black](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)
[![Tests: Passing](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](tests/)
[![FastAPI Engine](https://img.shields.io/badge/Microservice-FastAPI%20REST-009688.svg)](api.py)
[![Streamlit Interface](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](app.py)
[![Autonomous CI/CD](https://img.shields.io/badge/Automation-GitHub%20Actions-2088FF.svg)](.github/workflows/)

<p align="center">
  <b>An open-source quantitative research platform combining Transformer NLP (FinBERT), Walk-Forward Machine Learning (XGBoost), and Asymmetric Risk Management across US equities.</b>
</p>

</div>

---

## 🔬 Project Overview & Core Thesis

**Sentilyze** is an open-source quantitative trading research system and reproducible MLOps showcase. The project investigates whether combining **deep NLP financial sentiment** with **multi-timeframe technical momentum** can generate positive risk-adjusted expectancy when paired with **strict asymmetric trade management**.

### 💡 The Core Finding & Empirical Alpha Attribution
In daily financial time series, directional return prediction is notoriously non-stationary (out-of-sample directional accuracy is $\sim 50\%\text{--}53\%$). Sentilyze includes an automated **Alpha Attribution Engine** ([`src/attribution_analysis.py`](src/attribution_analysis.py)) that decomposes strategy performance against zero-alpha baselines under identical market friction ($0.10\%$ fees, $0.05\%$ slippage, $5\%$ margin rate):

#### **Attribution Decomposition Matrix (NVDA 2018–2026, 2,514 Trading Days)**
| Strategy Configuration | Total Return (%) | Win Rate (%) | Sharpe Ratio | Max Drawdown (%) | Trade Count | Primary Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 🎲 **Random Signal Baseline + Asymmetric ATR Rules** | **+4,185.16%** | 54.41% | 1.85 | -83.18% | 350 | **Positive Expectancy Baseline** |
| 📈 **Always-Long Baseline + Asymmetric ATR Rules** | **+13,029.48%** | 54.03% | 3.06 | -86.07% | 422 | **Market Beta Expansion** |
| 🧠 **Full ML Strategy (FinBERT + XGBoost Filter)** | **+3,178.95%** | **57.14%** | 1.17 | **-42.19%** | **266** | **Volatility & Drawdown Shield** |
| 🏛️ **Buy & Hold Benchmark** | +14,610.76% | N/A | ~1.10 | -66.36% | 1 | Passive Reference |

#### **Key Attribution Takeaways:**
1. **Trade Management Drives Baseline Expectancy**: Even purely random entries produce a positive Sharpe of $1.85$ under the $+2.5\text{ ATR}$ Take-Profit / $-1.5\text{ ATR}$ Stop-Loss / Breakeven Ratchet framework. The baseline positive edge is mathematical payoff geometry.
2. **The ML Model Acts as a Drawdown Shield**: The FinBERT + XGBoost classifier filters out 156 high-risk false breakouts, **halving Maximum Drawdown from $-86.07\%$ down to $-42.19\%$** and lifting the win rate to $57.14\%$.

---

## 📁 Clean Repository Structure

The repository maintains a strict separation between **100% Real Production Engine** code, **Persisted Out-of-Sample Results**, and **Isolated Research Sandboxes**:

```
Sentilyze/
├── app.py                          # Streamlit Live Trading Dashboard (Layout: Wide)
├── api.py                          # FastAPI REST Microservice with SHAP Explainability
│
├── src/                            # 🟢 PRODUCTION QUANT ENGINE (100% Real Data)
│   ├── agent_committee.py          # 4-Agent Grounded Decision Council & Quarter-Kelly
│   ├── autonomous_trader.py        # 24-Worker Parallel Live Scanner & Auto-Execution
│   ├── backtesting.py              # Walk-Forward Optimization & Monte Carlo Significance
│   ├── attribution_analysis.py     # Signal vs Trade-Management Decomposition Engine
│   ├── forensic_accounting.py      # Real 8-Variable 2-Year Comparative Beneish M-Score
│   ├── fundamental_valuation.py    # Piotroski F-Score, Altman Z-Score & DCF Fair Value
│   ├── opening_range_engine.py     # 15-Minute Opening Volatility Shield (09:30-09:45 EST)
│   ├── sentiment_analysis.py       # HuggingFace FinBERT Transformer Pipeline
│   ├── statistical_arbitrage.py    # Engle-Granger Cointegration & OU Mean-Reversion
│   ├── paper_broker.py             # Realistic Virtual Execution Broker (Slippage & Margin)
│   └── ...
│
├── experimental/                   # 🧪 ISOLATED RESEARCH PROTOTYPES (Sandbox / Non-Trading)
│   ├── README.md                   # Prototype inventory & prospective API documentation
│   ├── dark_pool_radar.py          # Prototype ATS block print data structure
│   ├── insider_tracker.py          # Prototype SEC Form 4 cluster data structure
│   ├── patent_contract_radar.py    # Prototype USPTO patent momentum data structure
│   ├── quant_engine.py             # Standalone 8-pillar orchestration prototype
│   ├── rl_allocator.py             # Standalone MDP environment & policy gradient prototype
│   └── sec_filing_diff.py          # Prototype 10-K risk factor diffing
│
├── results/                        # 📊 BENCHMARK SOURCE OF TRUTH (Out-of-Sample Metrics)
│   ├── attribution_analysis.json   # Persisted 4-way Alpha Attribution results
│   ├── *_metrics.json              # Out-of-Sample WFO performance metrics per ticker
│   ├── *_portfolio.csv             # Historical daily equity curves and drawdowns
│   └── *_shap_summary.png          # SHAP global feature impact visualizations
│
├── tests/                          # 🧪 AUTOMATED PYTEST SUITE (100% Passing)
└── models/                         # 🧠 TRAINED PRODUCTION MODELS (Native XGBoost JSON)
```

---

## 👥 The Grounded 4-Agent Deliberation Council

All automated trade entries in [`src/agent_committee.py`](src/agent_committee.py) are deliberated by 4 specialized agents operating exclusively on real market data:

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │            4-AGENT GROUNDED DECISION COUNCIL           │
                                  └────────────────────────────────────────────────────────┘
                                                              │
             ┌────────────────────────────────────────────────┼────────────────────────────────────────────────┐
             │                                                │                                                │
┌────────────▼────────────┐                      ┌────────────▼────────────┐                      ┌────────────▼────────────┐
│ 1. 📈 TECHNICAL ALPHA   │                      │ 2. 📰 FINBERT SENTIMENT │                      │ 3. 🏛️ SEC FORENSICS     │
├─────────────────────────┤                      ├─────────────────────────┤                      ├─────────────────────────┤
│ • Real Price Action     │                      │ • HuggingFace FinBERT   │                      │ • Piotroski F-Score     │
│ • RSI(14) & SMA(200)    │                      │ • Live News Ingestion   │                      │ • Altman Z-Score & DCF  │
│ • Trend Momentum Regimes│                      │ • Semantic Confidence   │                      │ • 2-Yr Beneish M-Score  │
└────────────┬────────────┘                      └────────────┬────────────┘                      └────────────┬────────────┘
             │                                                │                                                │
             └────────────────────────────────────────────────┼────────────────────────────────────────────────┘
                                                              │
                                                 ┌────────────▼────────────┐
                                                 │ 4. 🛡️ CHIEF RISK OFFICER│
                                                 ├─────────────────────────┤
                                                 │ • Formulaic Kelly Sizing│
                                                 │ • Macro VIX Vol Gate    │
                                                 │ • Final Veto Authority  │
                                                 └────────────┬────────────┘
                                                              │
                                                 ┌────────────▼────────────┐
                                                 │ EXECUTED ORDER WITH ATR │
                                                 │ TP0 / TP1 / TP2 / SL    │
                                                 └─────────────────────────┘
```

| Agent Specialist | Analytical Domain | Methodology & Verification |
| :--- | :--- | :--- |
| **1. 📈 Technical Alpha Specialist** | Trend structure & momentum regimes. | Real historical OHLCV series, `RSI(14)`, `SMA(200)`, `EMA(21)`, and 5-day momentum. |
| **2. 📰 Sentiment Catalyst Specialist** | Breaking news semantic polarity. | Real live headlines scored via HuggingFace `ProsusAI/finbert` Transformer pipeline. |
| **3. 🏛️ Forensic & Valuation Auditor** | Audited financial reporting health & valuation. | Live balance sheets from `yfinance`: **Piotroski F-Score (0–9)**, **Altman Z-Score**, **2-Year Comparative Beneish M-Score**, and **DCF Margin of Safety**. (Abstains if filings are missing). |
| **4. 🛡️ Chief Risk Officer (CRO)** | Risk budgeting & volatility gate. | **Mathematical Quarter-Kelly sizing**, **VIX Volatility Filter** ($VIX > 26.0$), and strict consensus validation. |

---

## 📐 Mathematical Fractional Kelly Sizing

Position allocation is calculated directly from empirical WFO strategy win rates ($p \approx 0.533$) and win/loss payoff ratios ($b \approx 1.75$):

$$f^* = \frac{p \cdot b - (1 - p)}{b}$$

$$\text{Position Allocation \%} = \min\left(15.0\%, \max\left(0.0\%, 0.25 \times f^* \times 100\right)\right)$$

*If the calculated edge is non-positive ($p \cdot b \le 1 - p$) or the CRO exercises a veto, allocation dynamically drops to **0.0%**.*

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
2. **High-Watermark Peak Profit Ratchet (75% Floor)**: Once unrealized position profit exceeds $+1.5\%$, a trailing floor is locked at $75\%$ of peak profit, preventing winning trades from turning into losses.

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

### 2. Running the Streamlit Dashboard

```bash
streamlit run app.py --server.fileWatcherType none
```
Open **`http://localhost:8501`** in your browser.

### 3. Running the FastAPI REST Microservice

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
* **Interactive Swagger UI**: `http://localhost:8000/docs`
* **Inference Endpoint**:
  ```bash
  curl -X GET "http://localhost:8000/predict?ticker=NVDA"
  ```

### 4. Running the Test Suite & Attribution Engine

```bash
# Run full unit tests
pytest

# Run Alpha Attribution Decomposition
python -c "from src.attribution_analysis import run_attribution_decomposition; print(run_attribution_decomposition('NVDA', n_random_trials=30))"
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
