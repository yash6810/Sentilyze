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

**Sentilyze** is an open-source quantitative trading research platform and reproducible MLOps showcase. The project investigates whether combining **deep NLP financial sentiment (FinBERT)** with **multi-timeframe technical momentum** can generate positive risk-adjusted expectancy when paired with **strict asymmetric trade management**.

### 💡 The Core Finding & Empirical Alpha Attribution

In daily financial time series, directional return prediction is notoriously non-stationary (out-of-sample directional accuracy is $\sim 49\%\text{--}53\%$). Sentilyze includes an automated **Alpha Attribution Engine** ([`src/attribution_analysis.py`](src/attribution_analysis.py)) that decomposes strategy performance against zero-alpha baselines under identical market friction ($0.10\%$ transaction fees, $0.05\%$ slippage, $5\%$ margin rate) using **50 Monte Carlo trials per ticker** over the 2018–2026 evaluation window:

#### **1. Multi-Ticker Empirical Attribution Decomposition (50 Monte Carlo Trials per Asset)**
| Ticker | ML Strategy Return (%) | ML Win Rate (%) | ML Sharpe | ML Max DD (%) | Random Entries Return (%) | Random Sharpe | Random Max DD (%) | Always-Long Max DD (%) | ML Predictive Edge Share (%) | Risk Management Baseline Share (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NVDA** | **+738.61%** | 52.67% | 0.90 | **-52.95%** | +1,486.21% | 1.84 | -83.73% | -86.07% | **0.0%** | **201.2%** |
| **AAPL** | **+76.56%** | 50.27% | 0.48 | **-67.87%** | +133.40% | 1.42 | -75.95% | -83.52% | **0.0%** | **174.2%** |
| **MSFT** | **+48.23%** | 49.03% | 0.37 | **-71.10%** | +167.21% | 1.37 | -74.83% | -84.82% | **0.0%** | **346.7%** |
| **GOOGL** | **+63.78%** | 47.49% | 0.36 | **-75.47%** | +50.74% | 1.42 | -83.78% | -87.93% | **20.4%** | **79.6%** |
| **AMZN** | **+17.18%** | 43.75% | 0.40 | **-72.61%** | +14.50% | 1.51 | -78.54% | -85.28% | **15.6%** | **84.4%** |
| **META** | **+50.15%** | 47.25% | 0.42 | **-69.01%** | +37.71% | 1.58 | -86.09% | -95.76% | **24.8%** | **75.2%** |
| **TSLA** | **+138.03%** | 49.35% | 0.47 | **-88.67%** | +28.72% | 1.97 | -92.60% | -97.92% | **79.2%** | **20.8%** |
| **SPY** | **+77.03%** | 50.51% | 0.33 | **-59.55%** | +46.96% | 1.01 | -68.45% | -72.68% | **39.0%** | **61.0%** |
| **AVERAGE** | **+151.20%** | **48.79%** | **0.47** | **-69.65%** | **+245.68%** | **1.52** | **-80.50%** | **-86.75%** | **22.38%** | **130.39%** |

#### **2. 4-Agent Committee Ablation Matrix (400-Day Out-of-Sample Horizon)**
| Ticker | Full Committee Sharpe | Full Committee Return (%) | Full Committee Max DD (%) | Minus-Forensic Sharpe | Minus-Sentiment Sharpe | Minus-CRO Sharpe | Technical-Only Sharpe |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NVDA** | **0.46** | +15.98% | -23.82% | 0.57 | 0.04 | -0.39 | 0.04 |
| **AAPL** | **0.70** | +26.89% | -16.83% | 0.76 | -0.11 | 0.19 | -0.11 |
| **MSFT** | **0.45** | +13.38% | -25.60% | 0.42 | 0.36 | -0.32 | 0.36 |
| **GOOGL** | **-0.03** | -4.84% | -23.76% | 0.08 | 0.36 | -0.38 | 0.36 |
| **AMZN** | **0.03** | -5.31% | -20.48% | -0.02 | -0.89 | -0.32 | -0.89 |
| **AVERAGE** | **+0.32** | **+9.22%** | **-22.10%** | **+0.36** | **-0.05** | **-0.24** | **-0.05** |

#### **Key Empirical Conclusions:**
1. **Asymmetric Trade Management Drives the Positive Expectancy Baseline**: Under $+2.5\times\text{ATR}$ take-profit, $-1.5\times\text{ATR}$ stop-loss, and breakeven ratchets, even purely randomized trade entries yield positive returns (average $+245.68\%$, Sharpe $1.52$) in an upward trending regime. However, random and unmanaged long signals experience catastrophic drawdowns (**$-80.50\%\text{ to }-86.75\%$**).
2. **The ML Model Acts as a Drawdown and Tail-Risk Shield**: Predictive ML signals reduce trade count, avoid adverse regimes, and **contract maximum drawdown from $-86.75\%$ down to $-69.65\%$** (and down to $-22.10\%$ in the 4-agent council).
3. **CRO Risk Management & FinBERT Sentiment are Critical**: Removing Chief Risk Officer sizing collapses Sharpe from $+0.32 \rightarrow -0.24$; removing FinBERT sentiment turns the council net-negative ($-0.05$ Sharpe). Technical indicators alone without sentiment and risk management fail to maintain positive risk-adjusted alpha.

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
