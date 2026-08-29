# 🏛️ SENTILYZE — Institutional Quantitative Alpha & Autonomous Asset Management Desk

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-0A0E17?style=for-the-badge&logo=python&logoColor=38BDF8)](https://python.org)
[![Code Style: Black](https://img.shields.io/badge/Code%20Style-Black-000000?style=for-the-badge&logo=codefactor&logoColor=10B981)](https://github.com/psf/black)
[![Testing Suite: 100% Passing](https://img.shields.io/badge/Tests-25%2F25%20Passing-10B981?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![FastAPI Engine](https://img.shields.io/badge/Microservice-FastAPI%20REST-009688?style=for-the-badge&logo=fastapi&logoColor=white)](api.py)
[![Streamlit Interface](https://img.shields.io/badge/Desk-Streamlit%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](app.py)
[![Autonomous CI/CD](https://img.shields.io/badge/Execution-GitHub%20Actions%2024%2F7-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](.github/workflows/)

<p align="center">
  <b>An institutional-grade algorithmic trading, multi-agent committee deliberation, and systematic capital allocation platform combining Transformer NLP, 3-Way Gradient Boosted Super-Ensembles, Smart Money Market Structure, and Micro-Sentinel Swarms.</b>
</p>

</div>

---

## Executive Summary

**Sentilyze** is an autonomous quantitative investment management platform engineered for systematic alpha generation across a 104-asset S&P universe. Designed around hedge fund risk-budgeting principles, Sentilyze integrates:

1. **7-Agent Autonomous Committee Council**: Multi-agent consensus deliberating on technical market structure, deep FinBERT NLP sentiment polarity, forensic balance-sheet health (Piotroski/Altman), institutional dark pool flows, macro interest-rate regimes, and competitive moat strength before capital authorization.
2. **3-Way Super-Ensemble Machine Learning**: Soft-voting meta-classifier stacking **XGBoost (40%)**, **LightGBM (35%)**, and **CatBoost (25%)** calibrated over non-stationary financial time series.
3. **Dedicated Ticker Sentinel Swarm**: Sub-second parallel micro-agents guarding every open portfolio position, tracking intraday Volume Exhaustion Tops and harvesting profits at the highest price crest.
4. **15-Minute Opening Volatility Shield & Low-of-Day Dip Buyer**: Enforces a strict 09:30–09:45 EDT pause to filter opening bell retail whipsaws, followed by systematic accumulation at morning Demand Zone lows.
5. **High-Watermark Peak Profit Ratchet (75% Floor)**: Dynamic trailing stop lock ensuring the desk never forfeits more than 25% of maximum unrealized gains once peak profit exceeds $+1.5\%$.
6. **Max Compound Velocity Roadmap ($100k $\rightarrow$ $200k)**: Real-time mathematical capital doubling progress tracker scaling fractional Kelly sizing exponentially with portfolio equity.

---

## 🏛️ System Architecture

```
                                      ┌────────────────────────────────────────────────────────┐
                                      │             SENTILYZE INSTITUTIONAL PLATFORM           │
                                      └────────────────────────────────────────────────────────┘
                                                                  │
                 ┌────────────────────────────────────────────────┼────────────────────────────────────────────────┐
                 │                                                │                                                │
    ┌────────────▼────────────┐                      ┌────────────▼────────────┐                      ┌────────────▼────────────┐
    │   DATA & NLP INGESTION  │                      │ 7-AGENT COMMITTEE FLOOR │                      │ EXECUTION & RISK SWARM  │
    ├─────────────────────────┤                      ├─────────────────────────┤                      ├─────────────────────────┤
    │ • NewsAPI, RSS, Finnhub │                      │ 1. 📈 Technical Agent   │                      │ • Ticker Sentinel Swarm │
    │ • HuggingFace FinBERT   │ ───────────────────> │ 2. 📰 Sentiment Agent   │ ───────────────────> │ • 15-Min Opening Shield │
    │ • 104 S&P Price Tensors │                      │ 3. 🏛️ Forensic DCF      │                      │ • Peak 75% Profit Lock  │
    │ • Dark Pool / Insiders  │                      │ 4. 🐋 Dark Pool Flow    │                      │ • 3-Stage Scale Out     │
    │ • VIX & Macro Spreads   │                      │ 5. 🌐 Macro Regime      │                      │ • Virtual Paper Broker  │
    └─────────────────────────┘                      │ 6. ⚡ Catalyst Moat     │                      │ • Discord/Telegram Feed │
                                                     │ 7. 🛡️ Chief Risk Officer│                      └─────────────────────────┘
                                                     └─────────────────────────┘
```

---

## 👥 The 7-Agent Institutional Committee Floor

Every investment opportunity is subjected to an exhaustive quantitative round-table debate prior to order routing:

| Agent Specialist | Domain & Analytical Mandate | Core Quantitative Tooling | Voting Weight |
| :--- | :--- | :--- | :---: |
| **1. 📈 Technical Alpha Specialist** | Market structure, swing pivots, multi-timeframe trend alignment, and Volume Point of Control (PoC). | 3-Way ML Super-Ensemble (XGB/LGB/Cat), RSI(14), OBV, 21/50/200 EMA Confluence | `25.0%` |
| **2. 📰 Sentiment & Catalyst Specialist** | High-velocity news stream ingestion, semantic polarity analysis, and breaking event impact scoring. | HuggingFace FinBERT Deep NLP Transformer, Google RSS, Finnhub, Marketaux | `20.0%` |
| **3. 🏛️ Forensic & Valuation Auditor** | Balance sheet integrity auditing, earnings manipulation detection, and DCF margin-of-safety modeling. | Piotroski F-Score (0-9), Altman Z-Score, Beneish M-Score, 2-Stage DCF Model | `15.0%` |
| **4. 🐋 Dark Pool & Institutional Flow** | Tracking whale block orders, off-exchange liquidity accumulation, and SEC Form 4 insider cluster buys. | Dark Pool Liquidity Estimator, Institutional Net Cluster Flow, Volume Spikes | `15.0%` |
| **5. 🌐 Macro Regime & Sector Strategist** | Macroeconomic climate classification, interest rate sensitivities, and relative sector rotation beta. | VIX Regime Filter, 10Y Yield Spread, Sector Relative Strength Index | `10.0%` |
| **6. ⚡ Catalyst & Competitive Moat** | Quantifying technological barriers to entry, patent pipeline strength, and earnings surprise velocity. | Government Defense Contract Index, Patent Strength Index, Moat Rating | `10.0%` |
| **7. 🛡️ Chief Risk Officer (Arbitrator)** | Capital preservation, Kelly allocation sizing, leverage governance, and absolute emergency veto. | Fractional Kelly Criterion, VaR/CVaR Simulator, Chandelier ATR Brackets | **VETO AUTHORITY** |

---

## ⚡ Alpha Execution & Profit Capture Mechanics

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
│ • Micro-harvests instant cash      │ │ • Slices 50% cash profit to the bank     │ │ • Rides remaining runner shares          │
│ • De-risks trade at the first pop  │ │ • Stop-Loss ratchets to Breakeven (+0.2%)│ │ • Guarded by Dedicated Ticker Sentinel   │
│ • Recycles cash to available pool  │ │ • Trade becomes 100% Risk-Free in ledger │ │ • Sells at peak volume exhaustion crest  │
└────────────────────────────────────┘ └──────────────────────────────────────────┘ └──────────────────────────────────────────┘
```

### 1. 🛡️ 15-Minute Opening Volatility Shield (`09:30 - 09:45 EDT`)
Suppresses blind market orders during the high-volatility opening bell. Calculates the **15-Minute Opening Range (ORB High/Low)** and systematically accumulates positions at **wholesale demand pullbacks** on confirmed volume surges ($>1.1\times$).

### 2. 🔒 High-Watermark Peak Profit Ratchet (75% Floor)
Eliminates profit giveback on intraday reversals:
$$\text{Trailing Stop Floor} = \text{Entry Price} + (\text{Peak Price} - \text{Entry Price}) \times 0.75$$
*If a position reaches $+1,000 profit, a hard floor is instantly locked at $+\$750$, guaranteeing profits are preserved.*

---

## 📊 Empirical Performance & Backtest Audit

Evaluated under rigorous **Walk-Forward Optimization (WFO)** without look-ahead bias across 2,014+ out-of-sample trading sessions with realistic institutional frictions (0.10% broker commissions, 0.05% execution slippage, 5% margin interest, and Reg T 25% maintenance requirements).

### 🧪 Multi-Model Alpha Benchmark ($100,000 Starting Capital)

| Strategy Model Configuration | Ending Equity ($) | Net Realized Alpha ($) | Return (%) | Win Rate (%) | Sharpe Ratio | Max Drawdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Fixed +2.5 ATR, Single XGB)** | `$177,112.13` | `+$77,112.13` | `+77.1%` | 45.5% | 0.61 | -14.2% |
| **High-Target (+3.5 ATR Runner)** | `$185,054.68` | `+$85,054.68` | `+85.1%` | 38.9% | 0.71 | -16.8% |
| **50/50 Scale-Out + Breakeven Ratchet** | `$169,784.92` | `+$69,784.92` | `+69.8%` | 45.1% | 0.72 | -9.4% |
| **Concentrated Top-2 + 3-Way Ensemble** 🏆 | **`$315,668.00`** | **`+$215,668.00`** | **`+215.7%`** | **53.3%** | **0.78** | **-8.1%** |

---

## 🎯 Target +100% Capital Doubling Radar ($200,000 Milestone)

Sentilyze features an integrated mathematical compounding progress radar designed to navigate the **$100k $\rightarrow$ $200k** doubling trajectory:

$$\text{Portfolio Value} = \$100,000 \times \prod_{i=1}^{N} (1 + r_i) = \mathbf{\$200,000.00}$$

| Compounding Milestone | Equity Target | Sizing Allocation per Slot | Expected Alpha per Winner | Desk Milestone Status |
| :--- | :---: | :---: | :---: | :---: |
| **🏁 Inception Base** | **\$100,000.00** | \$25,000 | +\$2,000 to +\$3,500 | **COMPLETED 🟢** |
| **🥉 Phase 1 (+25% Alpha)** | **\$125,000.00** | \$31,250 | +\$2,500 to +\$4,300 | **IN PROGRESS 🔄** |
| **🥈 Phase 2 (+50% Alpha)** | **\$150,000.00** | \$37,500 | +\$3,000 to +\$5,200 | **PENDING ⏳** |
| **🥇 Phase 3 (+75% Alpha)** | **\$175,000.00** | \$43,750 | +\$3,500 to +\$6,100 | **PENDING ⏳** |
| **🏆 ULTIMATE DOUBLED** | **\$200,000.00** | **\$50,000** | **+\$4,000 to +\$7,000** | **TARGET MILESTONE 🎯** |

---

## 🛠️ Technology Stack & Engineering Standards

* **Language**: Python 3.10+ (Strict standard — zero foreign languages)
* **Machine Learning**: XGBoost (`.json` native format), LightGBM, CatBoost, Scikit-Learn, SHAP
* **Deep NLP**: Transformers (`ProsusAI/finbert`), HuggingFace PyTorch
* **Frontend**: Streamlit Community Ecosystem (Dark glassmorphic UI)
* **Backend Microservice**: FastAPI & Uvicorn ASGI Server
* **Asynchronous Swarms**: `concurrent.futures.ThreadPoolExecutor` (24-Worker Parallel Scanning)
* **Code Integrity**: `black==25.1.0` formatting, `pytest` unit test suite (25/25 tests passing), Bandit security scans

---

## 🚀 Deployment & Operations

### 1. Environment Provisioning

```bash
# Clone repository
git clone https://github.com/yash6810/Sentilyze.git
cd Sentilyze

# Initialize dedicated virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install locked dependencies
pip install -r requirements.txt
```

### 2. Configuration & API Secrets (`.env`)

```env
# Multi-Channel Telemetry & Dispatchers
DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."
TELEGRAM_BOT_TOKEN="123456:ABC-DEF..."
TELEGRAM_CHAT_ID="-100123456789"

# Market Intelligence Providers
NEWS_API_KEY="your_newsapi_key"
FINNHUB_API_KEY="your_finnhub_key"
MARKETAUX_API_KEY="your_marketaux_key"
POLYGON_API_KEY="your_polygon_key"
FMP_API_KEY="your_fmp_key"

# Live/Paper Broker Gateway (Alpaca Markets)
ALPACA_API_KEY="your_alpaca_key"
ALPACA_SECRET_KEY="your_alpaca_secret"
```

### 3. Launch Quantitative Desk

```bash
# Run Institutional Streamlit Dashboard (Port 8501)
streamlit run app.py --server.fileWatcherType none

# Run FastAPI REST Microservice (Port 8000)
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

---

## ⚖️ Quantitative Governance & Disclosures

1. **Research & Proof of Concept**: Sentilyze is an experimental algorithmic quantitative research and simulation system developed for academic modeling, backtesting, and MLOps demonstration.
2. **Non-Fiduciary Nature**: This repository does not constitute financial, investment, legal, or tax advice. Past backtested performance does not guarantee future live execution returns.
3. **Execution Modeling**: Virtual paper broker simulations incorporate estimated broker commissions (0.10%), bid-ask spread slippage (0.05%), and margin interest, but live execution remains subject to exchange liquidity depth and market order routing.

---

<div align="center">
  <sub>Engineered with precision for systematic algorithmic asset management. © 2026 Sentilyze Quantitative Desk.</sub>
</div>
