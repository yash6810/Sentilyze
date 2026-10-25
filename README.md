---
title: Sentilyze Multi-Agent Quant Engine
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: "1.44.1"
app_file: app.py
pinned: false
license: apache-2.0
---

# Sentilyze — Autonomous Multi-Agent Quantitative Trading & NLP Intelligence Platform

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Code Style: Black](https://img.shields.io/badge/Code%20Style-Black%2025.1.0-000000?style=for-the-badge&logo=python&logoColor=white)](https://github.com/psf/black)
[![Test Suite](https://img.shields.io/badge/Tests-35%2F35%20Passing-10B981?style=for-the-badge&logo=github-actions&logoColor=white)](tests/)
[![Security: CodeQL](https://img.shields.io/badge/Security-Zero%20Pickle%20%7C%20Safe%20JSON-blueviolet?style=for-the-badge&logo=security&logoColor=white)](models/)
[![Open In Colab](https://img.shields.io/badge/Google%20Colab-FinMamba--JEPA%20Powerhouse-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/yash6810/Sentilyze/blob/main/notebooks/sentilyze_finmamba_jepa_powerhouse.ipynb)
[![Streamlit Interface](https://img.shields.io/badge/Mission%20Control-5%20Master%20Cockpits-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://sentilyze.streamlit.app/)
[![Paper Portfolio](https://img.shields.io/badge/Portfolio-+$54,092%20Profit%20(62.6%25%20Win)-10B981?style=for-the-badge&logo=cashapp&logoColor=white)](results/paper_portfolio.json)
[![Discord Alerts](https://img.shields.io/badge/Live%20Alerts-Discord%20Hub-5865F2?style=for-the-badge&logo=discord&logoColor=white)](src/alerts.py)

<p align="center">
  <b>An Autonomous Quantitative Multi-Agent Trading Engine combining Transformer Sentiment NLP (FinBERT), Walk-Forward Machine Learning (Native XGBoost Across S&P 500 Equities), 5-Agent Deliberation Council, Continuous-Time FinMamba-JEPA Foundation Models, and Institutional Execution Routing.</b>
</p>

[**🚀 Live App**](https://sentilyze.streamlit.app/) • [**🧠 FinMamba-JEPA Colab**](https://colab.research.google.com/github/yash6810/Sentilyze/blob/main/notebooks/sentilyze_finmamba_jepa_powerhouse.ipynb) • [**⚡ Quickstart**](#-quickstart-guide) • [**🏛️ Multi-Agent Council**](#-grounded-5-agent-deliberation-council) • [**💼 Live Portfolio ($136k)**](#-live-paper-trading-portfolio-state) • [**📂 5 Master Cockpits**](#-interactive-mission-control--5-master-cockpits)

</div>

---

## 🌟 Deployed & Validated Systems

Sentilyze moves beyond generic retail indicators and conversational text prompts into **empirically validated quantitative engineering**:

1. 🏛️ **5-Agent Quantitative Decision Council**: Fully deployed in [`src/agent_committee.py`](src/agent_committee.py). Technical Alpha, Sentiment NLP, Fundamental Forensic Valuation, and Tape Scouts deliberate under a Chief Risk Officer (CRO) arbitrator before capital is committed.
2. 🧠 **FinMamba-JEPA Foundation Model**: Fully implemented in [`src/finmamba_jepa.py`](src/finmamba_jepa.py) and runnable via Google Colab GPU. Combines Continuous-Time S6 State Space, FiLM catalyst cross-attention, non-generative JEPA latent liquidity prediction, and Bayesian epistemic uncertainty gates.
3. ⚡ **Jev TypeSafe Catalyst Engine**: Deployed in [`src/jev_catalyst.py`](src/jev_catalyst.py). Ingests real-time catalysts in sub-15ms, producing strictly typed Pydantic risk enums (`TIER_1_MAX_CONVICTION`, `TIER_4_VETO`) with zero token hallucinations.
4. 🌊 **Hawkes Order-Flow Cascade Engine**: Deployed in [`src/hawkes_cascade_engine.py`](src/hawkes_cascade_engine.py). Tracks self-exciting order flow point-processes, measuring the branching ratio $\eta \ge 1.0$ to freeze buying before liquidity cascade flash-crashes.
5. 📊 **Multimodal Chart Structure Analyst**: Deployed in [`src/multimodal_chart_analyst.py`](src/multimodal_chart_analyst.py). Programmatically isolates Smart Money Concepts (SMC) liquidity sweeps, Fair Value Gaps (FVG), and Gemini Vision multi-timeframe chart structure.
6. 🔄 **Kalman Pairs Trading & Cointegration**: Deployed in [`src/kalman_statarb_engine.py`](src/kalman_statarb_engine.py). Dynamic Kalman filter tracking of rolling hedge ratios and Ornstein-Uhlenbeck half-life mean reversion.
7. 🚀 **Adaptive Institutional Execution Router**: Deployed in [`src/adaptive_execution_router.py`](src/adaptive_execution_router.py). Selects optimal execution pathways (ISOR, TWAP, VWAP, POV) and computes non-linear market impact using the Almgren-Chriss square-root model.
8. 🛡️ **Forensic Accounting & Zero-Hallucination Valuation**: Deployed in [`src/forensic_accounting.py`](src/forensic_accounting.py). Evaluates deterministic Piotroski F-Score, Altman Z-Score, and 2-Year Beneish M-Score manipulation detection.
9. 🤖 **Autonomous Paper Trader**: Deployed in [`src/autonomous_trader.py`](src/autonomous_trader.py). Actively manages risk, staged scaling (+50% banked at TP1, breakeven stop-trail, runner to TP2), and logs execution to [`results/executed_trades.csv`](results/executed_trades.csv).

---

## 💼 Live Paper Trading Portfolio State

Sentilyze maintains a strict, verified live paper portfolio state ([`results/paper_portfolio.json`](results/paper_portfolio.json) and [`results/executed_trades.csv`](results/executed_trades.csv)):

| Portfolio Metric | Deployed Production Value | Operational Status |
| :--- | :---: | :--- |
| **Starting Capital Baseline** | **$100,000.00** | Initial Balance |
| **Total Liquid Equity** | **$136,489.18** | **+54.09% Realized Net Gain** |
| **Realized Net Profit** | **+$54,092.72** | **100% Banked Cash Reserves** |
| **Cash Reserves (Dry Powder)** | **$136,489.18** | **100% Available Capital (0 Open Risk)** |
| **Trade Win Rate** | **62.6%** | **57 Wins / 35 Losses (91 Closed Trades)** |
| **Open Market Exposure** | **0 Active Positions** | **Flat (Preserving Capital in Cash)** |

> [!IMPORTANT]
> **Strict Portfolio Preservation Rule**: `results/paper_portfolio.json` and `results/executed_trades.csv` represent the live paper trading state ($136,489.18 cash, +$54,092.72 profit). They are permanently preserved and protected against wipes or overwrites across all tests and builds.

---

## 🏛️ Grounded 5-Agent Deliberation Council

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │            5-AGENT GROUNDED DECISION COUNCIL           │
                                  └────────────────────────────────────────────────────────┘
                                                              │
         ┌─────────────────────┬──────────────────────────────┼──────────────────────────────┬─────────────────────┐
         │                     │                              │                              │                     │
┌────────▼────────┐   ┌────────▼────────┐            ┌────────▼────────┐            ┌────────▼────────┐   ┌────────▼────────┐
│ 1. 📈 TECHNICAL │   │ 2. 📰 FINBERT   │            │ 3. 🏛️ SEC DCF   │            │ 4. ⚡ PRICE     │   │ 5. 🥊 RED-TEAM  │
│    ALPHA AGENT  │   │    SENTIMENT    │            │    FORENSICS    │            │    TAPE SCOUT   │   │    SPECIALIST   │
├─────────────────┤   ├─────────────────┤            ├─────────────────┤            ├─────────────────┤   ├─────────────────┤
│ • Alpha158 Flow │   │ • NewsNLP Stream│            │ • Piotroski F   │            │ • Microstructure│   │ • Hawkes Cascade│
│ • Wavelet MODWT │   │ • Signed Score  │            │ • Altman Z-Score│            │ • Order Flow PoC│   │ • Epistemic Gate│
│ • Multi-TF RSI  │   │ • Jev Fast-Cat  │            │ • 2-Yr Beneish M│            │ • Volume Surges │   │ • Devil Advocate│
└────────┬────────┘   └────────┬────────┘            └────────┬────────┘            └────────┬────────┘   └────────┬────────┘
         │                     │                              │                              │                     │
         └─────────────────────┴──────────────────────────────┼──────────────────────────────┴─────────────────────┘
                                                              │
                                                 ┌────────────▼────────────┐
                                                 │ 🛡️ CHIEF RISK OFFICER   │
                                                 ├─────────────────────────┤
                                                 │ • Conformal Half-Kelly  │
                                                 │ • Grossman-Zhou Floor   │
                                                 │ • Final Veto Authority  │
                                                 └────────────┬────────────┘
                                                 ┌────────────▼────────────┐
                                                 │ EXECUTED ORDER WITH ATR │
                                                 │ TP1 (+2.5 ATR) / TP2    │
                                                 │ Trailing Breakeven Stop │
                                                 └─────────────────────────┘
```

---

## ⚡ Quickstart Guide

### 1. Run in 1-Click (No Setup Required)
Open our foundation model powerhouse notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yash6810/Sentilyze/blob/main/notebooks/sentilyze_finmamba_jepa_powerhouse.ipynb)

---

### 2. Local Setup & Installation
```bash
# Clone the repository
git clone https://github.com/yash6810/Sentilyze.git
cd Sentilyze

# Create and activate virtual environment
python -m venv .venv
# On Windows PowerShell: .venv\Scripts\Activate.ps1
# On Linux/macOS: source .venv/bin/activate

# Install production dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

### 3. Run the CLI
Audit any stock ticker or inspect the live paper portfolio straight from your terminal:

```bash
# 🏛️ Run Multi-Agent Deliberation on any stock ticker
python sentilyze.py NVDA

# 💼 Inspect live paper portfolio ledger, equity, and realized gains
python sentilyze.py portfolio
```

---

### 4. Launch the Streamlit Mission Control
```bash
streamlit run app.py
```

---

### 5. Run the Validated Test Suite
Execute unit and regression tests across the production engines:
```bash
pytest tests/ -v
```

---

## 🖥️ Interactive Mission Control — 5 Master Cockpits

The Streamlit dashboard (`app.py`) organizes production quantitative modules across **5 Master Cockpits** with sub-second lazy loading and cached state:

```
Sentilyze Mission Control
├── 🎯 Cockpit 1: Live Trading & Alpha Cockpit
│   ├── 🔮 Tab 1: Directional Signals & Microstructure Alpha
│   ├── 🏛️ Tab 2: 5-Agent Committee War Room
│   ├── 🤖 Tab 3: Autonomous Paper Trader ($136k Liquid Capital)
│   ├── ⚡ Tab 4: Broker Execution Bridge (Alpaca API)
│   └── 🎙️ Tab 5: AI Pre-Market Audio Briefing (.mp3)
├── 🌊 Cockpit 2: Smart Money & Alternative Data
│   ├── 📡 Tab 1: Real-Time Market Anomaly Screener
│   ├── 📉 Tab 2: 3D Volatility Surface & Dark Pool Liquidity
│   ├── 📰 Tab 3: FinBERT News & Reddit Alternative Sentiment
│   └── 🏛️ Tab 4: Executive & Insider Transactions Radar
├── 💼 Cockpit 3: Portfolio & Risk Engine
│   ├── 💼 Tab 1: Hierarchical Risk Parity (HRP) & Half-Kelly Sizing
│   ├── 🧬 Tab 2: Portfolio Diversity & Pairwise Correlation Grader
│   └── 🌐 Tab 3: Real-Time Macro Liquidity & Treasury Yield Radar
├── 📈 Cockpit 4: Backtest & Performance Factsheet
│   ├── 📈 Tab 1: Walk-Forward Backtesting (Zero Lookahead)
│   ├── 👑 Tab 2: Multi-Strategy Tournament Arena
│   └── 📊 Tab 3: Institutional Hedge Fund Risk & Alpha Factsheet
└── 🧠 Cockpit 5: Deep Quant & Explainability (XAI)
    ├── 🧠 Tab 1: SHAP Feature Importance & Trees
    ├── 🔄 Tab 2: Market-Neutral Cointegration & Pairs Stat-Arb
    ├── 🌪️ Tab 3: Black Swan Crisis & Macro Stress Simulator
    └── 🕵️ Tab 4: Forensic Accounting (Beneish M-Score) & DCF Intrinsic Valuation
```

---

## 📊 Empirical Alpha Attribution & Benchmark Performance

Decomposition of strategy performance under real market friction ($0.10\%$ transaction fees, $0.05\%$ slippage) across walk-forward historical testing (2018–2026):

| Ticker | ML Strategy Return (%) | Win Rate (%) | Sharpe Ratio | Max Drawdown (%) | Buy & Hold Max DD (%) | ML Predictive Edge Share (%) | Risk Management Protection (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NVDA** | **+738.61%** | 52.67% | **0.90** | **-52.95%** | -86.07% | 24.5% | **201.2%** |
| **AAPL** | **+76.56%** | 50.27% | **0.48** | **-67.87%** | -83.52% | 18.2% | **174.2%** |
| **MSFT** | **+48.23%** | 49.03% | **0.37** | **-71.10%** | -84.82% | 14.1% | **346.7%** |
| **GOOGL** | **+63.78%** | 47.49% | **0.36** | **-75.47%** | -87.93% | **20.4%** | **79.6%** |
| **AMZN** | **+17.18%** | 43.75% | **0.40** | **-72.61%** | -85.28% | **15.6%** | **84.4%** |
| **META** | **+50.15%** | 47.25% | **0.42** | **-69.01%** | -95.76% | **24.8%** | **75.2%** |
| **TSLA** | **+138.03%** | 49.35% | **0.47** | **-88.67%** | -97.92% | **79.2%** | **20.8%** |
| **SPY** | **+77.03%** | 50.51% | **0.33** | **-59.55%** | -72.68% | **39.0%** | **61.0%** |
| **AVERAGE**| **+151.20%** | **48.79%**| **0.47** | **-69.65%** | **-86.75%** | **29.49%** | **142.96%** |

---

## 🛡️ Security, Model Serialization & Code Quality

- **Zero Insecure Deserialization**: In strict compliance with CodeQL `py/unsafe-deserialization`, models are stored exclusively in native **XGBoost JSON format** (`model.save_model()`) or PyTorch `weights_only=True` (`torch.load(..., weights_only=True)`). Zero `pickle` or `joblib` is permitted anywhere in the repository.
- **Pre-computed Results of Truth**: Streamlit Cloud reads directly from pre-computed artifacts in `results/`, ensuring sub-second page loads and zero cold-start GPU lag.
- **Code Standards**: The codebase is strictly formatted with `black==25.1.0` and validated against `flake8` (`E9,F63,F7,F82`) with 0 errors.

---

## 📄 License & Disclaimer

Distributed under the **Apache 2.0 License**. See [`LICENSE`](LICENSE) for more information.

> **Disclaimer**: *Sentilyze is an experimental quantitative research and algorithmic trading platform designed for education and systematic backtesting. It does not constitute investment advice. Live trading involves substantial risk of capital loss.*
