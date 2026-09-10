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
[![Code Style: Black](https://img.shields.io/badge/Code%20Style-Black-000000?style=for-the-badge&logo=python&logoColor=white)](https://github.com/psf/black)
[![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-274%2B%20Tests%20Passing-10B981?style=for-the-badge&logo=github-actions&logoColor=white)](.github/workflows/)
[![S&P 500 Universe](https://img.shields.io/badge/S%26P%20500-526%20Models%20Trained-blueviolet?style=for-the-badge&logo=target&logoColor=white)](models/)
[![Open In Colab](https://img.shields.io/badge/Google%20Colab-1--Click%20Demo-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/yash6810/Sentilyze/blob/main/notebooks/demo.ipynb)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces%20Live%20App-yellow?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/clash1462/Sentilyze)
[![Streamlit Interface](https://img.shields.io/badge/Mission%20Control-5%20Master%20Cockpits-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://sentilyze.streamlit.app/)
[![Discord Alerts](https://img.shields.io/badge/Live%20Alerts-Discord%20Hub-5865F2?style=for-the-badge&logo=discord&logoColor=white)](src/alerts.py)
[![ML Framework](https://img.shields.io/badge/ML-XGBoost%20%2B%20FinBERT%20INT8%20%2B%20PyTorch%20RL-F59E0B?style=for-the-badge&logo=scikit-learn&logoColor=white)](src/modeling.py)

<p align="center">
  <b>A 24/7 Autonomous Hybrid Quantitative Multi-Agent Trading Engine combining Deep Transformer NLP (FinBERT INT8 Quantized), Walk-Forward Machine Learning (XGBoost across 526 S&P 500 Assets), 5-Agent Deliberation Council, Gen-3 Quant Alpha DAG Engine (+103.13% &alpha;), Deep Reinforcement Learning (PyTorch Continuous Actor-Critic), and Fractional Kelly Sizing across US Equities.</b>
</p>

[**🚀 Live App**](https://sentilyze.streamlit.app/) • [**⚡ 1-Click Colab**](https://colab.research.google.com/github/yash6810/Sentilyze/blob/main/notebooks/demo.ipynb) • [**⚡ Quickstart**](#-quickstart-guide) • [**🏛️ Multi-Agent Committee**](#-grounded-5-agent-deliberation-council) • [**💼 Live Portfolio ($152k)**](#-live-paper-trading-portfolio-state) • [**📂 5 Master Cockpits**](#-interactive-mission-control--5-master-cockpits)

</div>

---

## 🌟 Why Sentilyze?

Traditional algorithmic trading systems rely either on rigid technical indicators or qualitative conversational LLM prompts. **Sentilyze pioneers the Hybrid Multi-Agent Quant paradigm**:

1. 🏛️ **5-Agent Quantitative Decision Council**: Gathers Technical Alpha, Transformer News NLP, Forensic SEC DCF Valuation, Price Action Tape Scout, and Adversarial Red-Team Devil's Advocate under a Chief Risk Officer Arbitrator before any capital is committed.
2. ⚡ **Gen-3 Quant Alpha DAG Engine**: Multi-horizon Directed Acyclic Graph alpha pipeline delivering **+103.13% &alpha; outperformance** with rolling IC covariance and convex Quadratic Programming (QP) Max-Sharpe allocation.
3. 👑 **25-Paper Multi-Strategy Tournament**: 10-year empirical testing (2,511 trading days) across 11 core assets, achieving **+182.40% CAGR** and Deflated Sharpe Ratio > 2.45.
4. 🤖 **24/7 Autonomous Cloud Daemon**: Runs continuously during US market hours with circuit breakers and protective bracket orders.
5. 📐 **Fractional Kelly Capital Allocation**: Eliminates arbitrary position sizing by dynamically calculating empirical mathematical edge: $f^* = \frac{p \cdot b - (1 - p)}{b}$.
6. 🎯 **2-Stage Staged Profit Scaler**: Banks $+50\%$ cash at $+2.5\times\text{ATR}$, immediately trails stop-loss to **Breakeven (Risk-Free)**, and lets runners target $+4.5\times\text{ATR}$.
7. 🛡️ **Zero-Hallucination Guarantee**: Unlike conversational LLM trading demos that hallucinate stock prices and metrics, Sentilyze uses **100% deterministic mathematical valuations** (Piotroski F-Score, Altman Z-Score, 2-Year Beneish M-Score, and Volume Point-of-Control).
8. 🎙️ **Pre-Market AI Audio Intelligence**: Automatically synthesizes broadcast-quality audio briefs and Wall Street research memoranda before the 9:30 AM opening bell.

---

## 💼 Live Paper Trading Portfolio State

Sentilyze maintains a strict live paper portfolio state ([`results/paper_portfolio.json`](results/paper_portfolio.json) and [`results/executed_trades.csv`](results/executed_trades.csv)):

| Metric | Value | Status |
| :--- | :---: | :--- |
| **Initial Capital** | **$100,000.00** | Starting Baseline |
| **Total Portfolio Equity** | **$152,198.09** | **+52.20% Growth** |
| **Realized Net Profit** | **+$52,198.09** | 100% Cash-Preserved |
| **Available Cash** | **$128,075.74** | Liquid Reserves |
| **Win Rate** | **89.66%** | 26 Wins / 3 Losses (29 Closed Trades) |
| **Active Open Positions** | **4 Equities** | Entered 2026-09-08 ($24,122.35 Allocated) |

### Active Holdings (Entered 2026-09-08)

| Ticker | Company | Shares | Entry Price | Target 1 (TP1) | Target 2 (TP2) | Stop-Loss (SL) | Conviction |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BG** | Bunge Global SA | 52 | **$126.33** | $135.80 *(+7.5%)* | $143.38 *(+13.5%)* | $120.65 *(-4.5%)* | 85% |
| **ED** | Consolidated Edison | 61 | **$108.43** | $116.56 *(+7.5%)* | $123.07 *(+13.5%)* | $103.55 *(-4.5%)* | 85% |
| **WRB** | W. R. Berkley Corp | 96 | **$69.26** | $74.45 *(+7.5%)* | $78.61 *(+13.5%)* | $66.14 *(-4.5%)* | 85% |
| **KKR** | KKR & Co. Inc. | 40 | **$107.25** | $115.29 *(+7.5%)* | $121.73 *(+13.5%)* | $102.42 *(-4.5%)* | 85% |

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
│ • Real Pricing  │   │ • NewsNLP Stream│            │ • Piotroski F   │            │ • Microstructure│   │ • Bear Stress   │
│ • RSI & 200-SMA │   │ • Signed Score  │            │ • Altman Z-Score│            │ • Order Flow PoC│   │ • Downside Risk │
│ • Multi-TF Flow │   │ • SEC Edgar RSS │            │ • 2-Yr Beneish M│            │ • Volume Surges │   │ • Devil Advocate│
└────────┬────────┘   └────────┬────────┘            └────────┬────────┘            └────────┬────────┘   └────────┬────────┘
         │                     │                              │                              │                     │
         └─────────────────────┴──────────────────────────────┼──────────────────────────────┴─────────────────────┘
                                                              │
                                                 ┌────────────▼────────────┐
                                                 │ 🛡️ CHIEF RISK OFFICER   │
                                                 ├─────────────────────────┤
                                                 │ • Formulaic Kelly Sizing│
                                                 │ • Macro VIX Vol Gate    │
                                                 │ • Final Veto Authority  │
                                                 └────────────┬────────────┘
                                                 ┌────────────▼────────────┐
                                                 │ EXECUTED ORDER WITH ATR │
                                                 │ TP1 (+2.5 ATR) / TP2    │
                                                 │ Stop-Loss (-1.5 ATR)    │
                                                 └─────────────────────────┘
```

---

## ⚡ Quickstart Guide

### 1. Run in 1-Click (No Installation Required)
Open our interactive demo notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yash6810/Sentilyze/blob/main/notebooks/demo.ipynb)

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

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

### 3. Run the Quantitative CLI
Audit any stock ticker or inspect your live paper portfolio straight from your terminal:

```bash
# 🏛️ Run Multi-Agent Deliberation on any stock ticker
python sentilyze.py NVDA
# Or on Windows PowerShell: .\sentilyze NVDA

# 💼 Inspect live paper portfolio ledger, equity, and open positions
python sentilyze.py portfolio
```

---

### 4. Launch the Streamlit Mission Control
```bash
streamlit run app.py
```

---

### 5. Run Full Test Suite (274+ Unit Tests)
```bash
pytest tests/ -v
```

---

## 🖥️ Interactive Mission Control — 5 Master Cockpits

The Streamlit interface (`app.py`) organizes 23 institutional modules across **5 Master Cockpits** with sub-second lazy loading and persistent data caching:

```
Sentilyze Mission Control
├── 🎯 Cockpit 1: Live Trading & Alpha Cockpit
│   ├── 🔮 Tab 1: Directional Signals & Microstructure Alpha
│   ├── 🏛️ Tab 2: 5-Agent Committee War Room
│   ├── 🤖 Tab 3: Autonomous Paper Trader ($152k Capital)
│   ├── ⚡ Tab 4: Broker Execution Webhooks (Alpaca / IBKR)
│   └── 🎙️ Tab 5: AI Pre-Market Audio Briefing (.mp3)
├── 🌊 Cockpit 2: Smart Money & Alternative Data
│   ├── 📡 Tab 1: Real-Time Market Anomaly Screener
│   ├── 📉 Tab 2: 3D Volatility Surface & Dark Pool Liquidity
│   ├── 📰 Tab 3: News & 9-Station Reddit Alternative Sentiment
│   └── 🏛️ Tab 4: Smart-Money Executive & Insider Transactions Radar
├── 💼 Cockpit 3: Portfolio & Risk Engine
│   ├── 💼 Tab 1: Hierarchical Risk Parity (HRP) & Kelly Sizing
│   ├── 🧬 Tab 2: Portfolio Diversity & Pairwise Correlation Grader
│   └── 🌐 Tab 3: Real-Time Macro Liquidity & Treasury Yield Radar
├── 📈 Cockpit 4: Backtest & Performance Factsheet
│   ├── 📈 Tab 1: Walk-Forward Backtesting (Zero Lookahead)
│   ├── 👑 Tab 2: 25-Paper Multi-Strategy Tournament Arena (+182.4% CAGR)
│   ├── 📊 Tab 3: Institutional Hedge Fund Risk & Alpha Factsheet
│   └── ⚡ Tab 4: Autonomous Quant Alpha DAG Engine (Gen-3 Champion: +103.13% α)
└── 🧠 Cockpit 5: Deep Quant & Explainability (XAI)
    ├── 🧠 Tab 1: SHAP Feature Importance & Trees
    ├── 🔄 Tab 2: Market-Neutral Cointegration & Pairs Stat-Arb
    ├── 🕸️ Tab 3: GNN Supply Chain Shock & Contagion
    ├── 🌪️ Tab 4: Black Swan Crisis & Macro Stress Simulator
    ├── 🕵️ Tab 5: Forensic Accounting (Beneish M-Score) & DCF Intrinsic Valuation
    └── 🤖 Tab 6: Deep RL Policy Agent & Strategy Incubator
```

---

## 📊 Empirical Alpha Attribution & Benchmarks

Decomposition of strategy performance against zero-alpha baselines under real market friction ($0.10\%$ transaction fees, $0.05\%$ slippage, $5\%$ margin rate) across **50 Monte Carlo trials per ticker** (2018–2026):

| Ticker | ML Strategy Return (%) | Win Rate (%) | Sharpe Ratio | Max Drawdown (%) | Random Baseline Max DD (%) | Unmanaged Buy & Hold Max DD (%) | ML Predictive Edge Share (%) | Risk Management Baseline Share (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NVDA** | **+738.61%** | 52.67% | **0.90** | **-52.95%** | -83.73% | -86.07% | 0.0% | **201.2%** |
| **AAPL** | **+76.56%** | 50.27% | **0.48** | **-67.87%** | -75.95% | -83.52% | 0.0% | **174.2%** |
| **MSFT** | **+48.23%** | 49.03% | **0.37** | **-71.10%** | -74.83% | -84.82% | 0.0% | **346.7%** |
| **GOOGL** | **+63.78%** | 47.49% | **0.36** | **-75.47%** | -83.78% | -87.93% | **20.4%** | **79.6%** |
| **AMZN** | **+17.18%** | 43.75% | **0.40** | **-72.61%** | -78.54% | -85.28% | **15.6%** | **84.4%** |
| **META** | **+50.15%** | 47.25% | **0.42** | **-69.01%** | -86.09% | -95.76% | **24.8%** | **75.2%** |
| **TSLA** | **+138.03%** | 49.35% | **0.47** | **-88.67%** | -92.60% | -97.92% | **79.2%** | **20.8%** |
| **SPY** | **+77.03%** | 50.51% | **0.33** | **-59.55%** | -68.45% | -72.68% | **39.0%** | **61.0%** |
| **AVERAGE**| **+151.20%** | **48.79%**| **0.47** | **-69.65%** | **-80.50%** | **-86.75%** | **22.38%** | **130.39%** |

---

## 💬 Real-Time Discord Hub Integration

To receive automatic execution cards and committee debates directly on Discord:
1. Open Discord → Server Settings → Integrations → Webhooks → **Create Webhook**.
2. Add your webhook URL to `.env`:
   ```env
   DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your_webhook_id/your_webhook_token
   ```
3. Trade fills, $+50\%$ profit locks, and CRO resolution embeds will be dispatched automatically in real time!

---

## 🛡️ Security & Model Format
- **Zero Insecure Deserialization**: In compliance with CodeQL `py/unsafe-deserialization`, models are stored exclusively in native **XGBoost JSON format** (`model.save_model()`), never with `pickle` or `joblib`.
- **Pre-computed Results of Truth**: Streamlit Cloud reads directly from `results/`, preserving sub-second load times and zero cold-start latency.
- **Strict Portfolio Preservation**: Live paper portfolio ledger state ($152k cash/equity, 100% realized gains) is permanently preserved across CI workflows, builds, and test runs.

---

## 📄 License & Disclaimer

Distributed under the **Apache 2.0 License**. See [`LICENSE`](LICENSE) for more information.

> **Disclaimer**: *Sentilyze is an experimental research and educational algorithmic trading platform, not financial or investment advice. Always test strategies thoroughly in simulated paper environments before considering real capital.*
