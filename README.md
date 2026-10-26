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
[![Test Suite](https://img.shields.io/badge/Tests-Verified%20Passing-10B981?style=for-the-badge&logo=github-actions&logoColor=white)](tests/)
[![Security: CodeQL](https://img.shields.io/badge/Security-Zero%20Pickle%20%7C%20Safe%20JSON-blueviolet?style=for-the-badge&logo=security&logoColor=white)](models/)
[![Open In Colab](https://img.shields.io/badge/Google%20Colab-FinMamba--JEPA%20Powerhouse-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/yash6810/Sentilyze/blob/main/notebooks/sentilyze_finmamba_jepa_powerhouse.ipynb)
[![Streamlit Interface](https://img.shields.io/badge/Mission%20Control-5%20Master%20Cockpits-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://sentilyze.streamlit.app/)
[![Alpaca Paper Equity](https://img.shields.io/badge/Alpaca%20Paper%20Equity-$102,443.49%20(+2.44%25)-10B981?style=for-the-badge&logo=alpaca)](results/alpaca_vs_spy.csv)
[![Discord Alerts](https://img.shields.io/badge/Live%20Alerts-Discord%20Hub-5865F2?style=for-the-badge&logo=discord&logoColor=white)](src/alerts.py)

<p align="center">
  <b>An Experimental Quantitative Multi-Agent Trading Engine combining Transformer Sentiment NLP (FinBERT), Walk-Forward Machine Learning (Native XGBoost Across Equities), Multi-Agent Deliberation, and Institutional Brokerage Connectivity.</b>
</p>

[**🚀 Live App**](https://sentilyze.streamlit.app/) • [**🧠 FinMamba-JEPA Colab**](https://colab.research.google.com/github/yash6810/Sentilyze/blob/main/notebooks/sentilyze_finmamba_jepa_powerhouse.ipynb) • [**⚡ Quickstart**](#-quickstart-guide) • [**🏛️ Multi-Agent Council**](#-grounded-5-agent-deliberation-council) • [**📊 Current Status (Honest & Audited)**](#-current-status-honest--audited) • [**📂 5 Master Cockpits**](#-interactive-mission-control--5-master-cockpits)

</div>

---

## 📊 Current Status (Honest & Audited)

Following a comprehensive quantitative audit, all performance claims in this repository are grounded strictly in empirical files on disk. Previous ungrounded simulation ledgers have been archived to [`archive/invalid_simulation_2026/`](archive/invalid_simulation_2026/).

### 1. Alpaca Paper Trading Performance (Live Source of Truth)

The live paper trading account (`PA3NFG0UWP3R`) on Alpaca Markets serves as the sole external execution record. Below is the verified performance comparison against the SPY benchmark rebased to the identical starting balance over the exact same period:

| Metric | Alpaca Paper Trading (Sentilyze) | SPY Buy & Hold Benchmark | Delta / Spread |
| :--- | :---: | :---: | :---: |
| **Starting Balance** | **$100,000.00** | **$100,000.00** | $0.00 |
| **Current Liquid Equity** | **$102,443.49** | **$110,231.64** | -$7,788.15 |
| **Total Net Return** | **+2.44%** | **+10.23%** | -7.79% (Lagged Benchmark) |
| **Maximum Drawdown** | **-0.25%** | **-8.52%** | **+8.27% (Lower Peak Loss)** |
| **Annualized Sharpe Ratio** | **1.64** | **1.13** | +0.51 |
| **Total Realized Orders** | **62 orders** | Buy & Hold (1 entry) | — |
| **Active Open Positions** | 6 AMZN, 35 QCOM | 100% SPY | Conservative Risk Profile |

> [!NOTE]
> **Performance Disclosure**: The strategy achieved positive net return (+2.44%) with tight drawdown control (-0.25%), but **lagged the SPY benchmark (+10.23%)** due to conservative cash reservation policies (holding 85%+ cash) and selective entry gating.

---

### 1.1 Exit Behavior & Brokerage Execution Architecture

> [!IMPORTANT]
> **Alpaca Bracket-Only Execution**: The live Alpaca paper trading account exits **only through native entry-time OCO bracket orders** (`take_profit.limit_price` and `stop_loss.stop_price`) submitted at order initiation.
> Intraday trailing stops, committee veto exits, and Soros reflexivity exits evaluated by `AutonomousTradingEngine` apply **strictly to the local simulation ledger** (`results/paper_portfolio.json`) and do not emit secondary cancellation or closing orders to Alpaca.

---

### 2. Audited Walk-Forward Backtest Findings (2018–2026)

Walk-forward backtests across equities under real-world market friction (0.10% broker fee, 0.05% slippage, execution at $t+1$ Open) revealed key structural findings:

1. **Missing Historical News**: News headlines were unavailable for ~89% of trading days across 2018–2026; sentiment inputs were zero-filled (score = 0.0).
2. **FinBERT Sentiment Ablation**: Committee ablation experiments (`results/committee_ablation_study.json`) comparing multi-agent performance with vs without FinBERT news sentiment across all 5 evaluated tickers revealed an average **-11.81% return drag** and **-0.18 Sharpe drag** due to headline sparsity and sentiment false-positives:

   | Ticker | Evaluation Days | Return (With FinBERT) | Sharpe (With) | Return (Without FinBERT) | Sharpe (Without) | Delta Return | Delta Sharpe |
   | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
   | **NVDA** | 500 | -5.39% | 0.08 | +2.59% | 0.21 | **-7.98%** | -0.13 |
   | **AAPL** | 150 | +1.94% | 0.26 | +1.94% | 0.26 | **+0.00%** | +0.00 |
   | **MSFT** | 500 | -10.66% | -0.12 | +2.75% | 0.18 | **-13.41%** | -0.30 |
   | **GOOGL** | 500 | +22.51% | 0.53 | +67.62% | 1.14 | **-45.11%** | -0.61 |
   | **AMZN** | 500 | +2.89% | 0.19 | -4.54% | 0.07 | **+7.43%** | +0.12 |
   | **Average** | — | **+2.26%** | **0.19** | **+14.07%** | **0.37** | **-11.81%** | **-0.18** |

   *Note: Evaluation windows differ by ticker in `results/committee_ablation_study.json` (AAPL was evaluated over 150 days; NVDA, MSFT, GOOGL, AMZN over 500 days). Live trading is strictly gated to skip trades when verified news is unavailable.*
3. **Benchmark Comparison (3-Ticker Out-of-Sample Walk-Forward)**:
   - **NVDA**: Strategy return **+357.47%** (Sharpe 0.92, Max DD -44.16%) vs Buy & Hold **+1,019.21%** (Sharpe 1.23, Max DD -41.57%) — *Strategy did not beat Buy & Hold*.
   - **SPY**: Strategy return **+43.76%** (Sharpe 0.68, Max DD -26.34%) vs Buy & Hold **+85.51%** (Sharpe 0.98, Max DD -19.00%) — *Strategy did not beat Buy & Hold*.
   - **WMT**: Strategy return **+104.67%** (Sharpe 0.75, Max DD -29.44%) vs Buy & Hold **+158.01%** (Sharpe 0.89, Max DD -26.01%) — *Strategy did not beat Buy & Hold*.
4. **SPY Maximum Drawdown Correction**: SPY buy-and-hold historical maximum drawdown was **-33.72%** (during the March 2020 COVID crash). The previous figure of -72.68% was the drawdown of an experimental `always_long_strategy` subjected to 5% annual margin borrowing interest and trailing stop whipsaws, not passive index buy-and-hold.

---

### 3. Quantitative Architecture & Module Classification

To maintain scientific transparency, repository modules are classified into three operational categories:

- **(a) Live in Decision Loop**:
  - [`src/autonomous_trader.py`](src/autonomous_trader.py) — Autonomous cycle orchestrator with verified news gating.
  - [`src/agent_committee.py`](src/agent_committee.py) — 5-agent deliberation council (Technical Momentum, Sentiment Catalyst, Fundamental Forensic, CRO Arbitrator).
  - [`src/paper_broker.py`](src/paper_broker.py) — Local execution simulation and order book tracker.
  - [`src/alpaca_broker.py`](src/alpaca_broker.py) — Institutional Alpaca paper/live execution bridge.
  - [`src/realtime_tracker.py`](src/realtime_tracker.py) — Live price polling, stop-loss trailing, and Discord alerting.

- **(b) Dashboard Only**:
  - [`app.py`](app.py) & [`src/ui/*`](src/ui/) — Streamlit analytical mission control interface.
  - [`src/sentiment_analysis.py`](src/sentiment_analysis.py) — On-demand headline sentiment scoring.
  - [`src/stock_image_provider.py`](src/stock_image_provider.py) — Brand asset and company logo retrieval.
  - [`src/pdf_generator.py`](src/pdf_generator.py) & [`src/tearsheet_generator.py`](src/tearsheet_generator.py) — Factsheet generation.
  - [`src/alerts.py`](src/alerts.py) — Webhook dispatchers for Discord/Slack notifications.

- **(c) Research & Experimental (Unintegrated in Live Decision Loop)**:
  - [`src/finmamba_jepa.py`](src/finmamba_jepa.py) — Standalone prototype exploring continuous-time S6 state space and JEPA representations (runnable in Google Colab; not deployed in live execution).
  - [`src/hawkes_cascade_engine.py`](src/hawkes_cascade_engine.py) — Research point-process order-flow clustering model.
  - [`src/multimodal_chart_analyst.py`](src/multimodal_chart_analyst.py) — Computer vision liquidity sweep chart analyzer.
  - [`src/kalman_statarb_engine.py`](src/kalman_statarb_engine.py) — Statistical arbitrage pair cointegration scanner.
  - [`src/adaptive_execution_router.py`](src/adaptive_execution_router.py) — Theoretical Almgren-Chriss execution router.
  - [`src/forensic_accounting.py`](src/forensic_accounting.py) — Fundamental accounting metrics engine.

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

# 💼 Generate Alpaca Paper Trading Performance Report
python src/alpaca_report.py
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
│   ├── 🤖 Tab 3: Autonomous Paper Trader (Alpaca Broker Bridge)
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

## 🛡️ Security, Model Serialization & Code Quality

- **Zero Insecure Deserialization**: In strict compliance with CodeQL `py/unsafe-deserialization`, models are stored exclusively in native **XGBoost JSON format** (`model.save_model()`) or PyTorch `weights_only=True` (`torch.load(..., weights_only=True)`). Zero `pickle` or `joblib` is permitted anywhere in the repository.
- **Pre-computed Results of Truth**: Streamlit Cloud reads directly from pre-computed artifacts in `results/`, ensuring sub-second page loads and zero cold-start GPU lag.
- **Code Standards**: The codebase is strictly formatted with `black==25.1.0` and validated against `flake8` (`E9,F63,F7,F82`) with 0 errors.

---

## 📄 License & Disclaimer

Distributed under the **Apache 2.0 License**. See [`LICENSE`](LICENSE) for more information.

> **Disclaimer**: *Sentilyze is an experimental quantitative research and algorithmic trading platform designed for education and systematic backtesting. It does not constitute investment advice. Live trading involves substantial risk of capital loss.*
