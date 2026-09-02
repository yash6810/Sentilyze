---
title: Sentilyze Multi-Agent Quant Engine
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: "1.37.1"
app_file: app.py
pinned: false
license: apache-2.0
---

# Sentilyze — Autonomous Multi-Agent Quantitative Trading & NLP Intelligence Platform

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Code Style: Black](https://img.shields.io/badge/Code%20Style-Black-000000?style=for-the-badge&logo=python&logoColor=white)](https://github.com/psf/black)
[![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-240%2B%20Tests%20Passing-10B981?style=for-the-badge&logo=github-actions&logoColor=white)](.github/workflows/)
[![Open In Colab](https://img.shields.io/badge/Google%20Colab-1--Click%20Demo-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/yash6810/Sentilyze/blob/main/notebooks/demo.ipynb)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces%20Live%20App-yellow?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/clash1462/Sentilyze)
[![Streamlit Interface](https://img.shields.io/badge/Mission%20Control-23%20Workspaces%20Live-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://sentilyze.streamlit.app/)
[![Discord Alerts](https://img.shields.io/badge/Live%20Alerts-Discord%20Hub-5865F2?style=for-the-badge&logo=discord&logoColor=white)](src/alerts.py)
[![ML Framework](https://img.shields.io/badge/ML-XGBoost%20%2B%20FinBERT%20%2B%20PyTorch%20RL-F59E0B?style=for-the-badge&logo=scikit-learn&logoColor=white)](src/modeling.py)

<p align="center">
  <b>A 24/7 Autonomous Hybrid Quantitative Multi-Agent Trading Engine combining Deep Transformer NLP (FinBERT), Walk-Forward Machine Learning (XGBoost), 4-Agent Quorum Consensus, Deep Reinforcement Learning (PyTorch Continuous Actor-Critic), Genetic Strategy Incubation, and Fractional Kelly Sizing across US Equities.</b>
</p>

[**🚀 Live App**](https://sentilyze.streamlit.app/) • [**⚡ 1-Click Colab**](https://colab.research.google.com/github/yash6810/Sentilyze/blob/main/notebooks/demo.ipynb) • [**⚡ Quickstart**](#-quickstart-guide) • [**🏛️ Multi-Agent Committee**](#-grounded-4-agent-deliberation-council) • [**🎯 Staged ATR Scaling**](#-asymmetric-risk-management--staged-profit-scaling) • [**📂 23 Workspaces Matrix**](#-interactive-streamlit-app--23-mission-control-workspaces)

</div>

---

## 🌟 Why Sentilyze?

Traditional algorithmic trading systems rely either on rigid technical indicators or qualitative conversational LLM prompts. **Sentilyze pioneers the Hybrid Multi-Agent Quant paradigm**:

1. 🏛️ **4-Agent Quantitative Committee**: Gathers Technical Alpha, Transformer News NLP, Forensic SEC DCF Valuation, and Chief Risk Officer Arbitrator into an automated round-table quorum before any capital is committed.
2. 🤖 **24/7 Autonomous Cloud Daemon**: Runs continuously every 5 minutes during US market hours on GitHub Actions — zero local PC runtime required.
3. 📐 **Fractional Kelly Capital Allocation**: Eliminates arbitrary position sizing by dynamically calculating empirical mathematical edge: $f^* = \frac{p \cdot b - (1 - p)}{b}$.
4. 🎯 **2-Stage Staged Profit Scaler**: Banks $+50\%$ cash at $+2.5\times\text{ATR}$, immediately trails stop-loss to **Breakeven (Risk-Free)**, and lets runners target $+4.5\times\text{ATR}$.
5. 🛡️ **Zero-Hallucination Guarantee**: Unlike conversational LLM trading demos that hallucinate stock prices and metrics, Sentilyze uses **100% deterministic mathematical valuations** (Piotroski F-Score, Altman Z-Score, 2-Year Beneish M-Score, and Volume Point-of-Control).
6. 🎙️ **Pre-Market AI Audio Intelligence**: Automatically synthesizes broadcast-quality audio briefs and Wall Street research memoranda before the 9:30 AM opening bell.

---

## 🏛️ Grounded 4-Agent Deliberation Council

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

### 3. Run the 4-Agent Quantitative CLI
Audit any stock ticker or inspect your live paper portfolio straight from your terminal:

```bash
# 🏛️ Run 4-Agent Deliberation on any stock ticker
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

### 5. Run Full Test Suite (240+ Unit Tests)
```bash
pytest tests/ -v
```

---

## 🖥️ Interactive Streamlit App & 23 Mission Control Workspaces

The Streamlit interface (`app.py`) provides an institutional 23-workspace suite:

| # | Workspace | Domain & Technology |
|---|---|---|
| **1** | **🎯 Live Momentum Predictions** | FinBERT NLP + XGBoost Walk-Forward momentum predictions |
| **2** | **🏛️ Multi-Agent Deliberations** | 4-agent round-table quorum votes and Chief Risk Officer vetoes |
| **3** | **🤖 24/7 Autonomous Trader** | Real-time position tracking, multi-stage ATR scaling, and fill ledger |
| **4** | **🌐 Alternative Data & Macro Intelligence** | Congressional disclosures, Google Search trends, and Macro Yields |
| **5** | **💼 Portfolio Kelly Sizing & Risk Parity** | Fractional Kelly, Hierarchical Risk Parity, and Monte Carlo |
| **6** | **📈 Backtesting & Regime Stress Lab** | Non-overlapping walk-forward splits and dynamic regime leverage |
| **7** | **🧠 Explainable AI (XAI) & SHAP** | TreeExplainer waterfall attributions and beeswarm summary plots |
| **8** | **⚡ Implied Volatility Surface & GEX** | Black-Scholes surfaces, Put/Call ratios, and Net Dealer Gamma |
| **9** | **🌅 Opening Range Breakout (ORB)** | 9:30–10:00 AM volatility expansion breakouts with ATR stops |
| **10** | **🕸️ Market Graph Neural Network (GNN)** | Inter-asset contagion, sector graphs, and spectral centrality |
| **11** | **🌪️ Crisis Stress Testing Lab** | 2008 GFC, 2020 COVID, and 2022 Fed rate hike shock simulations |
| **12** | **🕵️ Forensic Accounting & Beneish M-Score** | Beneish M-Score (earnings manipulation) & Altman Z-Score |
| **13** | **🏛️ DCF Intrinsic Valuation** | 3-scenario Monte Carlo Discounted Cash Flow and Margin of Safety |
| **14** | **👑 25-Paper Tournament Arena** | Multi-strategy tournament arena, ADWIN drift, and Page-Hinkley test |
| **15** | **🧬 Portfolio Diversity & Correlation Grader** | $N \times N$ correlation matrix, PCA Shannon entropy $N_{\text{eff}}$, and $A+$ to $D$ score |
| **16** | **🏛️ Smart-Money Executive & Insider Radar** | SEC Form 4 cluster buy tracking and 0–100 Insider Conviction Index |
| **17** | **📊 Institutional Risk & Alpha Factsheet** | 30+ hedge fund ratios (Sortino, Calmar, Omega) and Monthly Returns grid |
| **18** | **🤖 Deep RL Autonomous Policy Agent** | PyTorch continuous Actor-Critic policy with Sortino-penalized rewards |
| **19** | **🔬 Evolutionary Strategy Incubator** | Genetic Algorithm breeding, 3-Zone In/Out-of-sample tests, Strategy Vault |
| **20** | **🔄 Market-Neutral Cointegration & Stat-Arb** | Engle-Granger ADF tests, Ornstein-Uhlenbeck half-life, $\pm 2.0\sigma$ Z-scores |
| **21** | **🎙️ AI Pre-Market Morning Audio Briefing** | Synthesized speech podcast audio (.mp3) + Wall Street research memo |
| **22** | **⚡ Automated Broker Webhooks & API Gateway** | Alpaca / IBKR bracket order payload generator with HMAC-SHA256 signatures |
| **23** | **🌐 Real-Time Macro Liquidity & Yield Radar** | 10Y-2Y Treasury spread inversion signals and Fed Net Liquidity index |

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
- **Pre-computed Results of Truth**: Streamlit Cloud reads directly from `results/`, preserving fast load times and zero cold-start latency.

---

## 📄 License & Disclaimer

Distributed under the **Apache 2.0 License**. See [`LICENSE`](LICENSE) for more information.

> **Disclaimer**: *Sentilyze is an experimental research and educational algorithmic trading platform, not financial or investment advice. Always test strategies thoroughly in simulated paper environments before considering real capital.*
