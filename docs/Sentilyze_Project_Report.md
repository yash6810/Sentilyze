---
# Sentilyze - Project Report & Recommendations

Date: 2025-10-11
Author: yash6810 (generated helper: GitHub Copilot)

## Executive Summary

Sentilyze is a sentiment-driven stock momentum predictor that combines financial news sentiment and historical price data to predict next-day stock momentum. This document collects everything needed to run, improve, evaluate, and present the project. It includes technical requirements, data sources, training guidance, recommended experiments, deployment & monitoring steps, governance and ethical considerations, and a prioritized roadmap of improvements.

## Contents

1. What the project does
2. Files & structure
3. Dependencies & environment
4. Data sources & datasets
5. Model training: recipes and tips
6. Feature engineering & labeling
7. Evaluation & backtesting
8. Experiment tracking & reproducibility
9. Deployment & monitoring
10. Security, privacy & ethics
11. Product recommendations & roadmap
12. How to generate a .docx of this report
13. Appendix: commands and example config


## 1. What the project does

Sentilyze fetches financial news (NewsAPI.org) and historical price data (yfinance), scores headlines using FinBERT, computes technical indicators, builds a combined feature set, and trains a classifier (XGBoost) to predict next-day momentum (up vs down). The project also supports a Streamlit UI and backtesting utilities.


## 2. Files & structure (where to look)

- app.py — Streamlit front-end
- train.py — end-to-end training pipeline for a ticker
- Dockerfile / docker-compose.yml — containerization and deployment
- src/data_ingestion.py — news & price fetchers
- src/sentiment_analysis.py — FinBERT wrapper
- src/feature_engineering.py — technical indicators & merges
- src/modeling.py — training, saving, loading models
- src/backtesting.py — portfolio simulation & metrics
- models/ — saved model artifacts
- data/ — cached raw and processed data
- tests/ — unit tests


## 3. Dependencies & environment

Minimum:
- Python 3.10
- pip packages: pandas, numpy, scikit-learn, xgboost, transformers, torch, tokenizers, python-docx, streamlit, yfinance, ta (or ta-lib), requests
- Dev: pytest or unittest, pre-commit hooks

Recommended install (venv):

pip install -r requirements.txt
pip install python-docx

Docker: build with Dockerfile provided.


## 4. Data sources & datasets

Primary:
- NewsAPI.org — headlines and metadata (free tier limited)
- yfinance — historical OHLCV price data

Training / labeling datasets:
- Financial PhraseBank — labeled sentences to validate FinBERT performance
- Kaggle stock news datasets (search for "stock news")
- Historical price CSVs (Kaggle or Quandl) for bulk training

Alternative / enrichment:
- Twitter / Stocktwits (public APIs)
- SEC EDGAR filings (10-K, 8-K) for company event signals
- Reddit (r/wallstreetbets) for retail-sentiment signals

Notes on licensing: always check Terms of Service before scraping or commercial usage.


## 5. Model training: recipes and tips

A standard training flow (reproducible):
1. Fetch news for the company for a historical window (e.g., last 3 years).
2. Fetch daily price history for same window.
3. Run FinBERT sentiment scoring on headlines. Cache results.
4. Aggregate sentiment to daily-level features (mean, median, counts, weighted by relevance/time).
5. Compute technical indicators: MA7, MA21, RSI(14), MACD, Bollinger Bands, stochastic oscillator, volume change, returns.
6. Align features with target: next-day momentum (Close_t+1 > Close_t ? 1 : 0) or thresholded returns.
7. Split data by time (train/validation/test) to avoid lookahead (e.g., train up to 2022, validate 2023, test 2024).
8. Train baseline models: LogisticRegression, XGBoost.
9. Evaluate with metrics: accuracy, precision, recall, F1, ROC-AUC, and return-based Sharpe ratio in backtest.
10. Use cross-validation on time series (e.g., expanding window) for hyperparameter tuning.

Hyperparameters to try:
- XGBoost: n_estimators 100-500, learning_rate 0.01-0.2, max_depth 3-8, subsample 0.6-1.0



## 6. Feature engineering & labeling

Feature ideas:
- Sentiment: daily mean, std, counts, positive_ratio, negative_ratio, last_headline_score
- Price: daily returns, rolling mean returns, volatility (rolling std), momentum features
- Event flags: earnings day, upgrade/downgrade news, SEC filings
- Interaction features: sentiment * volatility, sentiment * volume_change

Labeling:
- Binary next-day up/down using close-to-close returns
- Multi-class: up/flat/down with a threshold
- Regression: next-day return magnitude

Avoid leakage: ensure all news/headlines used to compute features are published before the close that defines the target.


## 7. Evaluation & backtesting

Offline metrics:
- Confusion matrix, precision/recall per class, ROC-AUC

Financial metrics (in backtest):
- Cumulative returns of strategy vs buy-and-hold
- Annualized return, annualized volatility, Sharpe ratio, Max drawdown, Calmar ratio
- Monthly returns heatmap

Backtest rules:
- Use signal shifting so predictions for day t are used to trade on day t+1 open.
- Transaction costs and slippage assumptions (e.g., 0.05% per trade) to make backtest realistic.


## 8. Experiment tracking & reproducibility

- Use MLflow or Weights & Biases to track experiments, metrics, and artifacts (models, tokenizer, metrics).
- Save environment information (pip freeze) with each run.
- Save random seeds and document training split timestamps.


## 9. Deployment & monitoring

Deployment options:
- Streamlit Cloud (for public demos)
- Docker + cloud VM (AWS/GCP/Azure) behind Nginx for private demos

Monitoring:
- Add health checks and logs
- Track model performance drift (rolling test accuracy) and data drift (feature distributions)
- Schedule automated retraining when drift exceeds thresholds


## 10. Security, privacy & ethics

- Keep API keys and secrets in Streamlit secrets or environment variables.
- Disclose risk and “not financial advice” prominently in UI.
- Audit data sources for bias and ensure labeling is appropriate.
- Rate-limit API calls and respect terms of use.


## 11. Product recommendations & roadmap (prioritized)

Immediate high impact (Weeks 1-2):
- Add SHAP explanations to show why a prediction was made.
- Add watchlist and simple email/Telegram alerts.
- Improve model evaluation and show test accuracy/confidence in UI.

Medium term (Weeks 3-8):
- Integrate social media signals (Twitter/Stocktwits)
- Multi-stock and sector dashboards with heatmaps
- Automated retraining and experiment tracking (MLflow)

Long term (Months):
- Portfolio optimization + risk management features
- Paid premium signals, API access, or integrations with brokerages
- Advanced models: transformer-based time series or multi-modal architectures


## 12. How to generate a .docx of this report

This repository includes a small Python utility to convert this markdown into a .docx. Steps:
1. Ensure python-docx is installed: pip install python-docx
2. Run the script: python scripts/generate_sentilyze_docx.py
3. Output: docs/Sentilyze_Project_Report.docx


## 13. Appendix: commands and example config

Docker build & run:

```bash
# build
docker-compose up --build

# run training inside container
docker-compose run --rm app python train.py --ticker NVDA
```

Streamlit secrets example (.streamlit/secrets.toml):

```toml
NEWS_API_KEY = "your_api_key_here"
```

Example MLflow usage:

```bash
mlflow ui --port 5000
```

---

End of report.

---