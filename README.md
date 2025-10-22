# Sentilyze

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
![CI/CD: GitHub Actions](https://img.shields.io/badge/CI/CD-GitHub%20Actions-green.svg)

---

## 🔭 Project Vision

To engineer an experimental, data-driven algorithmic trading tool that provides a significant competitive edge, capable of identifying and capitalizing on market opportunities with high accuracy. This project is intended as a prototype and proof-of-concept, not a production-ready trading system.

---

## ⚙️ How It Works

Sentilyze predicts next-day stock momentum by combining financial news sentiment with technical analysis. The pipeline is as follows:

1.  **Data Ingestion:** Fetches historical price data from `yfinance` and news headlines from `NewsAPI.org`.
2.  **Sentiment Analysis:** Uses a pre-trained FinBERT model to analyze the sentiment of each news headline.
3.  **Feature Engineering:** Calculates a rich set of features based on the ingested data, including sentiment scores and technical indicators (e.g., RSI, MACD).
4.  **Prediction:** A `RandomForestClassifier` model, trained on this combined data, predicts the momentum for the next trading day.

---

## 🚀 Getting Started: The 10-Minute Setup

This guide provides a foolproof, step-by-step process to get the Sentilyze application up and running on your local machine in under 10 minutes.

### Prerequisites

Before you begin, make sure you have the following software installed on your system:
*   **Python 3.10** or higher
*   **pip** (the Python package installer)
*   **Git** (for cloning the repository)

### 1. Clone the Repository

First, clone the repository to your local machine using the following command:

```bash
git clone https://github.com/yash6810/sentilyze.git
cd sentilyze
```

### 2. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage the project's dependencies. This will prevent conflicts with other Python projects on your system.

```bash
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.venv\Scripts\activate.bat
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

Install all the necessary dependencies using the `requirements.txt` file. This file includes all the packages needed to run the application and the training scripts.

```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys

Create a `.env` file in the root of the project. This file will store your NewsAPI.org API key. You can get a free API key from the [NewsAPI.org website](https://newsapi.org/).

```
NEWS_API_KEY="your_api_key_here"
```
**Important:** The application will not work without a valid NewsAPI key.

### 5. Train a Model (Optional for Initial Exploration)

While a pre-trained model for NVDA is included for immediate use, you can train your own models for different tickers.

To train a model for a specific stock, run the `train.py` script. For example, to train a model for NVDA:

```bash
python train.py --ticker NVDA
```
This script will:
*   Fetch the latest news and price data for the specified ticker.
*   Perform sentiment analysis and feature engineering.
*   Train a `RandomForestClassifier` model with hyperparameter tuning.
*   Save the trained model to the `models` directory.
*   **Log all training results, including metrics, backtest data, feature importances, SHAP values, and the classification report, to MLflow.**
*   Save all the training results (metrics, backtest data, feature importances, and SHAP values) to the `results` directory.

### 6. Using the Pre-trained Model (NVDA)

To get started immediately without training, a pre-trained model for **NVDA** is provided in the repository. This allows you to run the Streamlit app directly and explore its features.

### 7. Run the Streamlit App

Now you are ready to run the Streamlit application.

```bash
streamlit run app.py
```

This will open the application in your web browser. You can then enter a stock ticker (e.g., "NVDA" to use the pre-trained model), get predictions, run backtests, and view the model performance dashboard.

### 8. Track Experiments with MLflow

This project is integrated with MLflow for experiment tracking. After you have run a few training sessions, you can view and compare the results using the MLflow UI.

```bash
mlflow ui
```
Navigate to `http://localhost:5000` in your browser to see the MLflow dashboard.

---

## Troubleshooting

*   **`ModuleNotFoundError`:** If you get a `ModuleNotFoundError`, it means that you have not installed all the dependencies. Please run `pip install -r requirements.txt` again.
*   **API Key Errors:** If you are having issues with fetching news data, make sure that your `NEWS_API_KEY` in the `.env` file is correct and that you have not exceeded your API rate limit.
*   **`FileNotFoundError` or Missing Data in "Model Performance" Tab:** If you encounter this error or see missing data (e.g., SHAP plots, Classification Report) in the "Model Performance" tab, it likely means that the MLflow artifacts for the selected ticker are either missing or corrupted. Ensure you have trained a model for that ticker using `python train.py --ticker [TICKER]` and that MLflow successfully logged all artifacts.

---

## Project Philosophy

*   **Modularity:** The project is organized into a `src` directory with separate modules for each major component (data ingestion, feature engineering, modeling, etc.). This makes the code easier to understand, maintain, and extend.
*   **Reproducibility:** We use a `requirements.txt` file with pinned dependencies and MLflow for experiment tracking to ensure that our results are reproducible.
*   **Code Quality:** We use `black` for code formatting and `flake8` for linting to maintain a high level of code quality. These checks are automatically enforced by our CI/CD pipeline.
*   **Explainability:** We use `SHAP` to provide insights into our model's predictions, so we can understand *why* it is making certain decisions.

---

## 🐳 Usage with Docker

If you have Docker installed, you can use Docker Compose to build and run the application in a container. This is the simplest way to get started.

1.  **Set Up API Keys:** Create a `.streamlit/secrets.toml` file and add your NewsAPI.org key:

    ```toml
    NEWS_API_KEY = "your_api_key_here"
    ```

2.  **Build and Run:**

    ```bash
    docker-compose up --build
    ```

    The app will be available at `http://localhost:8501`.