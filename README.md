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

1. **Data Ingestion:** Fetches historical price data from `yfinance` and news headlines from `NewsAPI.org`.
2. **Sentiment Analysis:** Uses a pre-trained FinBERT model to analyze the sentiment of each news headline.
3. **Feature Engineering:** Calculates a rich set of features based on the ingested data, including:
    * **Sentiment Scores:** Mean sentiment, count of positive/negative/neutral headlines.
    * **Technical Indicators:**
        * 7 and 21-day Moving Averages (MA)
        * Relative Strength Index (RSI)
        * Moving Average Convergence Divergence (MACD)
        * Bollinger Bands
        * Stochastic Oscillator
4. **Prediction:** A `RandomForestClassifier` model, trained on this combined data, predicts the momentum for the next trading day.

---

## 📂 Project Structure

```
/sentilyze
|-- .github/workflows/ci.yml    # CI/CD pipeline for automated testing
|-- .streamlit/
|   |-- secrets.toml              # For API keys
|-- data/                         # Raw and processed data (cached)
|-- models/                       # Trained ML models (e.g., NVDA_model.joblib)
|-- src/
|   |-- data_ingestion.py
|   |-- feature_engineering.py
|   |-- modeling.py
|   |-- sentiment_analysis.py
|   |-- utils.py                  # Logging configuration
|-- tests/
|-- .pre-commit-config.yaml       # Configuration for pre-commit hooks
|-- app.py                        # Main Streamlit application
|-- requirements.txt              # Main application dependencies
|-- requirements-dev.txt          # Dependencies for development (e.g., pre-commit)
|-- train.py                      # Script for training models
|-- README.md
```

---

## 🚀 Getting Started

### How to Reproduce

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yash6810/sentilyze.git
   cd sentilyze
   ```

2. **Set up a virtual environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API Keys:**
   Create a `.env` file in the root of the project and add your NewsAPI.org key:

   ```
   NEWS_API_KEY="your_api_key_here"
   ```

5. **Train a model:**

   ```bash
   python train.py --ticker NVDA
   ```

6. **Run the Streamlit app:**

   ```bash
   streamlit run app.py
   ```

### Usage with Docker

This project is fully containerized with Docker, which is the simplest way to get started. All you need is Docker installed on your system.

#### 1. Build and Run the Application

Use Docker Compose to build the image and launch the Streamlit app.

```bash
docker-compose up --build
```

You can now access the Streamlit app in your browser at `http://localhost:8501`.

#### 2. Train a New Model

To train a model for a new stock, run the training script inside a temporary container using Docker Compose.

```bash
# Train for Apple
docker-compose run --rm app python train.py --ticker AAPL

# Train for Microsoft
docker-compose run --rm app python train.py --ticker MSFT
```

The trained models will be saved to your local `models` directory.

---

## 📈 Results (Placeholder)

This section will be updated with model performance metrics, backtesting results, and feature importance analysis.

* **Model Accuracy:** [TBD]
* **Precision, Recall, F1-Score:** [TBD]
* **Backtesting Results:**
  * Strategy Return vs. Buy & Hold
  * Sharpe Ratio
  * Max Drawdown
* **Feature Importance:** [TBD]

---

## 🔮 Future Improvements

* **Ablation Studies:** Conduct experiments to evaluate the impact of sentiment features vs. technical indicators.
* **Experiment Tracking:** Integrate MLflow or DVC to track experiments, models, and datasets.
* **Advanced Models:** Explore more complex models like Gradient Boosting, LSTMs, or Transformers for time-series forecasting.
* **Explainability:** Use SHAP or LIME to provide deeper insights into model predictions.
* **More Comprehensive Testing:** Increase test coverage for all modules.

---

## ✨ Code Quality

This project enforces high code quality standards through:

* **CI/CD:** An automated GitHub Actions workflow runs all tests on every push and pull request.
* **Pre-Commit Hooks:** Code is automatically formatted with `black` and linted with `flake8` before each commit.
* **Structured Logging:** All modules use a centralized logger for traceable and informative output.
