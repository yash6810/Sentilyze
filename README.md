# Sentilyze

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python 3.10">
  <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
  <img src="https://img.shields.io/badge/CI/CD-GitHub%20Actions-green.svg" alt="CI/CD: GitHub Actions">
</p>

---

## 🔭 Project Vision

To engineer a premier, data-driven algorithmic trading tool that provides a significant competitive edge, capable of identifying and capitalizing on market opportunities with high accuracy.

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

## 🚀 Getting Started with Docker

This project is fully containerized with Docker, which is the simplest way to get started. All you need is Docker installed on your system.

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd sentilyze
```

### 2. Set Up API Keys

Create a `.streamlit/secrets.toml` file and add your NewsAPI.org key:

```toml
NEWS_API_KEY = "your_api_key_here"
```

### 3. Build and Run the Application

Use Docker Compose to build the image and launch the Streamlit app.

```bash
docker-compose up --build
```

You can now access the Streamlit app in your browser at `http://localhost:8501`.

### 4. Train a New Model

To train a model for a new stock, run the training script inside a temporary container using Docker Compose.

```bash
# Train for Apple
docker-compose run --rm app python train.py --ticker AAPL

# Train for Microsoft
docker-compose run --rm app python train.py --ticker MSFT
```

The trained models will be saved to your local `models` directory.

### Development

This setup is ready for development. Because your local `src`, `models`, and `data` directories are synced with the container, any changes you make to the code will be reflected live in the running application.

---

## ✨ Code Quality

This project enforces high code quality standards through:

* **CI/CD:** An automated GitHub Actions workflow runs all tests on every push and pull request.
* **Pre-Commit Hooks:** Code is automatically formatted with `black` and linted with `flake8` before each commit.
* **Structured Logging:** All modules use a centralized logger for traceable and informative output.
