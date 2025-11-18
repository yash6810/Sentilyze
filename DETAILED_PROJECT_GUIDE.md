# DETAILED PROJECT GUIDE: Sentilyze

This document provides a detailed, function-by-function overview of the core logic in the `src/` directory, explaining the purpose, parameters, and outputs of each component.

---

## `src/data_ingestion.py`

**Purpose**: To fetch raw data from external sources (`yfinance` and `NewsAPI.org`) with robust caching and error handling.

### `get_news(ticker: str, api_key: str) -> pd.DataFrame`

- **Purpose**: Fetches recent news articles for a given stock ticker. It caches the results in a CSV file (`data/raw/{ticker}_news.csv`) to prevent redundant API calls.
- **Parameters**:
  - `ticker (str)`: The stock ticker to fetch news for (e.g., "NVDA").
  - `api_key (str)`: Your API key for NewsAPI.org.
- **Returns**: A `pandas.DataFrame` indexed by `publishedAt` (a timezone-aware datetime object). Columns include `title`, `description`, `url`, and `source`.

### `get_price_history(ticker: str, period: str = "1y", ...) -> pd.DataFrame`

- **Purpose**: Fetches historical stock price data for a given ticker from `yfinance`. It includes a file-based cache (`data/raw/{ticker}_price_history.csv`) and a retry mechanism with exponential backoff for API call failures.
- **Parameters**:
  - `ticker (str)`: The stock ticker (e.g., "NVDA").
  - `period (str)`: The time period for the data (e.g., "1y", "5y", "max"). Defaults to `"1y"`.
- **Returns**: A `pandas.DataFrame` indexed by `Date` (a timezone-aware datetime object). Columns include `Open`, `High`, `Low`, `Close`, and `Volume`.

---

## `src/sentiment_analysis.py`

**Purpose**: To analyze the sentiment of news articles using a pre-trained FinBERT model and to manage caching of the results.

### `get_sentiment_with_caching(articles: pd.DataFrame, sentiment_analyzer: Any, ticker: str) -> pd.DataFrame`

- **Purpose**: A wrapper function that orchestrates sentiment analysis with caching. It checks for a cached result (`data/processed/{ticker}_sentiment.csv`) before running the analysis. If no cache is found, it calls `get_sentiment` and saves the result.
- **Parameters**:
  - `articles (pd.DataFrame)`: The DataFrame of news articles from `get_news`.
  - `sentiment_analyzer (Any)`: An instantiated Hugging Face sentiment analysis pipeline.
  - `ticker (str)`: The stock ticker, used for naming the cache file.
- **Returns**: A `pandas.DataFrame` containing the original article data plus new columns for `sentiment_label` and `sentiment_score`.

### `get_sentiment(articles: pd.DataFrame, sentiment_analyzer: Any) -> pd.DataFrame`

- **Purpose**: The core function that performs sentiment analysis on the text of news articles.
- **Parameters**:
  - `articles (pd.DataFrame)`: DataFrame of news articles, must contain `title` and `description` columns.
  - `sentiment_analyzer (Any)`: The sentiment analysis pipeline.
- **Returns**: A `pandas.DataFrame` with two new columns: `sentiment_label` (e.g., 'positive', 'negative') and `sentiment_score` (a float representing confidence).

---

## `src/feature_engineering.py`

**Purpose**: To transform raw price and sentiment data into a feature set suitable for the machine learning model.

### `create_technical_indicators(price_history: pd.DataFrame) -> pd.DataFrame`

- **Purpose**: Calculates a suite of technical analysis indicators from the price history.
- **Parameters**:
  - `price_history (pd.DataFrame)`: The DataFrame of historical price data.
- **Returns**: The original `pandas.DataFrame` with added columns for each indicator: `ma7`, `ma21`, `rsi`, `macd`, `bollinger_upper`, `bollinger_lower`, and `stochastic_oscillator`.

### `aggregate_sentiment_scores(news_with_sentiment: pd.DataFrame) -> pd.DataFrame`

- **Purpose**: Aggregates individual sentiment scores into a daily summary. It resamples the data by day to handle multiple news articles within the same day.
- **Parameters**:
  - `news_with_sentiment (pd.DataFrame)`: The DataFrame of news with sentiment scores, indexed by datetime.
- **Returns**: A `pandas.DataFrame` indexed by day. Columns include `mean_sentiment_score`, and counts for `positive`, `negative`, and `neutral` labels.

### `create_features(price_history_with_indicators: pd.DataFrame, daily_sentiment: pd.DataFrame) -> pd.DataFrame`

- **Purpose**: Merges the technical indicators and daily sentiment scores into a single, final DataFrame for model training. It also creates the `target` variable.
- **Parameters**:
  - `price_history_with_indicators (pd.DataFrame)`: Price data with technical indicators.
  - `daily_sentiment (pd.DataFrame)`: The aggregated daily sentiment data.
- **Returns**: A `pandas.DataFrame` containing the complete feature set and the `target` column, which is `1` if the next day's close is higher and `0` otherwise.

---

## `src/modeling.py`

**Purpose**: To manage the machine learning model's lifecycle: training, evaluation, saving, and loading.

### `train_model(...) -> Tuple[XGBClassifier, Dict, pd.Series]`

- **Purpose**: Trains a `XGBClassifier` model and evaluates its performance.
- **Parameters**:
  - `X_train, y_train, X_test, y_test`: Standard training and testing data splits (DataFrames and Series).
- **Returns**: A `tuple` containing three items: the trained `XGBClassifier` object, a `dict` of performance metrics (accuracy, classification report), and a `pd.Series` of the predictions made on the test set.

### `save_model(model: XGBClassifier, filepath: str) -> None`

- **Purpose**: Saves a trained model to a file using `joblib`.
- **Parameters**:
  - `model (XGBClassifier)`: The trained model object to save.
  - `filepath (str)`: The path where the model will be saved.
- **Returns**: `None`.

### `load_model(filepath: str) -> XGBClassifier`

- **Purpose**: Loads a pre-trained model from a `.joblib` file.
- **Parameters**:
  - `filepath (str)`: The path to the saved model file.
- **Returns**: A trained `XGBClassifier` object.

### `make_prediction(...) -> Tuple[Any, Any]`

- **Purpose**: Uses a trained model to make a prediction on the latest available data.
- **Parameters**:
  - `model (XGBClassifier)`: The trained model.
  - `latest_data (pd.DataFrame)`: A DataFrame containing the single most recent row of features.
  - `features (List[str])`: A list of feature names to use for the prediction.
- **Returns**: A `tuple` containing the prediction (e.g., `[1]`) and the prediction probabilities (e.g., `[[0.2, 0.8]]`).

---

## `src/backtesting.py`

**Purpose**: To evaluate the historical performance of the trading strategy defined by the model's signals.

### `run_backtest(...) -> Tuple[pd.DataFrame, Dict, plt.Figure]`

- **Purpose**: Runs an iterative backtest simulation. It processes trades based on the model's signals and calculates the portfolio's value over time, accounting for transaction costs.
- **Parameters**:
  - `price_history (pd.DataFrame)`: The historical price data.
  - `signals (pd.Series)`: The trading signals (1 for buy, -1 for sell) generated by the model.
  - `initial_capital (float)`: The starting capital for the simulation.
  - `transaction_cost_pct (float)`: The percentage cost per trade.
- **Returns**: A `tuple` containing: the portfolio history `DataFrame`, a `dict` of performance metrics (e.g., Sharpe Ratio, Win Rate), and a `matplotlib.Figure` object for the monthly returns heatmap.

### `calculate_performance_metrics(portfolio: pd.DataFrame) -> Dict`

- **Purpose**: Calculates key performance metrics from a completed backtest.
- **Parameters**:
  - `portfolio (pd.DataFrame)`: The portfolio history DataFrame from `run_backtest`.
- **Returns**: A `dict` where keys are metric names (e.g., "Strategy Total Return", "Sharpe Ratio") and values are the calculated results.

### `create_monthly_returns_heatmap(portfolio: pd.DataFrame) -> plt.Figure`

- **Purpose**: Creates a visual heatmap of monthly returns to identify seasonal patterns or trends in performance.
- **Parameters**:
  - `portfolio (pd.DataFrame)`: The portfolio history DataFrame.
- **Returns**: A `matplotlib.Figure` object containing the heatmap plot.

---

## `src/utils.py`

**Purpose**: To hold common utility functions used across the project.

### `get_logger(name: str) -> Logger`

- **Purpose**: Configures and retrieves a standardized logger instance for consistent, informative logging across all modules.
- **Parameters**:
  - `name (str)`: The name for the logger, typically the module's `__name__`.
- **Returns**: A configured `logging.Logger` instance.
