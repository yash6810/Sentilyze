# Sentilyze Project Analysis & Improvement Plan

This document contains an analysis of the Sentilyze codebase based on a series of technical questions, followed by a set of actionable recommendations for improving the project's robustness, reliability, and performance.

## Codebase Q&A

**1. Are there duplicate logics across modules (e.g. same date handling)?**

Date handling logic is present in multiple modules. For example, `data_ingestion.py` parses dates when loading cached price history, and `feature_engineering.py` parses the `publishedAt` field from the news data. While the code isn't identical, the responsibility for date conversion is spread out rather than being centralized in the data ingestion step. This could be improved for better separation of concerns.

---

**2. Do all modules handle errors gracefully (timeouts, missing values, API limits)?**

* **Timeouts/API Limits:** Error handling is inconsistent. `data_ingestion.py` has a good retry mechanism for fetching price data but lacks similar handling for the NewsAPI. A failure in the NewsAPI call would crash the application.
* **Missing Values:** The project handles missing data well. The feature engineering pipeline uses `ffill()` and `fillna(0)` to ensure there are no gaps in the data before training or prediction.

---

**3. Is temporal leakage avoided (i.e. you don’t peek into future data)?**

**No, there is a critical issue here.** The `train.py` script uses `sklearn.model_selection.train_test_split` with its default settings, which shuffles the data randomly. For time-series forecasting, this is incorrect as it causes **temporal leakage**—the model is trained on data points that occur *after* the data in the test set. The data should be split chronologically to prevent the model from learning from the future.

---

**4. Are all modules tested and do tests reflect realistic scenarios?**

* There is good test coverage for core modules like `data_ingestion`, `feature_engineering`, and `backtesting`.
* However, `src/modeling.py` and `src/utils.py` appear to be untested.
* There is also a tooling inconsistency: some tests are written for `pytest`, but the CI pipeline runs them with `unittest`. The project should standardize on a single test runner.

---

**5. Are the interfaces / function signatures clean and decoupled?**

Yes, for the most part.

* The **sentiment engine** is very well-decoupled. The main functions accept a `sentiment_analyzer` object, so you could easily swap in a different model.
* The **machine learning model** functions are designed around a scikit-learn compatible API (`.fit`, `.predict`), making it easy to swap `XGBClassifier` for other similar models.
* **Data sources** are tied to their specific libraries, but they expose a simple interface to the rest of the app.

---

**6. Is there caching or memoization (especially in news sentiment, which is expensive)?**

* **Yes, for model loading and price data.** The app correctly uses `@st.cache_resource` to load the large FinBERT model only once, and `data_ingestion.py` caches price history in CSV files.
* **No, for sentiment results.** The expensive process of analyzing news sentiment is run every time, even for the same news articles. Caching these results would significantly improve performance, especially in the backtesting module.

---

**7. Are file paths and configurations robust?**

Yes. The project does a good job of managing paths. It programmatically determines the project's root directory to create reliable absolute paths for caching, and uses relative paths for models, which works well across different environments (local, Docker, CI).

---

**8. Are resource-intensive operations batched or asynchronous?**

* **Batching:** Yes. The sentiment analysis pipeline processes the list of news articles as a batch, which is very efficient.
* **Asynchronous:** No. The application is synchronous. Long-running operations will block the UI, but the use of Streamlit's `st.spinner` provides good user feedback during these waits.

---

**9. Does app.py degrade gracefully (e.g. if model missing, no news, API failed)?**

It's mixed.

* **Model Missing:** Yes, it handles this perfectly with a clear warning and instructions.
* **No News:** Yes, if no news is found, sentiment features default to zero, and the prediction proceeds using only technical indicators.
* **API Failure:** No. An unhandled exception from an API call will crash the app. This is a key area for improvement.

---

**10. Is logging consistent and informative?**

Yes. The logging is excellent. It uses a centralized logger with a consistent format that includes a timestamp, module name, log level, and a clear message, providing a great trace of the application's execution.

---

## Recommendations for Near-Term Improvement

Based on the analysis above, here are the most critical areas for improving the **existing per-stock architecture**, ordered by priority.

### 1. Fix Temporal Leakage in `train.py` (Priority: Critical)

The current model evaluation is likely inaccurate due to data shuffling. This is the most important issue to fix.

* **Action:** In `train.py`, modify the data splitting logic. Instead of using `train_test_split`, manually split the data chronologically. Ensure the DataFrame is sorted by date, then select the first 80% of rows for training and the remaining 20% for testing.

### 2. Improve Robustness in `app.py` (Priority: High)

The Streamlit app should not crash due to external factors like API failures.

* **Action:** In `app.py`, wrap the data fetching calls (`get_price_history`, `get_news`) and the subsequent feature engineering/prediction logic within a `try...except` block. If an exception occurs, display a user-friendly error message using `st.error()`.

### 3. Implement Caching for News and Sentiment (Priority: Medium)

The application currently re-fetches news and re-calculates sentiment on every run, which is inefficient and can lead to hitting API rate limits.

* **Action 1:** In `data_ingestion.py`, add file-based caching to the `get_news` function, similar to how `get_price_history` is cached.
* **Action 2:** Cache the results of the sentiment analysis. This can be done by saving the `news_with_sentiment_df` DataFrame to a file after the analysis is performed for the first time.

### 4. Standardize Testing and Increase Coverage (Priority: Medium)

A consistent and comprehensive test suite builds confidence and prevents regressions.

* **Action 1:** Decide on a single testing framework. `pytest` is recommended as it is powerful and already used in one of the test files.
* **Action 2:** Update the CI pipeline in `.github/workflows/ci.yml` to install and run tests using `pytest` (`pytest tests/`).
* **Action 3:** Write new tests for `src/modeling.py` and `src/utils.py` to increase overall test coverage.

### 5. Refactor Date Handling (Priority: Low)

Consolidating data-related logic will make the code cleaner and easier to maintain.

* **Action:** Refactor the `data_ingestion` module to be the single source of truth for data and its format. Ensure that both `get_news` and `get_price_history` return DataFrames with a consistent, timezone-aware DatetimeIndex.

---

## Roadmap to the Hybrid Prediction System

This section outlines the long-term vision for the project: evolving from a per-stock model architecture to a sophisticated hybrid system that combines a universal, generalist model with specialized, per-stock models.

### Phase 1: Develop the Universal "Generalist" Model

The first step is to build the foundation of the hybrid system: a single model trained on a vast dataset that understands general market dynamics.

*   **Action 1: Create a New Training Script (`train_universal.py`)**: This script will be responsible for training the universal model. It will use the Kaggle dataset (from `download_dataset.py`) and will need to implement a more complex, sequence-aware model architecture (e.g., LSTM, GRU, or a Transformer) using a framework like Keras or PyTorch.
*   **Action 2: Develop a Universal Modeling Module**: Create a new `universal_modeling.py` module to handle the loading of and prediction with this new type of model, keeping it separate from the existing scikit-learn logic.

### Phase 2: Integrate into a Hybrid System

With the universal model built, the next step is to integrate it into the user-facing application.

*   **Action: Overhaul `app.py`**: Refactor the Streamlit application to implement the core hybrid logic. When a user enters a ticker, the app will:
    1.  Check if a specialist model exists for that ticker.
    2.  If YES, get predictions from both the specialist and the universal model. Combine the results (e.g., by averaging confidence scores) to produce a robust, hybrid prediction.
    3.  If NO, fall back to providing the prediction from the universal model alone.
    4.  The UI must be updated to clearly communicate the source of the prediction (specialist, universal, or hybrid).

### Phase 3: Upgrade the Automated Bot

Finally, the capabilities of the hybrid system should be extended to the automated reporting bot.

*   **Action: Refactor `bot.py`**: Update the bot to use the same hybrid prediction logic from `app.py`. This will make its daily email reports more comprehensive and reliable, as it can provide analysis for any stock and richer analysis for stocks with specialist models.