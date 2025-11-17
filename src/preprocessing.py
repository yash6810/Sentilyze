import pandas as pd
import yfinance as yf
from tqdm import tqdm
import os
import time
from src.utils import get_logger
from src.data_ingestion import get_price_history, get_news
from src.sentiment_analysis import get_sentiment
from src.feature_engineering import create_technical_indicators, aggregate_sentiment_scores, create_features
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from functools import lru_cache
from typing import Any

logger = get_logger(__name__)

# Cache the sentiment analyzer to avoid reloading on every call
@lru_cache(maxsize=1)
def _load_sentiment_analyzer() -> Any:
    """
    Loads the FinBERT sentiment analysis model and tokenizer from the local
    './models/finbert-fine-tuned' directory and returns a Hugging Face pipeline.
    """
    logger.info("Loading FinBERT sentiment analysis model for preprocessing...")
    tokenizer = AutoTokenizer.from_pretrained("./models/finbert-fine-tuned")
    model = AutoModelForSequenceClassification.from_pretrained("./models/finbert-fine-tuned")
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def clean_headline_data(input_path: str, output_path: str, cache_dir: str = "data/processed"):
    """
    Cleans a headline CSV file by removing rows with invalid stock tickers.
    Caches the list of valid and invalid tickers to speed up subsequent runs.

    Args:
        input_path (str): The path to the input CSV file.
        output_path (str): The path to save the cleaned CSV file.
        cache_dir (str): The directory to store ticker validation cache files.
    """
    logger.info(f"Reading data from {input_path}...")
    df = pd.read_csv(input_path, encoding='ISO-8859-1')
    
    if "headline" in df.columns:
        df.rename(columns={"date": "Date", "headline": "Title", "stock": "Ticker"}, inplace=True)
    
    original_rows = len(df)
    unique_tickers = df["Ticker"].unique()
    
    # --- Ticker Validation Caching ---
    os.makedirs(cache_dir, exist_ok=True)
    valid_tickers_cache_path = os.path.join(cache_dir, "valid_tickers.json")
    
    if os.path.exists(valid_tickers_cache_path):
        logger.info(f"Loading valid tickers from cache: {valid_tickers_cache_path}")
        with open(valid_tickers_cache_path, "r") as f:
            import json
            valid_tickers = json.load(f)
    else:
        logger.info("No valid tickers cache found. Validating all tickers...")
        valid_tickers = []
        invalid_tickers = []
        
        for ticker in tqdm(unique_tickers, desc="Validating tickers"):
            is_valid = False
            for attempt in range(3):
                try:
                    stock = yf.Ticker(ticker)
                    if not stock.history(period="1d").empty:
                        is_valid = True
                        break
                except Exception as e:
                    logger.warning(f"Error validating {ticker} on attempt {attempt + 1}: {e}")
                    time.sleep(2) # Wait before retrying
            
            if is_valid:
                valid_tickers.append(ticker)
            else:
                invalid_tickers.append(ticker)
                
        logger.info(f"Found {len(valid_tickers)} valid tickers and {len(invalid_tickers)} invalid tickers.")
        if invalid_tickers:
            logger.info(f"Invalid tickers found: {invalid_tickers[:20]}...")
            
        # Cache the results
        with open(valid_tickers_cache_path, "w") as f:
            import json
            json.dump(valid_tickers, f)
        logger.info(f"Saved valid tickers to cache: {valid_tickers_cache_path}")

    # Filter the DataFrame
    cleaned_df = df[df["Ticker"].isin(valid_tickers)]
    cleaned_rows = len(cleaned_df)
    
    logger.info(f"Removed {original_rows - cleaned_rows} rows with invalid tickers.")
    
    logger.info(f"Saving cleaned data to {output_path}...")
    cleaned_df.to_csv(output_path, index=False)
    logger.info("Done.")


def preprocess_data(ticker: str, period: str = "1y") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates the data acquisition, sentiment analysis, and feature engineering
    for a given ticker.

    Args:
        ticker (str): The stock ticker to preprocess data for.
        period (str): The time period for the data (e.g., "1y", "5y", "max").

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing:
            - features_df: A DataFrame with engineered features ready for model training.
            - price_history_with_indicators: The price history with technical indicators.
            - news_with_sentiment_df: The news headlines with sentiment scores.
    """
    logger.info(f"Starting data preprocessing for {ticker}...")

    # 1. Fetch data
    logger.info("Fetching data...")
    news_df = get_news(ticker, os.environ.get("NEWS_API_KEY"))
    price_history_df = get_price_history(ticker, period=period)

    # 2. Analyze sentiment
    logger.info("Analyzing sentiment...")
    sentiment_analyzer = _load_sentiment_analyzer()
    news_with_sentiment_df = get_sentiment(news_df, sentiment_analyzer, ticker)

    # 3. Feature Engineering
    logger.info("Creating features...")
    price_history_with_indicators = create_technical_indicators(price_history_df)
    daily_sentiment = aggregate_sentiment_scores(news_with_sentiment_df)
    features_df = create_features(price_history_with_indicators, daily_sentiment)
    features_df = features_df.dropna().sort_index()

    logger.info(f"Preprocessing for {ticker} complete. Shape: {features_df.shape}")
    return features_df, price_history_with_indicators, news_with_sentiment_df


def bulk_preprocess_data(tickers: list[str], max_tickers: int | None = None, period: str = "1y") -> pd.DataFrame:
    """
    Orchestrates the data acquisition, sentiment analysis, and feature engineering
    for a list of tickers, and concatenates them into a single DataFrame.

    Args:
        tickers (list[str]): A list of stock tickers to preprocess data for.
        max_tickers (int | None): Optional: Limit the number of tickers to process.
        period (str): The time period for the data (e.g., "1y", "5y", "max").

    Returns:
        pd.DataFrame: A DataFrame with engineered features for all specified tickers.
    """
    logger.info("Starting bulk data preprocessing...")
    all_features_dfs = []
    processed_tickers_count = 0

    for ticker in tqdm(tickers, desc="Bulk Preprocessing"):
        if max_tickers and processed_tickers_count >= max_tickers:
            logger.info(f"Reached max_tickers limit of {max_tickers}. Stopping bulk preprocessing.")
            break
        try:
            features_df, _, _ = preprocess_data(ticker, period=period)
            if not features_df.empty:
                features_df['Ticker'] = ticker
                all_features_dfs.append(features_df)
                processed_tickers_count += 1
            else:
                logger.warning(f"No features generated for {ticker}. Skipping.")
        except Exception as e:
            logger.error(f"Error preprocessing data for {ticker}: {e}. Skipping this ticker.")
            
    if not all_features_dfs:
        logger.error("No data processed for any ticker.")
        return pd.DataFrame()

    final_df = pd.concat(all_features_dfs)
    # The Ticker column is already present, so we can set the index directly
    if 'Date' in final_df.columns:
        final_df = final_df.set_index(['Ticker', 'Date'])
    else: # In case Date is already the index
        final_df = final_df.reset_index().set_index(['Ticker', 'Date'])

    logger.info(f"Bulk data preprocessing complete. Final shape: {final_df.shape}")
    return final_df