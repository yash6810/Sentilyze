import os
import pandas as pd
import numpy as np
from src.utils import get_logger, create_sequences
from src.preprocessing import bulk_preprocess_data
from tqdm import tqdm
import argparse
import mlflow

logger = get_logger(__name__)

def optimize_data(max_tickers: int | None = None, output_path: str = "data/processed/universal_data.parquet"):
    """
    Orchestrates the data acquisition, cleaning, preprocessing, and feature engineering
    for the universal model, saving the result to a file.
    """
    logger.info("--- Starting Data Optimization Pipeline ---")

    raw_data_dir = os.path.join("data", "raw")
    available_files = os.listdir(raw_data_dir)
    
    dataset_path = None
    for f in available_files:
        if "headlines" in f.lower() and f.endswith('.csv'):
            dataset_path = os.path.join(raw_data_dir, f)
            logger.info(f"Found dataset file: {dataset_path}")
            break

    if dataset_path is None:
        logger.error(f"Could not find a suitable dataset file in {raw_data_dir}. Please run download_dataset.py first.")
        return None

    # Step 1.1: Load and Clean News Data
    logger.info(f"Loading news data from {dataset_path}...")
    news_df = pd.read_csv(dataset_path, encoding='ISO-8859-1')
    news_df.rename(columns={"date": "Date", "headline": "Title", "stock": "Ticker"}, inplace=True)
    news_df.dropna(subset=["Title", "Ticker", "Date"], inplace=True) # Drop rows with missing critical columns
    news_df.drop_duplicates(subset=["Title", "Date", "Ticker"], inplace=True)
    news_df["Ticker"] = news_df["Ticker"].str.upper() # Standardize ticker symbols
    news_df["Date"] = pd.to_datetime(news_df["Date"])

    unique_tickers = news_df["Ticker"].unique().tolist() # Convert to list for bulk_preprocess_data
    
    # Use bulk_preprocess_data to get the merged_df
    merged_df = bulk_preprocess_data(unique_tickers, max_tickers, period="max")

    if merged_df.empty:
        logger.error("Bulk preprocessing failed or returned empty DataFrame. Aborting.")
        return None, None, None # Return None for all expected values if merged_df is empty

    # Create the target variable for each stock
    merged_df['target'] = merged_df.groupby('Ticker')['Close'].shift(-1) > merged_df['Close']
    merged_df['target'] = merged_df['target'].astype(int)
    merged_df.dropna(subset=['target'], inplace=True) # Drop the last day for each stock

    logger.info("Feature engineering complete.")
    logger.info(f"Columns in merged_df: {merged_df.columns.tolist()}")

    # Save the merged dataframe for backtesting purposes
    merged_df_path = "data/processed/universal_merged_data.parquet"
    merged_df.to_parquet(merged_df_path)
    logger.info(f"Saved merged data for backtesting to {merged_df_path}")

    # Step 1.5: Create Sequences
    logger.info("Creating sequences for LSTM model...")
    X, y = create_sequences(merged_df, sequence_length=30)
    logger.info(f"Created {len(X)} sequences.")

    # Save the processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(X.reshape(X.shape[0], -1)).to_parquet(output_path.replace(".parquet", "_X.parquet"))
    pd.DataFrame(y).to_parquet(output_path.replace(".parquet", "_y.parquet"))
    logger.info(f"Processed data saved to {output_path.replace('.parquet', '_X.parquet')} and {output_path.replace('.parquet', '_y.parquet')}")

    logger.info("--- Data Optimization Pipeline Finished ---")
    return X, y, merged_df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize data for universal model training.')
    parser.add_argument('--max_tickers', type=int, default=None, help='Optional: Limit the number of tickers to process.')
    parser.add_argument('--output_path', type=str, default="data/processed/universal_data.parquet", help='Path to save the processed data.')
    args = parser.parse_args()
    
    with mlflow.start_run(run_name="Data Optimization"):
        mlflow.log_param("max_tickers", args.max_tickers)
        mlflow.log_param("output_path", args.output_path)
        X, y, merged_df = optimize_data(max_tickers=args.max_tickers, output_path=args.output_path)
        if X is not None and y is not None:
            mlflow.log_artifact(args.output_path.replace(".parquet", "_X.parquet"))
            mlflow.log_artifact(args.output_path.replace(".parquet", "_y.parquet"))
            mlflow.log_artifact("data/processed/universal_merged_data.parquet")
            logger.info("Data optimization run logged to MLflow.")
        else:
            logger.error("Data optimization failed. No artifacts logged.")
