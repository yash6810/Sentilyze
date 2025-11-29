from dotenv import load_dotenv
load_dotenv()
from src.utils import get_logger
from src.preprocessing import preprocess_data

logger = get_logger(__name__)

def run_preprocessing(ticker: str):
    logger.info(f"Starting preprocessing for {ticker}...")
    features_df, _, _ = preprocess_data(ticker)
    features_df.to_csv('temp_features_df.csv')
    logger.info("Preprocessing complete. Saved features_df to temp_features_df.csv")

if __name__ == '__main__':
    run_preprocessing('NVDA')
