from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import argparse
from src.data_ingestion import get_news, get_price_history
from src.sentiment_analysis import get_sentiment_with_caching
from src.feature_engineering import create_technical_indicators, aggregate_sentiment_scores, create_features
from src.modeling import train_model, save_model
from src.backtesting import run_backtest
from src.utils import get_logger
from transformers import pipeline as transformers_pipeline, AutoTokenizer, AutoModelForSequenceClassification

logger = get_logger(__name__)

def main(ticker):
    logger.info(f"Starting training pipeline for {ticker}...")

    import os

# 1. Fetch data
    logger.info("Fetching data...")
    news_df = get_news(ticker, os.environ.get("NEWS_API_KEY"))
    price_history_df = get_price_history(ticker)

    # 2. Analyze sentiment
    logger.info("Analyzing sentiment...")
    tokenizer = AutoTokenizer.from_pretrained('./models/finbert-fine-tuned')
    finbert_model = AutoModelForSequenceClassification.from_pretrained('./models/finbert-fine-tuned')
    sentiment_analyzer = transformers_pipeline("sentiment-analysis", model=finbert_model, tokenizer=tokenizer)
    news_with_sentiment_df = get_sentiment_with_caching(news_df, sentiment_analyzer, ticker)

    # 3. Feature Engineering
    logger.info("Creating features...")
    price_history_with_indicators = create_technical_indicators(price_history_df)
    daily_sentiment = aggregate_sentiment_scores(news_with_sentiment_df)
    features_df = create_features(price_history_with_indicators, daily_sentiment)

    # 4. Prepare data for training
    logger.info("Preparing data for training...")
    features_df = features_df.dropna().sort_index()  # Ensure data is sorted by date
    features = ['ma7', 'ma21', 'rsi', 'macd', 'bollinger_upper', 'bollinger_lower', 'stochastic_oscillator', 'mean_sentiment_score', 'positive', 'negative', 'neutral']
    target = 'target'

    X = features_df[features]
    y = features_df[target]
    idx = features_df.index

    # Chronological split to prevent temporal leakage
    split_point = int(len(X) * 0.8)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    idx_train, idx_test = idx[:split_point], idx[split_point:]


    # 5. Train Model
    logger.info("Training model...")
    model, metrics, y_pred = train_model(X_train, y_train, X_test, y_test)

    logger.info("Training complete.")
    logger.info(f"Model accuracy: {metrics['accuracy']:.4f}")

    # 6. Run Backtest
    logger.info("Running backtest on test set predictions...")
    test_price_history = price_history_df.loc[idx_test]
    # Simple signal: 1 for buy (positive prediction), 0 for hold/sell
    signals = pd.Series(y_pred, index=idx_test)
    signals = signals.replace({0: -1})
    portfolio, backtest_metrics, heatmap_fig = run_backtest(
        test_price_history, signals
    )
    logger.info(f"Backtest performance: {backtest_metrics}")

    # 7. Save Model and Results
    logger.info(f"Saving model to models/{ticker}_model.joblib...")
    save_model(model, f"models/{ticker}_model.joblib")

    # Save the heatmap
    heatmap_fig.savefig(f"results/{ticker}_monthly_returns_heatmap.png")
    logger.info(
        f"Saved monthly returns heatmap to results/{ticker}_monthly_returns_heatmap.png"
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a sentiment-driven stock momentum predictor.')
    parser.add_argument('--ticker', type=str, default='NVDA', help='Stock ticker to train the model on.')
    args = parser.parse_args()
    main(args.ticker)