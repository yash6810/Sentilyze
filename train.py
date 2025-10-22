from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import argparse
import json
import shap
import mlflow
import numpy as np
from src.data_ingestion import get_news, get_price_history
from src.sentiment_analysis import get_sentiment
from src.feature_engineering import create_technical_indicators, aggregate_sentiment_scores, create_features
from src.modeling import train_model, save_model
from src.backtesting import run_backtest
from src.config import FEATURES
from src.utils import get_logger
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from transformers import pipeline as transformers_pipeline, AutoTokenizer, AutoModelForSequenceClassification

logger = get_logger(__name__)

def main(ticker: str) -> None:
    """
    Main function to run the training pipeline for a given stock ticker.

    Args:
        ticker (str): The stock ticker to train the model on.
    """
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
    news_with_sentiment_df = get_sentiment(news_df, sentiment_analyzer, ticker)

    # 3. Feature Engineering
    logger.info("Creating features...")
    price_history_with_indicators = create_technical_indicators(price_history_df)
    daily_sentiment = aggregate_sentiment_scores(news_with_sentiment_df)
    features_df = create_features(price_history_with_indicators, daily_sentiment)

    # 4. Prepare data for training
    logger.info("Preparing data for training...")
    features_df = features_df.dropna().sort_index()  # Ensure data is sorted by date
    features = FEATURES
    target = 'target'

    X = features_df[features]
    y = features_df[target]
    idx = features_df.index

    # Chronological split to prevent temporal leakage
    split_point = int(len(X) * 0.8)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]
    idx_train, idx_test = idx[:split_point], idx[split_point:]

    with mlflow.start_run():
        mlflow.log_param("ticker", ticker)

        # 5. Train Model
        model, metrics, y_pred = train_model(X_train, y_train, X_test, y_test)

        mlflow.log_params(metrics['best_params'])
        mlflow.log_metric("accuracy", metrics['accuracy'])

        logger.info(f"Best parameters found: {metrics['best_params']}")
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
        mlflow.log_metrics(backtest_metrics)

        # Log classification report as an artifact
        classification_report_str = metrics.get('classification_report', "N/A")
        if classification_report_str != "N/A":
            report_path = f"results/{ticker}_classification_report.txt"
            with open(report_path, 'w') as f:
                f.write(classification_report_str)
            mlflow.log_artifact(report_path)
            logger.info(f"Saved classification report to {report_path}")

        # 7. Save Model and Results
        logger.info(f"Saving model to models/{ticker}_model.joblib...")
        save_model(model, f"models/{ticker}_model.joblib")
        mlflow.sklearn.log_model(model, "model")

        # Save the heatmap
        heatmap_fig.savefig(f"results/{ticker}_monthly_returns_heatmap.png")
        mlflow.log_artifact(f"results/{ticker}_monthly_returns_heatmap.png")
        logger.info(
            f"Saved monthly returns heatmap to results/{ticker}_monthly_returns_heatmap.png"
        )

        # Save metrics to a JSON file
        with open(f"results/{ticker}_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact(f"results/{ticker}_metrics.json")
        logger.info(f"Saved metrics to results/{ticker}_metrics.json")

        # Save portfolio to a CSV file
        portfolio.to_csv(f"results/{ticker}_portfolio.csv")
        mlflow.log_artifact(f"results/{ticker}_portfolio.csv")
        logger.info(f"Saved portfolio to results/{ticker}_portfolio.csv")

        # Save feature importances to a CSV file
        feature_importances = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        feature_importances.to_csv(f"results/{ticker}_feature_importances.csv", index=False)
        mlflow.log_artifact(f"results/{ticker}_feature_importances.csv")
        logger.info(f"Saved feature importances to results/{ticker}_feature_importances.csv")

        # Calculate and save SHAP values
        logger.info("Calculating SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        np.save(f"results/{ticker}_shap_values.npy", shap_values)
        mlflow.log_artifact(f"results/{ticker}_shap_values.npy")
        logger.info(f"Saved SHAP values to results/{ticker}_shap_values.npy")

        # Save X_test to a CSV file
        X_test.to_csv(f"results/{ticker}_X_test.csv")
        mlflow.log_artifact(f"results/{ticker}_X_test.csv")
        logger.info(f"Saved X_test to results/{ticker}_X_test.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a sentiment-driven stock momentum predictor.')
    parser.add_argument('--ticker', type=str, default='NVDA', help='Stock ticker to train the model on.')
    args = parser.parse_args()
    main(args.ticker)