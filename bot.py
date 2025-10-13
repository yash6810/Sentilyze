
import os
import pandas as pd
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from src.data_ingestion import get_news, get_price_history
from src.sentiment_analysis import get_sentiment_with_caching
from src.feature_engineering import create_technical_indicators, aggregate_sentiment_scores, create_features
from src.modeling import load_model, make_prediction
from src.universal_modeling import load_universal_model, make_universal_prediction
from src.utils import get_logger
import smtplib
from email.message import EmailMessage

# --- Setup ---
load_dotenv() # Load environment variables from .env file
logger = get_logger(__name__)

# --- Model Loading ---
logger.info("Loading models...")
sentiment_analyzer = pipeline("sentiment-analysis", model=AutoModelForSequenceClassification.from_pretrained('./models/finbert-fine-tuned'), tokenizer=AutoTokenizer.from_pretrained('./models/finbert-fine-tuned'))

universal_model_path = "models/universal_model.pth"
if os.path.exists(universal_model_path):
    # TODO: This input_size needs to be configured correctly.
    universal_model = load_universal_model(universal_model_path, input_size=15)
else:
    logger.warning("Universal model not found. The bot will only be able to use specialist models.")
    universal_model = None

def send_email_report(report_df):
    """Sends the report DataFrame via email."""
    logger.info("Attempting to send email report...")
    host, port, user, password, recipient = os.getenv("EMAIL_HOST"), os.getenv("EMAIL_PORT"), os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASSWORD"), os.getenv("EMAIL_RECIPIENT")

    if not all([host, port, user, password, recipient]):
        logger.warning("Email configuration is incomplete. Skipping email notification.")
        return

    msg = EmailMessage()
    msg.set_content(report_df.to_html(index=False), subtype='html')
    msg['Subject'] = f"Sentilyze Daily Momentum Report - {pd.Timestamp.now().strftime('%Y-%m-%d')}"
    msg['From'] = user
    msg['To'] = recipient

    try:
        with smtplib.SMTP(host, int(port)) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
        logger.info(f"Email report successfully sent to {recipient}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")

def run_bot():
    """
    Main function to run the analysis bot using the hybrid prediction system.
    """
    logger.info("Bot starting...")
    try:
        with open('stocks.txt', 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(tickers)} tickers to analyze: {tickers}")
    except FileNotFoundError:
        logger.error("stocks.txt not found! Please create it and add stock tickers, one per line.")
        return

    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        logger.error("NEWS_API_KEY not found in .env file!")
        return

    results = []
    for ticker in tickers:
        logger.info(f"--- Analyzing {ticker} ---")
        try:
            # 1. Fetch and prepare data
            price_history_df = get_price_history(ticker, period="3mo")
            news_df = get_news(ticker, api_key)
            news_with_sentiment_df = get_sentiment_with_caching(news_df, sentiment_analyzer, ticker)
            price_history_with_indicators = create_technical_indicators(price_history_df)
            daily_sentiment = aggregate_sentiment_scores(news_with_sentiment_df)
            features_df = create_features(price_history_with_indicators, daily_sentiment)

            # 2. Initialize prediction variables
            final_prediction, final_confidence, prediction_source = None, 0.0, ""

            # 3. Hybrid Prediction Logic
            specialist_model_path = f'models/{ticker}_model.joblib'
            specialist_model = load_model(specialist_model_path) if os.path.exists(specialist_model_path) else None

            sequence_length = 30
            feature_columns = [col for col in features_df.columns if col not in ['target']]
            latest_sequence = features_df[feature_columns].tail(sequence_length).values

            if specialist_model and universal_model and latest_sequence.shape[0] == sequence_length:
                prediction_source = "Hybrid"
                spec_latest = features_df.iloc[-1:][feature_columns]
                spec_pred, spec_conf = make_prediction(specialist_model, spec_latest, feature_columns)
                spec_prob = spec_conf[0][spec_pred[0]]
                uni_pred, uni_conf = make_universal_prediction(universal_model, latest_sequence)
                final_prob = (spec_prob * 0.7) + (uni_conf * 0.3)
                final_prediction = 1 if final_prob >= 0.5 else 0
                final_confidence = final_prob if final_prediction == 1 else 1 - final_prob
            elif specialist_model:
                prediction_source = "Specialist"
                spec_latest = features_df.iloc[-1:][feature_columns]
                spec_pred, spec_conf = make_prediction(specialist_model, spec_latest, feature_columns)
                final_prediction = spec_pred[0]
                final_confidence = spec_conf[0][final_prediction]
            elif universal_model and latest_sequence.shape[0] == sequence_length:
                prediction_source = "Universal"
                uni_pred, uni_conf = make_universal_prediction(universal_model, latest_sequence)
                final_prediction = uni_pred
                final_confidence = uni_conf
            else:
                logger.warning(f"Could not make a prediction for {ticker}. No suitable model or not enough data.")
                continue

            pred_label = "Positive" if final_prediction == 1 else "Negative"
            results.append({
                "Ticker": ticker,
                "Prediction": pred_label,
                "Confidence": f"{final_confidence:.2%}",
                "Source": prediction_source
            })
            logger.info(f"Analysis for {ticker} complete. Prediction: {pred_label} (Source: {prediction_source})")

        except Exception as e:
            logger.error(f"An error occurred while analyzing {ticker}: {e}")

    # --- Generate and Send Report ---
    if results:
        logger.info("--- Daily Momentum Report ---")
        report_df = pd.DataFrame(results)
        print(report_df.to_string(index=False))
        send_email_report(report_df)
    else:
        logger.warning("No analysis was completed. No report to generate.")

    logger.info("Bot finished.")

if __name__ == "__main__":
    run_bot()
