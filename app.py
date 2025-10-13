from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import pandas as pd
import os
from transformers import (
    pipeline, AutoTokenizer, AutoModelForSequenceClassification
)
from src.data_ingestion import get_price_history, get_news
from src.sentiment_analysis import get_sentiment
from src.feature_engineering import (
    create_technical_indicators,
    aggregate_sentiment_scores,
    create_features,
)
from src.modeling import load_model, make_prediction
from src.universal_modeling import load_universal_model, make_universal_prediction
from src.utils import get_logger
from src.backtesting import run_backtest

logger = get_logger(__name__)

st.set_page_config(layout="wide")

st.title("📈 Sentilyze")
st.write("A sentiment-driven stock momentum predictor.")


# --- Model and Tokenizer Loading ---
@st.cache_resource
def load_sentiment_analyzer():
    logger.info("Loading FinBERT sentiment analysis model...")
    tokenizer = AutoTokenizer.from_pretrained("./models/finbert-fine-tuned")
    model = AutoModelForSequenceClassification.from_pretrained(
        "./models/finbert-fine-tuned"
    )
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

@st.cache_resource
def load_universal_model_cached():
    """Loads the universal model, returns None if it doesn't exist."""
    logger.info("Loading universal prediction model...")
    model_path = "models/universal_model.pth"
    if not os.path.exists(model_path):
        logger.warning("Universal model not found. Continuing without it.")
        return None
    # TODO: The input_size needs to be known. For now, let's hardcode a placeholder.
    # This should be configured or inferred from the training script.
    input_size = 15 # Placeholder for number of features
    return load_universal_model(model_path, input_size=input_size)


sentiment_analyzer = load_sentiment_analyzer()
universal_model = load_universal_model_cached()

# --- Main App ---
ticker = st.text_input("Enter a stock ticker:", "NVDA")

model_path = f"models/{ticker}_model.joblib"
if not os.path.exists(model_path):
    st.warning(
        f"Model for {ticker} not found. Please train it first using "
        f"`docker-compose run --rm app python train.py --ticker {ticker}`"
    )
    st.stop()

model = load_model(model_path)

# --- TABS ---
tab1, tab2 = st.tabs(["Next-Day Prediction", "Backtest Analysis"])

with tab1:
    st.header(f"Predict Next-Day Momentum for {ticker}")
    if st.button("Analyze & Predict"):
        try:
            with st.spinner(f"Fetching latest data and making prediction for {ticker}..."):
                # 1. Fetch and prepare data for both models
                price_history_df = get_price_history(ticker, period="3mo")
                news_df = get_news(ticker, os.environ.get("NEWS_API_KEY"))
                news_with_sentiment_df = get_sentiment(news_df, sentiment_analyzer, ticker)
                price_history_with_indicators = create_technical_indicators(price_history_df)
                daily_sentiment = aggregate_sentiment_scores(news_with_sentiment_df)
                features_df = create_features(price_history_with_indicators, daily_sentiment)

                # 2. Initialize prediction variables
                final_prediction = None
                final_confidence = 0.0
                prediction_source = ""

                # 3. Get latest data for Specialist Model (RandomForest)
                specialist_model_path = f"models/{ticker}_model.joblib"
                specialist_model = load_model(specialist_model_path) if os.path.exists(specialist_model_path) else None

                # 4. Prepare sequence for Universal Model (LSTM)
                sequence_length = 30
                feature_columns = [col for col in features_df.columns if col not in ['target']]
                latest_sequence = features_df[feature_columns].tail(sequence_length).values

                # --- Hybrid Prediction Logic ---
                if specialist_model and universal_model and latest_sequence.shape[0] == sequence_length:
                    # --- HYBRID PREDICTION ---
                    prediction_source = "Hybrid (Specialist + Universal)"
                    
                    # Specialist prediction
                    spec_latest_features = features_df.iloc[-1:][feature_columns]
                    spec_pred, spec_conf = make_prediction(specialist_model, spec_latest_features, feature_columns)
                    spec_prob = spec_conf[0][spec_pred[0]]

                    # Universal prediction
                    uni_pred, uni_conf = make_universal_prediction(universal_model, latest_sequence)

                    # Combine results (weighted average)
                    final_prob = (spec_prob * 0.7) + (uni_conf * 0.3) # Give more weight to specialist
                    final_prediction = 1 if final_prob >= 0.5 else 0
                    final_confidence = final_prob if final_prediction == 1 else 1 - final_prob

                elif specialist_model:
                    # --- SPECIALIST ONLY ---
                    prediction_source = "Specialist"
                    spec_latest_features = features_df.iloc[-1:][feature_columns]
                    spec_pred, spec_conf = make_prediction(specialist_model, spec_latest_features, feature_columns)
                    final_prediction = spec_pred[0]
                    final_confidence = spec_conf[0][final_prediction]

                elif universal_model and latest_sequence.shape[0] == sequence_length:
                    # --- UNIVERSAL ONLY ---
                    prediction_source = "Universal"
                    uni_pred, uni_conf = make_universal_prediction(universal_model, latest_sequence)
                    final_prediction = uni_pred
                    final_confidence = uni_conf
                
                else:
                    st.error("Could not make a prediction. A trained model is not available or there is not enough data.")
                    st.stop()

                # 5. Display result
                prediction_label = "Positive" if final_prediction == 1 else "Negative"
                st.subheader(f"Prediction ({prediction_source})")
                if prediction_label == "Positive":
                    st.success(f"The predicted momentum for the next trading day is **{prediction_label}** with a confidence of **{final_confidence:.2%}**.")
                else:
                    st.error(f"The predicted momentum for the next trading day is **{prediction_label}** with a confidence of **{final_confidence:.2%}**.")

                st.subheader("Data Used for Prediction")
                st.write("Latest Price Data with Technical Indicators:")
                st.dataframe(price_history_with_indicators.tail())
                st.write("Latest News with Sentiment Analysis:")
                st.dataframe(news_with_sentiment_df.head())
        except Exception as e:
            logger.error(f"An error occurred during prediction: {e}")
            st.error(f"An error occurred: {e}. Please check the logs for more details.")

with tab2:
    st.header(f"Run Backtest Analysis for {ticker}")
    st.write(
        "This will simulate the trading strategy over the last 5 years of historical data."
    )

    # Backtest configuration
    col1, col2 = st.columns(2)
    with col1:
        initial_capital = st.number_input("Initial Capital", value=10000.0, step=1000.0)
    with col2:
        transaction_cost_pct = (
            st.number_input("Transaction Cost (%)", value=0.1, step=0.05, format="%.3f")
            / 100.0
        )

    if st.button("Run Backtest"):
        try:
            with st.spinner(
                f"Running 5-year backtest for {ticker}... This may take a moment."
            ):
                # 1. Fetch historical data
                logger.info("Fetching 5 years of historical data for backtest...")
                price_history_df = get_price_history(ticker, period="5y")
                news_df = get_news(ticker, os.environ.get("NEWS_API_KEY"))  # Using historical news for backtest

                # 2. Feature Engineering
                logger.info("Performing feature engineering for backtest...")
                news_with_sentiment_df = get_sentiment(news_df, sentiment_analyzer, ticker)
                price_history_with_indicators = create_technical_indicators(
                    price_history_df
                )
                daily_sentiment = aggregate_sentiment_scores(news_with_sentiment_df)
                features_df = create_features(
                    price_history_with_indicators, daily_sentiment
                )
                features_df = features_df.dropna()

                # 3. Generate Signals
                logger.info("Generating trading signals for backtest...")
                features = [
                    "ma7", "ma21", "rsi", "macd", "bollinger_upper", "bollinger_lower",
                    "stochastic_oscillator", "mean_sentiment_score", "positive", "negative", "neutral"
                ]
                predictions = model.predict(features_df[features])
                signals = pd.Series(predictions, index=features_df.index).replace({0: -1})

                # 4. Run Backtest
                logger.info("Running backtest simulation...")
                portfolio, metrics, heatmap_fig = run_backtest(
                    price_history=price_history_df,
                    signals=signals,
                    initial_capital=initial_capital,
                    transaction_cost_pct=transaction_cost_pct,
                )

                # 5. Display Results
                st.subheader("Backtest Performance Metrics")
                row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
                row1_col1.metric("Strategy Return", metrics["Strategy Total Return"])
                row1_col2.metric("Buy & Hold Return", metrics["Buy & Hold Total Return"])
                row1_col3.metric("Sharpe Ratio", metrics["Sharpe Ratio"])
                row1_col4.metric("Sortino Ratio", metrics["Sortino Ratio"])

                row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
                row2_col2.metric("Win Rate", metrics["Win Rate"])
                row2_col1.metric("Total Trades", metrics["Total Trades"])
                row2_col3.metric("Strategy Max Drawdown", metrics["Strategy Max Drawdown"])
                row2_col4.metric(
                    "Buy & Hold Max Drawdown", metrics["Buy & Hold Max Drawdown"]
                )

                st.subheader("Portfolio Value Over Time")
                chart_data = portfolio[["total", "benchmark"]].rename(
                    columns={"total": "Strategy", "benchmark": "Buy & Hold"}
                )
                st.line_chart(chart_data)

                st.subheader("Monthly Returns Heatmap")
                st.pyplot(heatmap_fig)
        except Exception as e:
            logger.error(f"An error occurred during backtest: {e}")
            st.error(f"An error occurred: {e}. Please check the logs for more details.")