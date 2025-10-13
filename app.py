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
from src.modeling import load_model
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


sentiment_analyzer = load_sentiment_analyzer()

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
        st.write("Button clicked!")

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
        with st.spinner(
            f"Running 5-year backtest for {ticker}... This may take a moment."
        ):
            # 1. Fetch historical data
            logger.info("Fetching 5 years of historical data for backtest...")
            price_history_df = get_price_history(ticker, period="5y")
            news_df = get_news(ticker, os.environ.get("NEWS_API_KEY"))  # Using historical news for backtest

            # 2. Feature Engineering
            logger.info("Performing feature engineering for backtest...")
            news_with_sentiment_df = get_sentiment(news_df, sentiment_analyzer)
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
                "ma7",
                "ma21",
                "rsi",
                "macd",
                "bollinger_upper",
                "bollinger_lower",
                "stochastic_oscillator",
                "mean_sentiment_score",
                "positive",
                "negative",
                "neutral",
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
