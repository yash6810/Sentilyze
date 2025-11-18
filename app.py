from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import pandas as pd
import requests
import json
import os
import shap
import numpy as np
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
from typing import Any, Dict
from transformers import (
    pipeline, AutoTokenizer, AutoModelForSequenceClassification
)
from src.preprocessing import preprocess_data
from src.modeling import load_model, get_prediction_on_latest_data
from src.universal_modeling import load_universal_model, get_universal_prediction_on_latest_data
from src.utils import get_logger
from src.backtesting import run_backtest
from src.config import FEATURES

import mlflow

logger = get_logger(__name__)

def get_latest_mlflow_run_data(ticker: str) -> Dict[str, Any] | None:
    """
    Retrieves the latest MLflow run data (metrics and artifact URIs) for a given ticker.
    """
    # Ensure MLflow is configured to use the local 'mlruns' directory
    mlflow.set_tracking_uri("file:./mlruns")

    runs = mlflow.search_runs(
        experiment_ids=["0"],  # Assuming experiment_id 0 for now
        filter_string=f"params.ticker = '{ticker}'",
        order_by=["start_time DESC"],
        max_results=1
    )

    if runs.empty:
        return None

    latest_run = runs.iloc[0]
    run_id = latest_run.run_id
    artifact_uri = latest_run.artifact_uri

    # Fetch metrics
    metrics = mlflow.get_run(run_id).data.metrics

    return {
        "run_id": run_id,
        "metrics": metrics,
        "artifact_uri": artifact_uri
    }


@st.cache_resource
def load_sentiment_analyzer() -> Any:
    """
    Loads the FinBERT sentiment analysis model and tokenizer from the local
    './models/finbert-fine-tuned' directory and returns a Hugging Face pipeline.

    The model is cached using Streamlit's cache_resource to prevent reloading
    on every app rerun.

    Returns:
        Any: A Hugging Face sentiment-analysis pipeline object.
    """
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
    config_path = "models/universal_model_config.json"

    if not os.path.exists(model_path) or not os.path.exists(config_path):
        logger.warning("Universal model or config not found. Continuing without it.")
        return None

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            input_size = config["input_size"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error loading universal model config: {e}")
        return None

    return load_universal_model(model_path, input_size=input_size)



def display_prediction_results(
    prediction_label: str,
    prediction_source: str,
    final_confidence: float,
    price_history_with_indicators: pd.DataFrame,
    news_with_sentiment_df: pd.DataFrame
) -> None:
    """
    Displays the prediction results in a structured and analytical way.
    """
    st.subheader(f"Prediction ({prediction_source})")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Model Signal", value=prediction_label)
    
    with col2:
        st.metric(label="Confidence Score", value=f"{final_confidence:.2%}")

    with st.expander("View Data Used for Prediction"):
        st.write("Latest Price Data with Technical Indicators:")
        st.dataframe(price_history_with_indicators.tail())
        st.write("Latest News with Sentiment Analysis:")
        st.dataframe(news_with_sentiment_df.head())

def parse_classification_report(report_path: str) -> Dict[str, float]:
    """
    Parses a classification report from a text file and returns a dictionary of metrics.
    """
    metrics = {}
    try:
        with open(report_path, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            
            # Find the line corresponding to class '1' (positive class)
            for line in lines:
                if line.strip().startswith('1 '): # Ensure it's for class 1 and not part of '10', '11', etc.
                    parts = line.split()
                    if len(parts) >= 4: # Expected: '1', 'precision', 'recall', 'f1-score', 'support'
                        metrics['precision'] = float(parts[1])
                        metrics['recall'] = float(parts[2])
                        break
    except (FileNotFoundError, IndexError, ValueError) as e:
        logger.error(f"Error parsing classification report from {report_path}: {e}")
    return metrics

def main():

    st.set_page_config(layout="wide", page_title="Sentilyze", page_icon="📈")



    # --- Initialize Session State for Backtest ---

    if 'portfolio' not in st.session_state:

        st.session_state.portfolio = None

    if 'metrics' not in st.session_state:

        st.session_state.metrics = None

    if 'heatmap_fig' not in st.session_state:

        st.session_state.heatmap_fig = None



    # --- Sidebar ---

    st.sidebar.title("How it Works")

    st.sidebar.info(

        """

        Sentilyze predicts next-day stock momentum by combining financial news sentiment with technical analysis.

        

        1.  **Data Ingestion**: Fetches historical price data from `yfinance` and news headlines from `NewsAPI.org`.

        2.  **Sentiment Analysis**: Uses a pre-trained FinBERT model to analyze the sentiment of each news headline.

        3.  **Feature Engineering**: Calculates a rich set of features, including sentiment scores and technical indicators (e.g., RSI, MACD).

        4.  **Prediction**: An `XGBClassifier` model, trained on this combined data, predicts the momentum for the next trading day.

        """

    )



    st.title("📈 Sentilyze")

    st.write("A sentiment-driven stock momentum predictor.")



    # --- Model and Tokenizer Loading ---

    sentiment_analyzer = load_sentiment_analyzer()

    universal_model = load_universal_model_cached()



    # --- Main App ---

    ticker = st.text_input("Enter a stock ticker:", "NVDA")

    model_path = f"models/{ticker}_model.joblib"
    specialist_model = load_model(model_path) if os.path.exists(model_path) else None

    if not specialist_model and not universal_model:
        st.warning(
            "No models available. Please train a specialist model (e.g., `python train.py --ticker NVDA`) or a universal model (`python train_universal.py`)."
        )
        st.stop()



    # --- TABS ---

    tab1, tab2, tab3, tab4 = st.tabs(["Prediction Analysis", "Results Dashboard", "Backtest Analysis", "Advanced Model Analysis"])



    with tab1:

        st.header(f"Analyze Model Predictions for {ticker}")

        if st.button("Run Analysis"):

            try:

                with st.spinner(f"Fetching latest data and making prediction for {ticker}..."):

                    # 1. Fetch and prepare data
                    features_df, price_history_with_indicators, news_with_sentiment_df = preprocess_data(ticker)

                    # 2. Initialize prediction variables

                    final_prediction = None

                    final_confidence = 0.0

                    prediction_source = ""



                    # 3. Get latest data for models

                    specialist_model = load_model(model_path) if os.path.exists(model_path) else None

                    sequence_length = 30

                    feature_columns = [col for col in features_df.columns if col not in ["target"]]

                    latest_sequence = features_df[feature_columns].tail(sequence_length).values



                    # --- Hybrid Prediction Logic ---

                    if specialist_model and universal_model and latest_sequence.shape[0] == sequence_length:

                        prediction_source = "Hybrid (Specialist + Universal)"

                        spec_latest_features = features_df.iloc[-1:][FEATURES]

                        spec_pred, spec_conf = get_prediction_on_latest_data(specialist_model, spec_latest_features, FEATURES)

                        spec_prob = spec_conf[0][spec_pred[0]]

                        uni_pred, uni_conf = get_universal_prediction_on_latest_data(universal_model, latest_sequence)

                        final_prob = (spec_prob * 0.7) + (uni_conf * 0.3)

                        final_prediction = 1 if final_prob >= 0.5 else 0

                        final_confidence = final_prob if final_prediction == 1 else 1 - final_prob

                    elif specialist_model:

                        prediction_source = "Specialist"

                        spec_latest_features = features_df.iloc[-1:][FEATURES]

                        spec_pred, spec_conf = get_prediction_on_latest_data(specialist_model, spec_latest_features, FEATURES)

                        final_prediction = spec_pred[0]

                        final_confidence = spec_conf[0][final_prediction]

                    elif universal_model and latest_sequence.shape[0] == sequence_length:

                        prediction_source = "Universal"

                        st.write(f"Number of features in latest_sequence: {latest_sequence.shape[-1]}")

                        uni_pred, uni_conf = get_universal_prediction_on_latest_data(universal_model, latest_sequence)

                        final_prediction = uni_pred

                        final_confidence = uni_conf

                    else:

                        st.error("Could not make a prediction. A trained model is not available or there is not enough data.")

                        st.stop()



                    # 5. Display result

                    prediction_label = "Positive" if final_prediction == 1 else "Negative"

                    display_prediction_results(

                        prediction_label, prediction_source, final_confidence, price_history_with_indicators, news_with_sentiment_df

                    )



            except requests.exceptions.RequestException as e:

                logger.error(f"A network error occurred during prediction: {e}")

                st.error("A network error occurred. Please check your internet connection and NewsAPI key.")

            except Exception as e:

                logger.error(f"An unexpected error occurred during prediction: {e}")

                st.error(f"An unexpected error occurred: {e}. Please check the logs for more details.")



    with tab2:

        st.header(f"Results Dashboard for {ticker}")



        mlflow_data = get_latest_mlflow_run_data(ticker)



        if mlflow_data:

            metrics = mlflow_data["metrics"]

            artifact_uri = mlflow_data["artifact_uri"]



            try:

                local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)



                st.subheader("Key Performance Metrics")

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Accuracy", f"{metrics.get('accuracy', 0.0):.2%}")

                

                classification_report_path = os.path.join(local_path, f"{ticker}_classification_report.txt")

                if os.path.exists(classification_report_path):

                    class_metrics = parse_classification_report(classification_report_path)

                    col2.metric("Precision", f"{class_metrics.get('precision', 0.0):.2%}")

                    col3.metric("Recall", f"{class_metrics.get('recall', 0.0):.2%}")

                else:

                    col2.metric("Precision", "N/A")

                    col3.metric("Recall", "N/A")



                col4.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0.0):.2f}")



                st.subheader("Backtest Performance: Strategy vs. Buy & Hold")

                portfolio_path = os.path.join(local_path, f"{ticker}_portfolio.csv")

                if os.path.exists(portfolio_path):

                    portfolio = pd.read_csv(portfolio_path, index_col=0, parse_dates=True)

                    st.line_chart(portfolio[["total", "benchmark"]].rename(columns={"total": "Strategy", "benchmark": "Buy & Hold"}))

                else:

                    st.warning("Portfolio artifact not found.")



                st.subheader("Monthly Returns Heatmap")

                heatmap_path = os.path.join(local_path, f"{ticker}_monthly_returns_heatmap.png")

                if os.path.exists(heatmap_path):

                    st.image(heatmap_path)

                else:

                    st.warning("Monthly returns heatmap artifact not found.")



                st.subheader("Feature Importance")

                feature_importances_path = os.path.join(local_path, f"{ticker}_feature_importances.csv")

                if os.path.exists(feature_importances_path):

                    feature_importances = pd.read_csv(feature_importances_path)

                    st.bar_chart(feature_importances.set_index('feature'))

                else:

                    st.warning("Feature importances artifact not found.")



            except Exception as e:

                st.error(f"Error loading MLflow data: {e}")

                logger.error(f"Error loading MLflow data: {e}")

        else:

            st.warning(f"No MLflow run data found for {ticker}. Please ensure a model has been trained and logged with MLflow.")



    with tab3:

        st.header(f"Run Backtest Analysis for {ticker}")

        st.write("This will simulate the trading strategy over the last 5 years of historical data.")



        # Backtest configuration

        col1, col2 = st.columns(2)

        with col1:

            initial_capital = st.number_input("Initial Capital", value=10000.0, step=1000.0)

        with col2:

            transaction_cost_pct = (st.number_input("Transaction Cost (%)", value=0.1, step=0.05, format="%.3f") / 100.0)



        if st.button("Run Backtest"):

            try:

                with st.spinner(f"Running 5-year backtest for {ticker}... This may take a moment."):
                    features_df_backtest, _, _ = preprocess_data(ticker, period="5y")
                    features_df_backtest = features_df_backtest.dropna()

                    if not specialist_model:
                        st.error(f"A specialist model for {ticker} is required to run a backtest.")
                        st.stop()
                    predictions = specialist_model.predict(features_df_backtest[FEATURES])

                    signals = pd.Series(predictions, index=features_df_backtest.index).replace({0: -1})

                    st.session_state.portfolio, st.session_state.metrics, st.session_state.heatmap_fig = run_backtest(
                        price_history=features_df_backtest,
                        signals=signals,
                        initial_capital=initial_capital,
                        transaction_cost_pct=transaction_cost_pct,
                    )



            except requests.exceptions.RequestException as e:

                logger.error(f"A network error occurred during backtest: {e}")

                st.error("A network error occurred. Please check your internet connection and NewsAPI key.")

            except Exception as e:

                logger.error(f"An unexpected error occurred during backtest: {e}")

                st.error(f"An unexpected error occurred: {e}. Please check the logs for more details.")



        # --- Display Backtest Results from Session State ---

        if st.session_state.portfolio is not None:

            st.subheader("Backtest Performance Metrics")

            row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)

            row1_col1.metric("Strategy Return", f"{st.session_state.metrics['strategy_total_return']:.2%}")

            row1_col2.metric("Buy & Hold Return", f"{st.session_state.metrics['buy_and_hold_total_return']:.2%}")

            row1_col3.metric("Sharpe Ratio", f"{st.session_state.metrics['sharpe_ratio']:.2f}")

            row1_col4.metric("Sortino Ratio", f"{st.session_state.metrics['sortino_ratio']:.2f}")



            row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)

            row2_col2.metric("Win Rate", f"{st.session_state.metrics['win_rate']:.2%}")

            row2_col1.metric("Total Trades", st.session_state.metrics['total_trades'])

            row2_col3.metric("Strategy Max Drawdown", f"{st.session_state.metrics['strategy_max_drawdown']:.2%}")

            row2_col4.metric("Buy & Hold Max Drawdown", f"{st.session_state.metrics['buy_and_hold_max_drawdown']:.2%}")



            st.subheader("Portfolio Value Over Time")

            chart_data = st.session_state.portfolio[["total", "benchmark"]].rename(columns={"total": "Strategy", "benchmark": "Buy & Hold"})

            st.line_chart(chart_data)



            st.subheader("Monthly Returns Heatmap")

            st.pyplot(st.session_state.heatmap_fig)



    with tab4:

        st.header(f"Advanced Model Analysis for {ticker}")



        mlflow_data = get_latest_mlflow_run_data(ticker)



        if mlflow_data:

            artifact_uri = mlflow_data["artifact_uri"]



            try:

                local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)



                st.subheader("Explainable AI (XAI) - SHAP Analysis")

                # Load SHAP values and X_test

                shap_values_path = os.path.join(local_path, f"{ticker}_shap_values.npy")

                X_test_path = os.path.join(local_path, f"{ticker}_X_test.csv")

                if os.path.exists(shap_values_path) and os.path.exists(X_test_path):

                    shap_values = np.load(shap_values_path)

                    X_test = pd.read_csv(X_test_path, index_col=0, parse_dates=True)

                    st.write("SHAP Summary Plot")

                    shap.summary_plot(shap_values, X_test, show=False)

                    st.pyplot(plt.gcf())

                else:

                    st.warning("SHAP values or X_test artifact not found.")



                with st.expander("Detailed Classification Report"):

                    classification_report_path = os.path.join(local_path, f"{ticker}_classification_report.txt")

                    if os.path.exists(classification_report_path):

                        with open(classification_report_path, 'r') as f:

                            st.text(f.read())

                    else:

                        st.warning("Classification report artifact not found.")



            except Exception as e:

                st.error(f"Error loading MLflow data: {e}")

                logger.error(f"Error loading MLflow data: {e}")



        else:

            st.warning(f"No MLflow run data found for {ticker}. Please ensure a model has been trained and logged with MLflow.")





if __name__ == "__main__":

    main()



    
