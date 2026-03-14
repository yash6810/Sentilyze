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
from src.utils import get_logger
from src.backtesting import run_backtest
from src.config import FEATURES


logger = get_logger(__name__)

def get_historical_results(ticker: str) -> Dict[str, Any] | None:
    """
    Retrieves pre-computed result data (metrics and file paths) from the results directory.
    """
    results_dir = "results"
    metrics_path = os.path.join(results_dir, f"{ticker}_metrics.json")
    
    if not os.path.exists(metrics_path):
        return None

    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        return {
            "metrics": metrics,
            "results_dir": results_dir,
            "portfolio_path": os.path.join(results_dir, f"{ticker}_portfolio.csv"),
            "heatmap_path": os.path.join(results_dir, f"{ticker}_monthly_returns_heatmap.png"),
            "importances_path": os.path.join(results_dir, f"{ticker}_feature_importances.csv"),
            "report_path": os.path.join(results_dir, f"{ticker}_classification_report.txt"),
            "shap_path": os.path.join(results_dir, f"{ticker}_shap_values.npy"),
            "xtest_path": os.path.join(results_dir, f"{ticker}_X_test.csv")
        }
    except Exception as e:
        logger.error(f"Error reading metrics for {ticker}: {e}")
        return None


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



    # --- Main App ---

    ticker = st.text_input("Enter a stock ticker:", "NVDA")

    model_path = f"models/{ticker}_model.json"
    specialist_model = load_model(model_path) if (os.path.exists(model_path) or os.path.exists(model_path.replace('.json', '.joblib'))) else None

    if not specialist_model:
        st.warning(
            f"No trained model found for {ticker}. Please train a model first: `python train.py --ticker {ticker}`"
        )
        st.stop()



    # --- TABS ---

    tab1, tab2, tab3, tab4 = st.tabs(["Prediction Analysis", "Results Dashboard", "Pre-computed Backtest Analysis", "Advanced Model Analysis"])



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
                    if specialist_model:
                        prediction_source = "Specialist"
                        spec_latest_features = features_df.iloc[-1:][FEATURES]
                        spec_pred, spec_conf = get_prediction_on_latest_data(specialist_model, spec_latest_features, FEATURES)
                        
                        raw_prediction = spec_pred[0]
                        final_confidence = spec_conf[0][1] # Probability of 'Up' (class 1)
                        
                        # Apply Regime Filter
                        latest_close = price_history_with_indicators['Close'].iloc[-1]
                        latest_sma200 = price_history_with_indicators['sma200'].iloc[-1]
                        latest_rsi = price_history_with_indicators['rsi'].iloc[-1]
                        
                        regime_blocked = False
                        block_reason = ""
                        
                        if final_confidence > 0.80 and latest_rsi < 70:
                            final_prediction = 1
                            block_reason = "LOCKED: 2.0x Leveraged Buy (Extreme Conviction)"
                        elif final_confidence > 0.50 and latest_rsi < 70:
                            final_prediction = 1 # Confirmed Buy
                        elif final_confidence <= 0.50:
                            final_prediction = 0 # Sell/Cash
                            regime_blocked = True
                            if latest_rsi >= 70:
                                block_reason += f"Overbought (RSI {latest_rsi:.2f} >= 70). "
                            if final_confidence <= 0.50:
                                block_reason += f"Low Confidence (P(Up) {final_confidence:.2%} <= 50%)."
                        else:
                            final_prediction = 0 # Sell/Cash

                    else:
                        st.error("Could not make a prediction. A trained model is not available or there is not enough data.")
                        st.stop()

                    # 5. Display result
                    if regime_blocked:
                        prediction_label = f"Cash (Regime Filter Blocked: {block_reason})"
                    else:
                        prediction_label = "Positive (Buy)" if final_prediction == 1 else "Negative (Sell/Cash)"

                    display_prediction_results(

                        prediction_label, prediction_source, final_confidence, price_history_with_indicators, news_with_sentiment_df

                    )

                    # --- SHAP Explanation for Specialist Model ---
                    if prediction_source in ["Hybrid (Specialist + Universal)", "Specialist"]:
                        st.subheader("Prediction Explanation (SHAP Analysis)")
                        
                        with st.spinner("Calculating SHAP explanation..."):
                            try:
                                explainer = shap.TreeExplainer(specialist_model)
                                
                                # Using spec_latest_features which is already prepared
                                shap_values_latest = explainer.shap_values(spec_latest_features)
                                
                                # Create a SHAP force plot for the latest prediction
                                st_shap(shap.force_plot(explainer.expected_value, shap_values_latest, spec_latest_features))
                                
                            except Exception as e:
                                st.error(f"Could not generate SHAP explanation: {e}")
                                logger.error(f"SHAP explanation generation failed: {e}")




            except requests.exceptions.RequestException as e:

                logger.error(f"A network error occurred during prediction: {e}")

                st.error("A network error occurred. Please check your internet connection and NewsAPI key.")

            except Exception as e:

                logger.error(f"An unexpected error occurred during prediction: {e}")

                st.error(f"An unexpected error occurred: {e}. Please check the logs for more details.")



    with tab2:
        st.header(f"Results Dashboard for {ticker}")

        results_data = get_historical_results(ticker)

        if results_data:
            metrics = results_data["metrics"]

            st.subheader("Key Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics.get('accuracy', 0.0):.2%}")
            
            report_path = results_data["report_path"]
            if os.path.exists(report_path):
                class_metrics = parse_classification_report(report_path)
                col2.metric("Precision", f"{class_metrics.get('precision', 0.0):.2%}")
                col3.metric("Recall", f"{class_metrics.get('recall', 0.0):.2%}")
            else:
                col2.metric("Precision", "N/A")
                col3.metric("Recall", "N/A")

            col4.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0.0):.2f}")

            st.subheader("Backtest Performance: Strategy vs. Buy & Hold")
            portfolio_path = results_data["portfolio_path"]
            if os.path.exists(portfolio_path):
                portfolio = pd.read_csv(portfolio_path, index_col=0, parse_dates=True)
                st.line_chart(portfolio[["total", "benchmark"]].rename(columns={"total": "Strategy", "benchmark": "Buy & Hold"}))
            else:
                st.warning("Portfolio file not found in results directory.")

            st.subheader("Monthly Returns Heatmap")
            heatmap_path = results_data["heatmap_path"]
            if os.path.exists(heatmap_path):
                st.image(heatmap_path)
            else:
                st.warning("Monthly returns heatmap not found in results directory.")

            st.subheader("Feature Importance")
            importances_path = results_data["importances_path"]
            if os.path.exists(importances_path):
                feature_importances = pd.read_csv(importances_path)
                st.bar_chart(feature_importances.set_index('feature'))
            else:
                st.warning("Feature importances not found in results directory.")

        else:
            st.warning(f"No pre-computed results found for {ticker} in the 'results/' directory. Run training locally to generate these files.")



    with tab3:
        st.header(f"Pre-computed Backtest Analysis for {ticker}")
        st.write("This tab displays the results for the full 10-year leveraged simulation.")

        results_data = get_historical_results(ticker)

        if results_data:
            metrics = results_data["metrics"]
            
            st.subheader("Backtest Performance Metrics")
            row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
            row1_col1.metric("Strategy Return", f"{metrics.get('strategy_total_return', 0.0):.2%}")
            row1_col2.metric("Buy & Hold Return", f"{metrics.get('buy_and_hold_total_return', 0.0):.2%}")
            row1_col3.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0.0):.2f}")
            row1_col4.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0.0):.2f}")

            row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
            row2_col2.metric("Win Rate", f"{metrics.get('win_rate', 0.0):.2%}")
            row2_col1.metric("Total Trades", metrics.get('total_trades', 0))
            row2_col3.metric("Strategy Max Drawdown", f"{metrics.get('strategy_max_drawdown', 0.0):.2%}")
            row2_col4.metric("Buy & Hold Max Drawdown", f"{metrics.get('buy_and_hold_max_drawdown', 0.0):.2%}")

            st.subheader("Portfolio Value Over Time")
            portfolio_path = results_data["portfolio_path"]
            if os.path.exists(portfolio_path):
                portfolio = pd.read_csv(portfolio_path, index_col=0, parse_dates=True)
                st.line_chart(portfolio[["total", "benchmark"]].rename(columns={"total": "Strategy", "benchmark": "Buy & Hold"}))
            else:
                st.warning("Portfolio file no found.")

            st.subheader("Monthly Returns Heatmap")
            heatmap_path = results_data["heatmap_path"]
            if os.path.exists(heatmap_path):
                st.image(heatmap_path)
            else:
                st.warning("Heatmap file not found.")
        else:
            st.warning(f"No results data found for {ticker}. Run a training session locally.")



    with tab4:
        st.header(f"Advanced Model Analysis for {ticker}")

        results_data = get_historical_results(ticker)

        if results_data:
            try:
                st.subheader("Explainable AI (XAI) - SHAP Analysis")
                
                shap_values_path = results_data["shap_path"]
                xtest_path = results_data["xtest_path"]

                if os.path.exists(shap_values_path) and os.path.exists(xtest_path):
                    shap_values = np.load(shap_values_path)
                    X_test = pd.read_csv(xtest_path, index_col=0, parse_dates=True)

                    st.write("SHAP Summary Plot")
                    fig = plt.figure()
                    shap.summary_plot(shap_values, X_test, show=False)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning(f"SHAP data files not found for {ticker}.")

                with st.expander("Detailed Classification Report"):
                    report_path = results_data["report_path"]
                    if os.path.exists(report_path):
                        with open(report_path, 'r') as f:
                            st.text(f.read())
                    else:
                        st.warning("Classification report text file not found.")

            except Exception as e:
                st.error(f"Error loading analysis data: {e}")
                logger.error(f"Error loading analysis data: {e}")
        else:
            st.warning(f"No historical analysis data found for {ticker}.")





if __name__ == "__main__":

    main()



    
