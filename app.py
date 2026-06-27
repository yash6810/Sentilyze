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
from typing import Any, Dict, List
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from src.preprocessing import preprocess_data
from src.modeling import load_model, get_prediction_on_latest_data
from src.utils import get_logger
from src.backtesting import run_backtest
from src.config import FEATURES

logger = get_logger(__name__)

# --- Supported Tickers ---
SUPPORTED_TICKERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "TSLA", "AMZN"]


def inject_custom_css():
    """Inject premium financial dashboard CSS."""
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* --- Global --- */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    /* --- Header Banner --- */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid rgba(0, 212, 170, 0.15);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00D4AA, #7C3AED, #00D4AA);
        background-size: 200% 100%;
        animation: shimmer 3s ease-in-out infinite;
    }
    @keyframes shimmer {
        0%, 100% { background-position: 200% 0; }
        50% { background-position: -200% 0; }
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00D4AA, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1rem;
        font-weight: 400;
        margin-top: 0.3rem;
    }

    /* --- Metric Cards --- */
    .metric-card {
        background: linear-gradient(145deg, #111827, #1a2332);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(0, 212, 170, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 212, 170, 0.08);
    }
    .metric-label {
        font-size: 0.75rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #E2E8F0;
    }
    .metric-value.positive { color: #00D4AA; }
    .metric-value.negative { color: #EF4444; }
    .metric-value.neutral { color: #F59E0B; }

    /* --- Signal Badge --- */
    .signal-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.8rem 1.8rem;
        border-radius: 50px;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .signal-buy {
        background: rgba(0, 212, 170, 0.12);
        border: 2px solid #00D4AA;
        color: #00D4AA;
    }
    .signal-sell {
        background: rgba(239, 68, 68, 0.12);
        border: 2px solid #EF4444;
        color: #EF4444;
    }
    .signal-hold {
        background: rgba(245, 158, 11, 0.12);
        border: 2px solid #F59E0B;
        color: #F59E0B;
    }

    /* --- Section Headers --- */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #CBD5E1;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        letter-spacing: 0.3px;
    }

    /* --- Sidebar --- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A0E17 0%, #111827 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

    /* --- Tabs --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #111827;
        border-radius: 10px;
        padding: 4px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        font-size: 0.85rem;
        color: #64748B;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0, 212, 170, 0.1) !important;
        color: #00D4AA !important;
    }

    /* --- Divider --- */
    .subtle-divider {
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        margin: 1.5rem 0;
    }

    /* --- Hide default Streamlit elements --- */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, style: str = ""):
    """Render a single styled metric card."""
    css_class = ""
    if style == "positive":
        css_class = "positive"
    elif style == "negative":
        css_class = "negative"
    elif style == "neutral":
        css_class = "neutral"

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {css_class}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_badge(signal: str, confidence: float):
    """Render a trading signal badge."""
    if signal == "BUY":
        badge_class = "signal-buy"
        icon = "▲"
    elif signal == "SELL":
        badge_class = "signal-sell"
        icon = "▼"
    else:
        badge_class = "signal-hold"
        icon = "●"

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 1.5rem; margin: 1rem 0;">
            <div class="signal-badge {badge_class}">{icon} {signal}</div>
            <div style="color: #94A3B8; font-size: 0.9rem;">
                Confidence: <span style="color: #E2E8F0; font-weight: 600;">{confidence:.1%}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Data Loading ---


def get_historical_results(ticker: str) -> Dict[str, Any] | None:
    """Retrieves pre-computed result data from the results directory."""
    results_dir = "results"
    metrics_path = os.path.join(results_dir, f"{ticker}_metrics.json")

    if not os.path.exists(metrics_path):
        return None

    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        return {
            "metrics": metrics,
            "results_dir": results_dir,
            "portfolio_path": os.path.join(results_dir, f"{ticker}_portfolio.csv"),
            "heatmap_path": os.path.join(
                results_dir, f"{ticker}_monthly_returns_heatmap.png"
            ),
            "importances_path": os.path.join(
                results_dir, f"{ticker}_feature_importances.csv"
            ),
            "report_path": os.path.join(
                results_dir, f"{ticker}_classification_report.txt"
            ),
            "shap_path": os.path.join(results_dir, f"{ticker}_shap_values.npy"),
            "xtest_path": os.path.join(results_dir, f"{ticker}_X_test.csv"),
        }
    except Exception as e:
        logger.error(f"Error reading metrics for {ticker}: {e}")
        return None


@st.cache_resource
def load_sentiment_analyzer() -> Any:
    """Loads the FinBERT sentiment analysis model (cached)."""
    logger.info("Loading FinBERT sentiment analysis model...")
    tokenizer = AutoTokenizer.from_pretrained("./models/finbert-fine-tuned")
    model = AutoModelForSequenceClassification.from_pretrained(
        "./models/finbert-fine-tuned"
    )
    task_name: Any = "sentiment-analysis"
    return pipeline(task_name, model=model, tokenizer=tokenizer)


def parse_classification_report(report_path: str) -> Dict[str, float]:
    """Parses a classification report text file for precision and recall."""
    metrics = {}
    try:
        with open(report_path, "r") as f:
            for line in f:
                if line.strip().startswith("1 "):
                    parts = line.split()
                    if len(parts) >= 4:
                        metrics["precision"] = float(parts[1])
                        metrics["recall"] = float(parts[2])
                    break
    except (FileNotFoundError, IndexError, ValueError) as e:
        logger.error(f"Error parsing report from {report_path}: {e}")
    return metrics


# --- Tab Renderers ---


def render_prediction_tab(ticker: str, model_path: str):
    """Tab 1: Live Prediction Analysis."""
    st.markdown(
        '<div class="section-header">📡 Live Signal Generation</div>',
        unsafe_allow_html=True,
    )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button("⚡ Generate Signal", use_container_width=True)
    with col_info:
        st.caption("Fetches latest news & prices, runs FinBERT + XGBoost pipeline")

    if not run_clicked:
        st.info(
            "Click **⚡ Generate Signal** to fetch the latest data and produce a "
            "real-time momentum prediction for this ticker."
        )
        return

    try:
        with st.spinner("Fetching data & running inference pipeline..."):
            features_df, price_hist, news_df = preprocess_data(ticker)
            specialist_model = (
                load_model(model_path) if os.path.exists(model_path) else None
            )

            if not specialist_model:
                st.error("Model unavailable.")
                return

            spec_features = features_df.iloc[-1:][FEATURES]
            pred, conf = get_prediction_on_latest_data(
                specialist_model, spec_features, FEATURES
            )

            raw_pred = pred[0]
            confidence = conf[0][1]
            rsi = price_hist["rsi"].iloc[-1]

            # Regime filter logic
            if confidence > 0.80 and rsi < 70:
                signal, final_pred = "BUY", 1
            elif confidence > 0.50 and rsi < 70:
                signal, final_pred = "BUY", 1
            elif confidence <= 0.50:
                signal, final_pred = "SELL", 0
            else:
                signal, final_pred = "HOLD", 0

        # --- Display Signal ---
        st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)
        render_signal_badge(signal, confidence)

        # Key indicators row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric_card("Close Price", f"${price_hist['Close'].iloc[-1]:.2f}")
        with c2:
            render_metric_card(
                "RSI (14)",
                f"{rsi:.1f}",
                "negative" if rsi >= 70 else ("positive" if rsi <= 30 else ""),
            )
        with c3:
            render_metric_card(
                "P(Up)",
                f"{confidence:.1%}",
                "positive" if confidence > 0.5 else "negative",
            )
        with c4:
            sma = price_hist["sma200"].iloc[-1]
            above = price_hist["Close"].iloc[-1] > sma
            render_metric_card(
                "Trend (SMA200)",
                "▲ Bullish" if above else "▼ Bearish",
                "positive" if above else "negative",
            )

        # SHAP explanation
        st.markdown(
            '<div class="section-header">🧠 Decision Explanation (SHAP)</div>',
            unsafe_allow_html=True,
        )
        with st.spinner("Computing SHAP values..."):
            try:
                explainer = shap.TreeExplainer(specialist_model)
                shap_vals = explainer.shap_values(spec_features)
                st_shap(
                    shap.force_plot(explainer.expected_value, shap_vals, spec_features)
                )
            except Exception as e:
                st.warning(f"SHAP unavailable: {e}")

        # Raw data
        with st.expander("📊 View Raw Data"):
            st.write("**Price History (last 5 days)**")
            st.dataframe(price_hist.tail(), use_container_width=True)
            st.write("**News Headlines with Sentiment**")
            st.dataframe(news_df.head(10), use_container_width=True)

    except requests.exceptions.RequestException as e:
        st.error("⚠️ Network error. Check your connection and NewsAPI key.")
        logger.error(f"Network error: {e}")
    except Exception as e:
        st.error(f"⚠️ Pipeline error: {e}")
        logger.error(f"Prediction error: {e}")


def render_dashboard_tab(ticker: str):
    """Tab 2: Results Dashboard."""
    results = get_historical_results(ticker)
    if not results:
        st.warning(
            f"No pre-computed results for **{ticker}**. "
            f"Run `python train.py --ticker {ticker}` locally."
        )
        return

    metrics = results["metrics"]

    # --- KPI Row ---
    st.markdown(
        '<div class="section-header">📊 Model Performance</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    rpt = results["report_path"]
    cm = parse_classification_report(rpt) if os.path.exists(rpt) else {}
    with c1:
        render_metric_card("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
    with c2:
        if cm:
            render_metric_card("Precision", f"{cm.get('precision', 0):.1%}")
        else:
            render_metric_card("Precision", "N/A")
    with c3:
        if cm:
            render_metric_card("Recall", f"{cm.get('recall', 0):.1%}")
        else:
            render_metric_card("Recall", "N/A")
    with c4:
        render_metric_card(
            "Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}", "positive"
        )

    # --- Equity Curve ---
    st.markdown(
        '<div class="section-header">📈 Strategy vs Buy & Hold</div>',
        unsafe_allow_html=True,
    )
    ppath = results["portfolio_path"]
    if os.path.exists(ppath):
        portfolio = pd.read_csv(ppath, index_col=0, parse_dates=True)
        portfolio_subset = pd.DataFrame(portfolio[["total", "benchmark"]])
        st.line_chart(
            portfolio_subset.rename(
                columns={"total": "Strategy", "benchmark": "Buy & Hold"}
            )
        )

    else:
        st.warning("Portfolio data not found.")

    # --- Feature Importance & Heatmap ---
    left, right = st.columns(2)
    with left:
        st.markdown(
            '<div class="section-header">🏆 Feature Importance</div>',
            unsafe_allow_html=True,
        )
        ipath = results["importances_path"]
        if os.path.exists(ipath):
            fi = pd.read_csv(ipath)
            st.bar_chart(fi.set_index("feature"))
        else:
            st.warning("Feature importances not found.")

    with right:
        st.markdown(
            '<div class="section-header">🗓️ Monthly Returns</div>',
            unsafe_allow_html=True,
        )
        hpath = results["heatmap_path"]
        if os.path.exists(hpath):
            st.image(hpath, use_container_width=True)
        else:
            st.warning("Heatmap not found.")


def render_backtest_tab(ticker: str):
    """Tab 3: Backtest Analysis."""
    results = get_historical_results(ticker)
    if not results:
        st.warning(f"No backtest data for **{ticker}**.")
        return

    metrics = results["metrics"]

    st.markdown(
        '<div class="section-header">⚡ Performance Summary</div>',
        unsafe_allow_html=True,
    )

    # Row 1: Returns
    c1, c2, c3, c4 = st.columns(4)
    strat_ret = metrics.get("strategy_total_return", 0)
    bh_ret = metrics.get("buy_and_hold_total_return", 0)
    with c1:
        render_metric_card(
            "Strategy Return",
            f"{strat_ret:.1%}",
            "positive" if strat_ret > 0 else "negative",
        )
    with c2:
        render_metric_card(
            "Buy & Hold",
            f"{bh_ret:.1%}",
            "positive" if bh_ret > 0 else "negative",
        )
    with c3:
        render_metric_card(
            "Sharpe Ratio",
            f"{metrics.get('sharpe_ratio', 0):.2f}",
            "positive" if metrics.get("sharpe_ratio", 0) > 1 else "",
        )
    with c4:
        render_metric_card(
            "Sortino Ratio",
            f"{metrics.get('sortino_ratio', 0):.2f}",
            "positive" if metrics.get("sortino_ratio", 0) > 1 else "",
        )

    # Row 2: Risk
    st.markdown(
        '<div class="section-header">🛡️ Risk Metrics</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Total Trades", str(metrics.get("total_trades", 0)))
    with c2:
        wr = metrics.get("win_rate", 0)
        render_metric_card(
            "Win Rate", f"{wr:.1%}", "positive" if wr > 0.5 else "negative"
        )
    with c3:
        sdd = metrics.get("strategy_max_drawdown", 0)
        render_metric_card("Strategy Drawdown", f"{sdd:.1%}", "negative")
    with c4:
        bhdd = metrics.get("buy_and_hold_max_drawdown", 0)
        render_metric_card("B&H Drawdown", f"{bhdd:.1%}", "negative")

    # Alpha badge
    alpha = strat_ret - bh_ret
    if alpha > 0:
        st.success(f"🏆 Strategy generated **{alpha:.1%}** alpha over Buy & Hold.")
    else:
        st.warning(f"Strategy underperformed Buy & Hold by **{abs(alpha):.1%}**.")

    # Equity curve
    st.markdown(
        '<div class="section-header">📈 Portfolio Value</div>',
        unsafe_allow_html=True,
    )
    ppath = results["portfolio_path"]
    if os.path.exists(ppath):
        portfolio = pd.read_csv(ppath, index_col=0, parse_dates=True)
        portfolio_subset = pd.DataFrame(portfolio[["total", "benchmark"]])
        st.line_chart(
            portfolio_subset.rename(
                columns={"total": "Strategy", "benchmark": "Buy & Hold"}
            )
        )

    hpath = results["heatmap_path"]
    if os.path.exists(hpath):
        st.markdown(
            '<div class="section-header">🗓️ Monthly Returns</div>',
            unsafe_allow_html=True,
        )
        st.image(hpath, use_container_width=True)


def render_xai_tab(ticker: str):
    """Tab 4: Explainable AI."""
    results = get_historical_results(ticker)
    if not results:
        st.warning(f"No analysis data for **{ticker}**.")
        return

    st.markdown(
        '<div class="section-header">🔬 SHAP Feature Analysis</div>',
        unsafe_allow_html=True,
    )

    spath = results["shap_path"]
    xpath = results["xtest_path"]

    if os.path.exists(spath) and os.path.exists(xpath):
        shap_values = np.load(spath)
        X_test = pd.read_csv(xpath, index_col=0, parse_dates=True)

        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("SHAP data files not found.")

    with st.expander("📋 Full Classification Report"):
        rpath = results["report_path"]
        if os.path.exists(rpath):
            with open(rpath, "r") as f:
                st.code(f.read(), language="text")
        else:
            st.warning("Report not found.")


# --- Main App ---


def main():
    st.set_page_config(
        layout="wide",
        page_title="Sentilyze | AI Trading Intelligence",
        page_icon="📈",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    # --- Sidebar ---
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding: 1rem 0;">
                <div style="font-size: 2.5rem;">📈</div>
                <div style="font-size: 1.3rem; font-weight: 700;
                    background: linear-gradient(135deg, #00D4AA, #7C3AED);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    Sentilyze
                </div>
                <div style="color: #64748B; font-size: 0.75rem; margin-top: 0.3rem;">
                    AI Trading Intelligence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        ticker = st.selectbox(
            "Select Ticker",
            SUPPORTED_TICKERS,
            index=0,
            help="Choose from pre-trained models",
        )

        st.markdown("---")

        with st.expander("ℹ️ How it Works", expanded=False):
            st.markdown(
                """
                **1. Data Ingestion**
                Fetches price data (yfinance) + news (NewsAPI)

                **2. Sentiment Analysis**
                FinBERT classifies each headline

                **3. Feature Engineering**
                RSI, MACD, Bollinger Bands, VIX + sentiment scores

                **4. Prediction**
                XGBoost with Walk-Forward Optimization
                """
            )

        st.markdown("---")
        st.caption("Built with Streamlit • XGBoost • FinBERT")

    # --- Hero Banner ---
    st.markdown(
        f"""
        <div class="hero-banner">
            <p class="hero-title">Sentilyze</p>
            <p class="hero-subtitle">
                AI-powered momentum prediction combining NLP sentiment with technical analysis
                &nbsp;·&nbsp; Analyzing <b>{ticker}</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Load Model ---
    model_path = f"models/{ticker}_model.json"
    model_exists = os.path.exists(model_path) or os.path.exists(
        model_path.replace(".json", ".joblib")
    )

    if not model_exists:
        st.error(
            f"⚠️ No trained model for **{ticker}**. "
            f"Run: `python train.py --ticker {ticker}`"
        )
        st.stop()

    # Load sentiment analyzer in background
    _ = load_sentiment_analyzer()

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(
        ["⚡ Live Signal", "📊 Dashboard", "🏦 Backtest", "🧠 XAI"]
    )

    with tab1:
        render_prediction_tab(ticker, model_path)
    with tab2:
        render_dashboard_tab(ticker)
    with tab3:
        render_backtest_tab(ticker)
    with tab4:
        render_xai_tab(ticker)


if __name__ == "__main__":
    main()
