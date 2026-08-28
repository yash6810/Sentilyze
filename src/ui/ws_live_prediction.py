"""
Workspace 1: Live Directional Predictions & Fast Real-Time Inference.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.ui.components import render_workspace_header, render_conviction_gauge
from src.data_ingestion import get_stock_data, get_news
from src.feature_engineering import generate_features
from src.modeling import load_model, make_prediction


def render_live_prediction_workspace(ticker: str):
    """Renders the high-speed live inference and directional prediction workspace."""
    render_workspace_header(
        title=f"🔮 {ticker} Live Algorithmic Signal & Inference",
        subtitle="Walk-Forward Optimized XGBoost Momentum Classifier & Real-Time FinBERT News Sentiment",
        badge_text="SUB-SECOND INFERENCE",
        badge_color="#10B981",
    )

    with st.spinner(f"Running high-speed feature inference for {ticker}..."):
        try:
            # 1. Fetch live market price data & news
            df_price = get_stock_data(ticker, period="1y", use_cache=True)
            df_news = get_news(ticker, use_cache=True)

            if df_price.empty:
                st.warning(f"No price data available for {ticker}.")
                return

            # 2. Build feature vector
            feat_df = generate_features(df_price, df_news, ticker=ticker)
            if feat_df.empty:
                st.warning(f"Insufficient feature history for {ticker}.")
                return

            latest_row = feat_df.tail(1)
            current_price = float(df_price["Close"].iloc[-1])
            prev_price = (
                float(df_price["Close"].iloc[-2])
                if len(df_price) > 1
                else current_price
            )
            price_chg = ((current_price - prev_price) / prev_price) * 100.0

            # 3. Load Walk-Forward Model & Predict
            model_path = os.path.join("models", f"{ticker}_model.json")
            if not os.path.exists(model_path):
                model_path = os.path.join("models", "NVDA_model.json")

            model = load_model(model_path)
            pred_class, pred_prob = make_prediction(model, latest_row)

            # Sizing & ATR Stops
            atr_val = (
                float(latest_row.get("atr_14", current_price * 0.025).iloc[0])
                if "atr_14" in latest_row
                else current_price * 0.025
            )
            tp1 = current_price + (2.5 * atr_val)
            tp2 = current_price + (4.5 * atr_val)
            stop_loss = current_price - (1.5 * atr_val)

        except Exception as e:
            st.error(f"Inference error for {ticker}: {e}")
            return

    # Top KPI Metrics Bar
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "💵 Spot Market Price", f"${current_price:.2f}", delta=f"{price_chg:+.2f}%"
    )
    col2.metric(
        "🎯 Directional Signal",
        "🟢 BUY (LONG)" if pred_class == 1 else "🔴 SELL / CASH",
        delta=f"{pred_prob*100:.1f}% Confidence",
    )
    col3.metric(
        "🎯 Target 1 (+2.5 ATR)",
        f"${tp1:.2f}",
        delta=f"+{((tp1-current_price)/current_price)*100:.1f}%",
    )
    col4.metric(
        "🛡️ ATR Stop Floor",
        f"${stop_loss:.2f}",
        delta=f"{((stop_loss-current_price)/current_price)*100:.1f}%",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    render_conviction_gauge(
        pred_prob * 100.0, label=f"QUANTITATIVE ALPHA CONVICTION ({ticker})"
    )

    # Interactive Price & ATR Chart
    st.markdown("### 📈 Interactive Price Action & Target Bands")
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df_price.index[-90:],
            open=df_price["Open"].iloc[-90:],
            high=df_price["High"].iloc[-90:],
            low=df_price["Low"].iloc[-90:],
            close=df_price["Close"].iloc[-90:],
            name="Price Action",
        )
    )
    fig.add_hline(
        y=tp1,
        line_dash="dash",
        line_color="#10B981",
        annotation_text=f"TP1: ${tp1:.2f}",
    )
    fig.add_hline(
        y=tp2,
        line_dash="dash",
        line_color="#8B5CF6",
        annotation_text=f"TP2: ${tp2:.2f}",
    )
    fig.add_hline(
        y=stop_loss,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"SL: ${stop_loss:.2f}",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Live News Feed
    if not df_news.empty:
        st.markdown("### 📰 Ingested Real-Time News Stream")
        for _, n in df_news.head(4).iterrows():
            title = n.get("title", "")
            src = n.get("source", "Financial Wire")
            url = n.get("url", "#")
            st.markdown(
                f"""
                <div class="glass-card" style="padding: 12px 18px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <a href="{url}" target="_blank" style="color: #F3F4F6; text-decoration: none; font-weight: 600;">{title}</a>
                        <span style="color: #64748B; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace;">{src}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
