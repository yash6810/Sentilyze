"""
Workspace 1: Live Directional Predictions & Fast Real-Time Inference.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.ui.components import render_workspace_header, render_conviction_gauge
from src.config import FEATURES
from src.preprocessing import preprocess_data
from src.modeling import load_model, get_prediction_on_latest_data
from src.realtime_tracker import fetch_live_quote


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
            # 1. Fetch live market spot quote
            quote = fetch_live_quote(ticker)
            current_price = float(quote.get("price", 100.0))
            price_chg = float(quote.get("change_pct", 0.0))

            # 2. Preprocess features and price history
            features_df, price_df, news_df = preprocess_data(
                ticker, period="1y", use_cache=True
            )

            if features_df.empty:
                st.warning(f"Insufficient feature history for {ticker}.")
                return

            latest_row = features_df.tail(1)

            # 3. Load Walk-Forward Model & Predict
            model_path = os.path.join("models", f"{ticker}_model.json")
            if not os.path.exists(model_path):
                model_path = os.path.join("models", "NVDA_model.json")

            model = load_model(model_path)
            prediction, confidence = get_prediction_on_latest_data(
                model, latest_row, FEATURES
            )
            pred_class = int(prediction[0])
            pred_prob = float(confidence[0][1])

            # ATR Take-Profit & Stop-Loss Levels
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
            x=price_df.index[-90:],
            open=price_df["Open"].iloc[-90:],
            high=price_df["High"].iloc[-90:],
            low=price_df["Low"].iloc[-90:],
            close=price_df["Close"].iloc[-90:],
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
    if not news_df.empty:
        st.markdown("### 📰 Ingested Real-Time News Stream")
        for _, n in news_df.head(4).iterrows():
            title = n.get("title", "") or n.get("Title", "")
            src = n.get("source", "Financial Wire")
            if isinstance(src, dict):
                src = src.get("name", "Financial Wire")
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
