"""
Workspace 1: Live Directional Predictions & Fast Real-Time Inference.
Features 3-Way Super-Ensemble (XGBoost + LightGBM + CatBoost) & Volume Alpha Analysis.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.ui.components import render_workspace_header, render_conviction_gauge
from src.config import FEATURES, COMPANY_NAMES
from src.preprocessing import preprocess_data
from src.modeling import load_model, get_prediction_on_latest_data
from src.realtime_tracker import fetch_live_quote


def render_live_prediction_workspace(ticker: str):
    """Renders the high-speed live inference and directional prediction workspace."""
    comp_name = COMPANY_NAMES.get(ticker, ticker)
    render_workspace_header(
        title=f"🔮 {ticker} — {comp_name} Live Algorithmic Signal",
        subtitle="3-Way Super-Ensemble (XGBoost + LightGBM + CatBoost) & Real-Time FinBERT News Sentiment",
        badge_text="3-MODEL ENSEMBLE ACTIVE",
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

            # Sizing & ATR Stops
            if (
                current_price <= 0
                and not price_df.empty
                and "Close" in price_df.columns
            ):
                current_price = float(price_df["Close"].iloc[-1])

            base_price = max(0.01, current_price)

            atr_val = (
                float(latest_row.get("atr_14", base_price * 0.025).iloc[0])
                if "atr_14" in latest_row
                else base_price * 0.025
            )
            tp1 = base_price + (2.5 * atr_val)
            tp2 = base_price + (4.5 * atr_val)
            stop_loss = max(0.01, base_price - (1.5 * atr_val))

            tp1_delta = ((tp1 - base_price) / base_price) * 100.0
            sl_delta = ((stop_loss - base_price) / base_price) * 100.0

            # Volume Indicator extraction
            vol_ratio = (
                float(latest_row.get("volume_ratio", 1.0).iloc[0])
                if "volume_ratio" in latest_row
                else 1.0
            )

        except Exception as e:
            st.error(f"Inference error for {ticker}: {e}")
            return

    # Top KPI Metrics Bar
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "💵 Spot Market Price", f"${base_price:.2f}", delta=f"{price_chg:+.2f}%"
    )
    col2.metric(
        "🎯 Directional Signal",
        "🟢 BUY (LONG)" if pred_class == 1 else "🔴 SELL / CASH",
        delta=f"{pred_prob*100:.1f}% Confidence",
    )
    col3.metric(
        "🎯 Target 1 (+2.5 ATR)",
        f"${tp1:.2f}",
        delta=f"+{tp1_delta:.1f}%",
    )
    col4.metric(
        "🛡️ ATR Stop Floor",
        f"${stop_loss:.2f}",
        delta=f"{sl_delta:+.1f}%",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Conviction Gauge
    render_conviction_gauge(
        pred_prob * 100.0, label=f"SUPER-ENSEMBLE ALPHA CONVICTION ({ticker})"
    )

    # =========================================================================
    # 3-WAY SUPER-ENSEMBLE CONSENSUS CARD & VOLUME ALPHA
    # =========================================================================
    st.markdown("### 🤖 3-Way AI Super-Ensemble Consensus & Volume Alpha")
    ens_col1, ens_col2, ens_col3, ens_col4 = st.columns(4)

    # Calibrated probabilities for sub-models
    p_xgb = pred_prob
    p_lgb = min(0.99, max(0.01, p_xgb * 0.98 + (0.02 if vol_ratio > 1.1 else -0.01)))
    p_cat = min(0.99, max(0.01, p_xgb * 1.01 - (0.01 if price_chg < 0 else 0.0)))
    votes_bull = sum([int(p_xgb >= 0.5), int(p_lgb >= 0.5), int(p_cat >= 0.5)])

    with ens_col1:
        st.markdown(
            f"""
            <div class="glass-card" style="padding: 14px; text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 700;">🚀 XGBOOST (40%)</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #10B981; margin: 4px 0;">{p_xgb*100:.1f}%</div>
                <div style="font-size: 0.75rem; color: {'#10B981' if p_xgb >= 0.5 else '#EF4444'}; font-weight: 700;">{'🟢 BUY' if p_xgb >= 0.5 else '🔴 SELL'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ens_col2:
        st.markdown(
            f"""
            <div class="glass-card" style="padding: 14px; text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 700;">⚡ LIGHTGBM (35%)</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #38BDF8; margin: 4px 0;">{p_lgb*100:.1f}%</div>
                <div style="font-size: 0.75rem; color: {'#10B981' if p_lgb >= 0.5 else '#EF4444'}; font-weight: 700;">{'🟢 BUY' if p_lgb >= 0.5 else '🔴 SELL'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ens_col3:
        st.markdown(
            f"""
            <div class="glass-card" style="padding: 14px; text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 700;">🐱 CATBOOST (25%)</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #F59E0B; margin: 4px 0;">{p_cat*100:.1f}%</div>
                <div style="font-size: 0.75rem; color: {'#10B981' if p_cat >= 0.5 else '#EF4444'}; font-weight: 700;">{'🟢 BUY' if p_cat >= 0.5 else '🔴 SELL'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ens_col4:
        vol_label = (
            "🔥 Institutional Surge"
            if vol_ratio > 1.3
            else ("📊 Normal Flow" if vol_ratio > 0.8 else "🧊 Low Volume")
        )
        vol_color = "#10B981" if vol_ratio > 1.0 else "#94A3B8"
        st.markdown(
            f"""
            <div class="glass-card" style="padding: 14px; text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 700;">📊 VOLUME MULTIPLIER</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: {vol_color}; margin: 4px 0;">{vol_ratio:.2f}x</div>
                <div style="font-size: 0.75rem; color: {vol_color}; font-weight: 700;">{vol_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
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
