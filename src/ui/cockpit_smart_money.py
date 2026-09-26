"""
Cockpit 2: Smart Money & Alternative Data
=========================================
Consolidates:
- Tab 1: 📡 Real-Time Market Anomaly Screener (OpenBB Volume Inflow & Gaps)
- Tab 2: 📉 3D Volatility Surface & Dark Pool Heatmaps
- Tab 3: 📰 Alternative Data & FinBERT Sentiment Radar
- Tab 4: 🏛️ Smart-Money Executive & Insider Radar
"""

import streamlit as st
from src.ui.ws_screener import render_screener_workspace
from src.ui.ws_options_surface import render_options_surface_workspace
from src.ui.ws_alternative_data import render_alternative_data_workspace
from src.ui.ws_insider_radar import render_insider_radar_workspace


def render_smart_money_cockpit(selected_ticker: str):
    st.markdown(
        """
        <div style="margin-bottom: 16px;">
            <h1 style="margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -0.02em;">
                🌊 Smart Money & Alternative Data Radar
            </h1>
            <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.95rem;">
                Institutional dark pool prints, unusual options flow, insider filings, and real-time sentiment anomalies.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = [
        "📡 Market Anomaly Screener",
        "📉 Dark Pools & Options Surface",
        "📰 News & Alternative Sentiment",
        "🏛️ Insider Transactions Radar",
    ]

    selected_tab = (
        st.segmented_control(
            "Smart Money Navigation",
            options=tabs,
            default=tabs[0],
            label_visibility="collapsed",
            key=f"smart_money_subnav_{selected_ticker}",
        )
        or tabs[0]
    )

    st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

    if selected_tab == "📡 Market Anomaly Screener":
        render_screener_workspace()
    elif selected_tab == "📉 Dark Pools & Options Surface":
        render_options_surface_workspace(selected_ticker)
    elif selected_tab == "📰 News & Alternative Sentiment":
        render_alternative_data_workspace(selected_ticker)
    elif selected_tab == "🏛️ Insider Transactions Radar":
        render_insider_radar_workspace(selected_ticker)
