"""
Cockpit 2: Smart Money & Alternative Data
=========================================
Consolidates:
- Tab 1: 📡 Real-Time Market Anomaly Screener (Institutional Volume Inflow & Gaps)
- Tab 2: 📉 3D Volatility Surface & Dark Pool Heatmaps
- Tab 3: 📰 Alternative Data & FinBERT Sentiment Radar
- Tab 4: 🏛️ Smart-Money Executive & Insider Radar
"""

import streamlit as st
from src.ui.ws_screener import render_screener_workspace
from src.ui.ws_options_surface import render_options_surface_workspace
from src.ui.ws_alternative_data import render_alternative_data_workspace
from src.ui.ws_insider_radar import render_insider_radar_workspace
from src.ui.ws_biotech_radar import render_biotech_radar_workspace
from src.ui.ws_sec_filings import render_sec_filings_workspace
from src.ui.ws_arxiv_research import render_arxiv_research_workspace


def render_smart_money_cockpit(selected_ticker: str):
    t1, t2, t3, t4, t5, t6, t7 = st.tabs(
        [
            "📡 Market Anomaly Screener",
            "📉 Dark Pools & Options Surface",
            "📰 News & Alternative Sentiment",
            "🏛️ Insider Transactions Radar",
            "🧬 BioTech & FDA Catalyst Radar",
            "📑 SEC EDGAR 8-K Live Radar",
            "🔬 arXiv Quant Alpha Papers",
        ]
    )

    with t1:
        render_screener_workspace()

    with t2:
        render_options_surface_workspace(selected_ticker)

    with t3:
        render_alternative_data_workspace(selected_ticker)

    with t4:
        render_insider_radar_workspace(selected_ticker)

    with t5:
        render_biotech_radar_workspace(selected_ticker)

    with t6:
        render_sec_filings_workspace(selected_ticker)

    with t7:
        render_arxiv_research_workspace()
