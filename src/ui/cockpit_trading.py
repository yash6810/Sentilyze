"""
Cockpit 1: Live Trading & Alpha Execution
=========================================
Consolidates:
- Tab 1: 🔮 Live Directional ML Prediction & Microstructure Alpha
- Tab 2: 🏛️ 5-Agent Adversarial Committee War Room
- Tab 3: 🤖 24/7 Autonomous Live Paper Trader
- Tab 4: ⚡ Automated Broker Webhooks & API Dispatcher
- Tab 5: 🎙️ AI Pre-Market Morning Audio & Briefing
"""

import streamlit as st
from src.ui.ws_live_prediction import render_live_prediction_workspace
from src.ui.ws_committee import render_committee_workspace
from src.ui.ws_autonomous_trader import render_autonomous_trader_workspace
from src.ui.ws_broker_webhooks import render_broker_webhooks_workspace
from src.ui.ws_morning_briefing import render_morning_briefing_workspace


def render_trading_cockpit(selected_ticker: str):
    st.markdown(
        """
        <div style="margin-bottom: 16px;">
            <h1 style="margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -0.02em;">
                🎯 Live Trading & Alpha Cockpit
            </h1>
            <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.95rem;">
                Autonomous multi-agent consensus, real-time directional signals, and direct broker execution.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t1, t2, t3, t4, t5 = st.tabs(
        [
            "🔮 Directional Signals",
            "🏛️ 5-Agent Committee War Room",
            "🤖 Autonomous Paper Trader",
            "⚡ Broker Execution Webhooks",
            "🎙️ Morning AI Audio Briefing",
        ]
    )

    with t1:
        render_live_prediction_workspace(selected_ticker)

    with t2:
        render_committee_workspace(selected_ticker)

    with t3:
        render_autonomous_trader_workspace(selected_ticker)

    with t4:
        render_broker_webhooks_workspace(selected_ticker)

    with t5:
        render_morning_briefing_workspace(selected_ticker)
