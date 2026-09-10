"""
Cockpit 5: Deep Quant & Explainability (XAI)
============================================
Consolidates:
- Tab 1: 🧠 SHAP Feature Importance & Trees
- Tab 2: 🔄 Market-Neutral Cointegration & Pairs Stat-Arb
- Tab 3: 🕸️ GNN Supply Chain Contagion & Spillover
- Tab 4: 🌪️ Black Swan Crisis & Macro Stress Simulator
- Tab 5: 🕵️ Forensic Accounting (Beneish M-Score) & DCF Intrinsic Valuation
- Tab 6: 🤖 Deep RL Policy Agent & Strategy Incubator
"""

import streamlit as st
from src.ui.ws_xai_shap import render_xai_workspace
from src.ui.ws_market_neutral_statarb import render_market_neutral_statarb_workspace
from src.ui.ws_deep_quant import render_deep_quant_workspace
from src.ui.ws_drl_agent import render_drl_agent_workspace
from src.ui.ws_strategy_incubator import render_strategy_incubator_workspace


def render_deep_quant_cockpit(selected_ticker: str):
    st.markdown(
        """
        <div style="margin-bottom: 16px;">
            <h1 style="margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -0.02em;">
                🧠 Deep Quant & Explainability (XAI) Cockpit
            </h1>
            <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.95rem;">
                SHAP attribution trees, causal beta residualization, supply chain graph networks, and crisis stress testing.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t1, t2, t3, t4, t5, t6 = st.tabs(
        [
            "🧠 SHAP Explainability",
            "🔄 Market-Neutral Stat-Arb",
            "🕸️ GNN Supply Chain Shock",
            "🌪️ Black Swan Stress Test",
            "🕵️ Forensic & DCF Valuation",
            "🤖 DRL Policy & Incubator",
        ]
    )

    with t1:
        render_xai_workspace(selected_ticker)

    with t2:
        render_market_neutral_statarb_workspace(selected_ticker)

    with t3:
        render_deep_quant_workspace(selected_ticker, mode="gnn")

    with t4:
        render_deep_quant_workspace(selected_ticker, mode="stress")

    with t5:
        st.subheader("🕵️ Forensic Accounting & Intrinsic Valuation")
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            st.markdown("#### Beneish M-Score Earnings Manipulation")
            render_deep_quant_workspace(selected_ticker, mode="forensic")
        with sub_col2:
            st.markdown("#### DCF Intrinsic Valuation Model")
            render_deep_quant_workspace(selected_ticker, mode="dcf")

    with t6:
        st.subheader("🤖 Deep Reinforcement Learning & Evolutionary Incubator")
        drl_tab1, drl_tab2 = st.tabs(["DRL Policy Agent", "Strategy Incubator"])
        with drl_tab1:
            render_drl_agent_workspace(selected_ticker)
        with drl_tab2:
            render_strategy_incubator_workspace(selected_ticker)
