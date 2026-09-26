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

    tabs = [
        "🧠 SHAP Explainability",
        "🔄 Market-Neutral Stat-Arb",
        "🕸️ GNN Supply Chain Shock",
        "🌪️ Black Swan Stress Test",
        "🕵️ Forensic & DCF Valuation",
        "🤖 DRL Policy & Incubator",
    ]

    selected_tab = (
        st.segmented_control(
            "Deep Quant Navigation",
            options=tabs,
            default=tabs[0],
            label_visibility="collapsed",
            key=f"deep_quant_subnav_{selected_ticker}",
        )
        or tabs[0]
    )

    st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

    if selected_tab == "🧠 SHAP Explainability":
        render_xai_workspace(selected_ticker)
    elif selected_tab == "🔄 Market-Neutral Stat-Arb":
        render_market_neutral_statarb_workspace(selected_ticker)
    elif selected_tab == "🕸️ GNN Supply Chain Shock":
        render_deep_quant_workspace(selected_ticker, mode="gnn")
    elif selected_tab == "🌪️ Black Swan Stress Test":
        render_deep_quant_workspace(selected_ticker, mode="stress")
    elif selected_tab == "🕵️ Forensic & DCF Valuation":
        st.subheader("🕵️ Forensic Accounting & Intrinsic Valuation")
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            st.markdown("#### Beneish M-Score Earnings Manipulation")
            render_deep_quant_workspace(selected_ticker, mode="forensic")
        with sub_col2:
            st.markdown("#### DCF Intrinsic Valuation Model")
            render_deep_quant_workspace(selected_ticker, mode="dcf")
    elif selected_tab == "🤖 DRL Policy & Incubator":
        st.subheader("🤖 Deep Reinforcement Learning & Evolutionary Incubator")
        sub_pills = (
            st.pills(
                "DRL Mode",
                ["DRL Policy Agent", "Strategy Incubator"],
                default="DRL Policy Agent",
                label_visibility="collapsed",
                key=f"drl_submode_{selected_ticker}",
            )
            or "DRL Policy Agent"
        )
        if sub_pills == "DRL Policy Agent":
            render_drl_agent_workspace(selected_ticker)
        else:
            render_strategy_incubator_workspace(selected_ticker)
