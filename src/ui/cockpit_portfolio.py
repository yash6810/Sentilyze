"""
Cockpit 3: Portfolio & Risk Optimization
========================================
Consolidates:
- Tab 1: 💼 Portfolio Kelly Sizing & HRP Allocation
- Tab 2: 🧬 Portfolio Diversity & Correlation Grader
- Tab 3: 🌐 Real-Time Macro Liquidity & Yield Radar
"""

import streamlit as st
from src.ui.ws_portfolio import render_portfolio_workspace
from src.ui.ws_portfolio_diversity import render_portfolio_diversity_workspace
from src.ui.ws_macro_liquidity import render_macro_liquidity_workspace


def render_portfolio_cockpit(selected_ticker: str):
    st.markdown(
        """
        <div style="margin-bottom: 16px;">
            <h1 style="margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -0.02em;">
                💼 Portfolio & Risk Optimization Engine
            </h1>
            <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.95rem;">
                Hierarchical Risk Parity (HRP), regime-aware Kelly capital growth, and macroeconomic yield curves.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = [
        "💼 HRP & Kelly Allocation",
        "🧬 Correlation & Diversity Grader",
        "🌐 Macro Liquidity & Yields",
    ]

    selected_tab = (
        st.segmented_control(
            "Portfolio Navigation",
            options=tabs,
            default=tabs[0],
            label_visibility="collapsed",
            key=f"portfolio_subnav_{selected_ticker}",
        )
        or tabs[0]
    )

    st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

    if selected_tab == "💼 HRP & Kelly Allocation":
        render_portfolio_workspace(selected_ticker)
    elif selected_tab == "🧬 Correlation & Diversity Grader":
        render_portfolio_diversity_workspace()
    elif selected_tab == "🌐 Macro Liquidity & Yields":
        render_macro_liquidity_workspace()
