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

    t1, t2, t3 = st.tabs(
        [
            "💼 HRP & Kelly Allocation",
            "🧬 Correlation & Diversity Grader",
            "🌐 Macro Liquidity & Yields",
        ]
    )

    with t1:
        render_portfolio_workspace(selected_ticker)

    with t2:
        render_portfolio_diversity_workspace()

    with t3:
        render_macro_liquidity_workspace()
