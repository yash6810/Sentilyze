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
from src.ui.ws_tail_risk import render_tail_risk_workspace
from src.ui.ws_execution_sor import render_execution_sor_workspace


def render_portfolio_cockpit(selected_ticker: str):
    t1, t2, t3, t4, t5 = st.tabs(
        [
            "💼 HRP & Kelly Allocation",
            "🧬 Correlation & Diversity Grader",
            "🌐 Macro Liquidity & Yields",
            "🌪️ EVT Tail-Risk & Crisis Replay",
            "⚡ Smart Execution SOR & Impact",
        ]
    )

    with t1:
        render_portfolio_workspace(selected_ticker)

    with t2:
        render_portfolio_diversity_workspace()

    with t3:
        render_macro_liquidity_workspace()

    with t4:
        render_tail_risk_workspace(selected_ticker)

    with t5:
        render_execution_sor_workspace(selected_ticker)
