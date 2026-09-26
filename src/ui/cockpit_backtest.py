"""
Cockpit 4: Multi-Decade Backtest & Performance Factsheet
========================================================
Consolidates:
- Tab 1: 📈 Walk-Forward Backtesting Engine
- Tab 2: 👑 300-Resource Master Tournament & Benchmark
- Tab 3: 📊 Institutional Risk & Alpha Factsheet
- Tab 4: ⚡ Autonomous Quant Alpha DAG Engine (Gen-3)
"""

import streamlit as st
from src.ui.ws_backtesting import render_backtesting_workspace
from src.ui.ws_quantum_tournament import render_quantum_tournament_workspace
from src.ui.ws_performance_factsheet import render_performance_factsheet_workspace
from src.ui.ws_alpha_dag import render_alpha_dag_workspace


def render_backtest_cockpit(selected_ticker: str):
    st.markdown(
        """
        <div style="margin-bottom: 16px;">
            <h1 style="margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -0.02em;">
                📈 Multi-Decade Backtesting & Performance Factsheet
            </h1>
            <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.95rem;">
                Purged walk-forward validation, the 300-resource multi-decade tournament, and institutional tear sheets.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = [
        "📈 Walk-Forward Backtest",
        "👑 300-Resource Master Benchmark",
        "📊 Institutional Factsheet",
        "⚡ Quant Alpha DAG Engine",
    ]

    selected_tab = (
        st.segmented_control(
            "Backtest Navigation",
            options=tabs,
            default=tabs[0],
            label_visibility="collapsed",
            key=f"backtest_subnav_{selected_ticker}",
        )
        or tabs[0]
    )

    st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

    if selected_tab == "📈 Walk-Forward Backtest":
        render_backtesting_workspace(selected_ticker)
    elif selected_tab == "👑 300-Resource Master Benchmark":
        render_quantum_tournament_workspace(selected_ticker)
    elif selected_tab == "📊 Institutional Factsheet":
        render_performance_factsheet_workspace()
    elif selected_tab == "⚡ Quant Alpha DAG Engine":
        render_alpha_dag_workspace(selected_ticker)
