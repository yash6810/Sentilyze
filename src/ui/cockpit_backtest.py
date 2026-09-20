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
from src.ui.ws_trade_autopsy import render_trade_autopsy_workspace


def render_backtest_cockpit(selected_ticker: str):
    t1, t2, t3, t4, t5 = st.tabs(
        [
            "📈 Walk-Forward Backtest",
            "👑 300-Resource Master Benchmark",
            "📊 Institutional Factsheet",
            "⚡ Quant Alpha DAG Engine",
            "🔬 Trade Autopsy & WFO Lab",
        ]
    )

    with t1:
        render_backtesting_workspace(selected_ticker)

    with t2:
        render_quantum_tournament_workspace(selected_ticker)

    with t3:
        render_performance_factsheet_workspace()

    with t4:
        render_alpha_dag_workspace(selected_ticker)

    with t5:
        render_trade_autopsy_workspace(selected_ticker)
