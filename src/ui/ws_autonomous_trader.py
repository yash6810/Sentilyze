"""
Workspace 3: 24/7 Autonomous Broker, Kelly Sizing & Staged Profit Scaler.
"""

import os
import json
import streamlit as st
import pandas as pd
from src.ui.components import render_workspace_header
from src.autonomous_trader import AutonomousTradingEngine


def render_autonomous_trader_workspace(selected_ticker: str):
    """Renders the 24/7 Autonomous Live Trading & News Agent interface."""
    render_workspace_header(
        title="🤖 24/7 Autonomous Live Trading & News Agent",
        subtitle="Multi-Source News Ingestion + 4-Agent Committee + Kelly Allocation + 2-Stage Staged Profit Scaler",
        badge_text="24/7 DAEMON ACTIVE",
        badge_color="#10B981",
    )

    auto_engine = AutonomousTradingEngine()
    broker_instance = auto_engine.broker
    portfolio_summary = broker_instance.get_portfolio_summary()

    # Metrics Bar
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "💰 Total Equity", f"${portfolio_summary.get('total_equity', 100000.0):,.2f}"
    )
    m2.metric("💵 Cash Balance", f"${portfolio_summary.get('cash', 100000.0):,.2f}")
    m3.metric(
        "📈 Unrealized PnL",
        f"${portfolio_summary.get('unrealized_pnl', 0.0):+,.2f}",
        delta=f"{portfolio_summary.get('unrealized_pnl_pct', 0.0):+.2f}%",
    )
    m4.metric("🏆 Win Rate", f"{portfolio_summary.get('win_rate', 0.0):.1f}%")

    # Controls Row
    st.markdown("#### ⚡ Autonomous Cycle Execution")
    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        st.markdown(
            """
            <div class="glass-card">
                <b>24/7 Autonomous Agent Loop:</b> Continuously ingests <b>Google News RSS + Finnhub + Marketaux</b>, 
                evaluates the <b>4-Agent Committee</b>, allocates capital via <b>Kelly Sizing</b>, and manages 
                <b>2-Stage Staged Profit Exits (50% @ TP1, Trailing Breakeven Stop, 50% @ TP2)</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ctrl_col2:
        if st.button("🚀 Run Autonomous Decision Cycle Now", use_container_width=True):
            with st.spinner(
                "Executing autonomous news scan and trade management cycle..."
            ):
                cycle_res = auto_engine.run_autonomous_cycle()
                st.success(
                    f"✅ Cycle complete in {cycle_res.get('elapsed_seconds', 0)}s! "
                    f"Buys: {len(cycle_res.get('buys', []))}, TP1s: {len(cycle_res.get('take_profits_tp1', []))}, TP2s: {len(cycle_res.get('take_profits_tp2', []))}"
                )
                st.rerun()

    # Open Positions Table
    st.markdown("#### 📦 Active Open Positions & Scale-Out Status")
    open_df = broker_instance.get_open_positions_df()
    if not open_df.empty:
        st.dataframe(
            open_df.style.format(
                {
                    "Entry Price": "${:,.2f}",
                    "Current Price": "${:,.2f}",
                    "Unrealized PnL": "${:+,.2f}",
                    "Return (%)": "{:+.2f}%",
                    "TP1 Target": "${:,.2f}",
                    "TP2 Target": "${:,.2f}",
                    "Stop-Loss": "${:,.2f}",
                }
            ),
            use_container_width=True,
        )
    else:
        st.info(
            "No active open positions. The Autonomous Agent is waiting for high-conviction committee clearances."
        )

    # Execution Logs Tab
    st.markdown("#### 📜 Live Execution Audit Log")
    log_file = os.path.join("results", "autonomous_execution_log.json")
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                logs_data = json.load(f)
            st.json(logs_data)
        except Exception:
            st.write(
                "Execution logs available in `results/autonomous_execution_log.json`."
            )
