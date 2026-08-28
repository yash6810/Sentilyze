"""
Workspace 3: 24/7 Autonomous Broker, Kelly Sizing & Staged Profit Scaler.
Includes Native Streamlit Live Position Tracking Chart (Zero Plotly Overhead).
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
from src.ui.components import render_workspace_header
from src.autonomous_trader import AutonomousTradingEngine
from src.data_ingestion import get_price_history


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
        "💰 Total Equity",
        f"${portfolio_summary.get('total_equity', 100000.0):,.2f}",
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
        st.dataframe(open_df, use_container_width=True)
    else:
        st.info(
            "No active open positions. The Autonomous Agent is waiting for high-conviction committee clearances."
        )

    # =========================================================================
    # LIVE POSITION TRACKER CHART (Native Streamlit, Zero Plotly)
    # =========================================================================
    st.markdown("---")
    st.markdown("### 📊 Live Holdings Chart & Staged Execution Levels (Native Engine)")

    open_positions = broker_instance.state.get("open_positions", {})
    available_tickers = (
        list(open_positions.keys()) if open_positions else [selected_ticker]
    )

    chart_ticker = st.selectbox(
        "Select Active Position to Track Live:",
        options=available_tickers,
        index=0,
    )

    try:
        # 1. Fetch recent price history (3-month window for clean high-res view)
        df_hist = get_price_history(chart_ticker, period="3mo", use_cache=True)

        if not df_hist.empty and "Close" in df_hist.columns:
            chart_data = pd.DataFrame(index=df_hist.index)
            chart_data["Market Price ($)"] = df_hist["Close"].values

            # If the stock is an active holding bought by the agent
            if chart_ticker in open_positions:
                pos = open_positions[chart_ticker]
                entry_p = float(pos.get("entry_price", df_hist["Close"].iloc[-1]))
                tp1_p = float(pos.get("tp1_target", entry_p * 1.06))
                tp2_p = float(pos.get("tp2_target", entry_p * 1.12))
                sl_p = float(pos.get("sl_target", entry_p * 0.95))
                shares = int(pos.get("shares", 100))
                scaled = pos.get("scaled_out", False)

                chart_data["Agent Entry Price ($)"] = entry_p
                chart_data["Target 1 (+2.5 ATR Take Profit)"] = tp1_p
                chart_data["Target 2 (+4.5 ATR Runner)"] = tp2_p
                chart_data["Stop-Loss Floor ($)"] = sl_p

                # Status banner
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                curr_p = float(df_hist["Close"].iloc[-1])
                pos_pnl = (curr_p - entry_p) * shares
                pos_ret = ((curr_p - entry_p) / entry_p) * 100.0

                stat_col1.metric("Bought Shares", f"{shares:,} Shares")
                stat_col2.metric("Bot Entry Basis", f"${entry_p:,.2f}")
                stat_col3.metric(
                    "Position PnL",
                    f"${pos_pnl:+,.2f}",
                    delta=f"{pos_ret:+.2f}%",
                )
                stat_col4.metric(
                    "Strategy State",
                    "🛡️ RISK-FREE (Banked 50%)" if scaled else "⚡ 100% ACTIVE",
                )
            else:
                # Stock not currently held: display calibrated reference ATR brackets
                last_p = float(df_hist["Close"].iloc[-1])
                chart_data["Reference Entry ($)"] = last_p
                chart_data["Target 1 (+2.5 ATR)"] = last_p * 1.05
                chart_data["Stop-Loss Floor"] = last_p * 0.96
                st.caption(
                    f"Showing live market price structure and ATR reference brackets for {chart_ticker}."
                )

            # Native Streamlit line chart (Ultra-fast, responsive, NO Plotly)
            st.line_chart(
                chart_data,
                height=450,
                use_container_width=True,
                color=[
                    "#38BDF8",  # Market Price (Sky Blue)
                    "#F59E0B",  # Entry Price (Gold Amber)
                    "#10B981",  # Target 1 (Emerald Green)
                    "#818CF8",  # Target 2 (Indigo)
                    "#EF4444",  # Stop Loss (Crimson Red)
                ][: len(chart_data.columns)],
            )
        else:
            st.warning(f"Could not load price history for {chart_ticker}.")
    except Exception as e:
        st.error(f"Error rendering live chart: {e}")

    # Execution Logs Tab
    st.markdown("---")
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
