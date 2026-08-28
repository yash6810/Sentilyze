"""
Workspace 3: 24/7 Autonomous Broker, Kelly Sizing & Staged Profit Scaler.
Includes Native Streamlit Live Position Tracking Chart & Detailed Trade History Timing Logs.
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
    st.markdown("#### ⚡ Autonomous Cycle Execution & Trade Timing")
    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        st.markdown(
            """
            <div class="glass-card">
                <b>Execution Mechanics:</b>
                <ul>
                    <li><b>News Timing:</b> Continuously evaluates <b>4-Station Reddit News (1-Day-Prior) + NewsAPI/Finnhub</b> before and during market sessions.</li>
                    <li><b>Trade Execution:</b> When the 4-Agent Committee reaches quorum (>60% confidence), orders execute at <b>Spot Market Price</b> (during market hours 09:30-16:00 EST) or previous close.</li>
                    <li><b>Risk Asymmetry:</b> Stops capped at <b>-1.5 ATR</b>; Take-Profits staged at <b>+2.5 ATR (50% Banked & Breakeven Stop)</b> and <b>+4.5 ATR (Runner)</b>.</li>
                </ul>
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

    # =========================================================================
    # CLOSED TRADE HISTORY & TIMING AUDIT
    # =========================================================================
    st.markdown("---")
    st.markdown("#### 📜 Executed Trade History & Fill Timing Log")
    closed_df = broker_instance.get_closed_trades_df()
    if not closed_df.empty:
        st.dataframe(
            closed_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Shares": st.column_config.NumberColumn(
                    "Shares", format="%d", width="small"
                ),
                "Entry Price ($)": st.column_config.NumberColumn(
                    "Entry Price", format="$%.2f", width="medium"
                ),
                "Exit Price ($)": st.column_config.NumberColumn(
                    "Exit Price", format="$%.2f", width="medium"
                ),
                "Entry Date": st.column_config.TextColumn(
                    "Bought Date / Time",
                    help="When the agent bought the shares",
                    width="medium",
                ),
                "Exit Date": st.column_config.TextColumn(
                    "Closed Date / Time",
                    help="When the agent sold/exited the position",
                    width="medium",
                ),
                "Net PnL ($)": st.column_config.NumberColumn(
                    "Net PnL ($)", format="$%+.2f", width="medium"
                ),
                "Return (%)": st.column_config.NumberColumn(
                    "Return (%)", format="%+.2f%%", width="small"
                ),
                "Exit Reason": st.column_config.TextColumn(
                    "Exit Trigger Reason", width="large"
                ),
            },
        )
    else:
        st.info("No closed trades yet. Open holdings are actively running.")

    # =========================================================================
    # SELF-IMPROVING AGENT LEARNING MEMORY & AUTOPSY PANEL
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🧠 Autonomous Agent Self-Improvement & Learning Memory")
    memory_file = os.path.join("results", "agent_learning_memory.json")
    if os.path.exists(memory_file):
        try:
            with open(memory_file, "r") as mf:
                mem_data = json.load(mf)

            # Top weights bar
            weights = mem_data.get("agent_voting_weights", {})
            w1, w2, w3, w4 = st.columns(4)
            w1.metric(
                "📈 Technicals Weight",
                f"{weights.get('technicals_weight', 0.30)*100:.1f}%",
            )
            w2.metric(
                "📰 NLP Sentiment Weight",
                f"{weights.get('sentiment_weight', 0.35)*100:.1f}%",
            )
            w3.metric(
                "🏛️ Valuation Weight",
                f"{weights.get('valuation_weight', 0.15)*100:.1f}%",
            )
            w4.metric(
                "🛡️ CRO Risk Weight",
                f"{weights.get('cro_weight', 0.20)*100:.1f}%",
            )

            # Trade autopsies list
            autopsies = mem_data.get("recent_trade_autopsies", [])
            if autopsies:
                st.markdown("#### 🔬 Recent Trade Autopsy Lessons")
                for a in reversed(autopsies[-5:]):
                    st.info(
                        f"**{a.get('verdict', 'TRADE AUTOPSY')} ({a.get('ticker')})**: {a.get('lesson')} (PnL: `${a.get('pnl', 0):+,.2f}` | `{a.get('return_pct', 0):+.2f}%`)"
                    )
        except Exception as me:
            st.caption(f"Learning memory status: {me}")
    else:
        st.info("Agent learning memory will initialize on the next closed trade cycle.")

    # Execution Logs Tab
    st.markdown("---")
    st.markdown("#### 🔍 Live Execution Audit Log (Raw JSON)")
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
