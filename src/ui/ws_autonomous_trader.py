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

    # =========================================================================
    # TARGET +100% ACCOUNT DOUBLING RADAR ($200,000 MILESTONE TRACKER)
    # =========================================================================
    from src.compound_engine import calculate_doubling_progress

    curr_eq = float(portfolio_summary.get("total_equity", 100000.0))
    init_cap = float(broker_instance.initial_cash)
    progress_data = calculate_doubling_progress(
        initial_capital=init_cap, current_equity=curr_eq
    )

    st.markdown("#### 🎯 Target +100% Account Doubling Radar ($200,000 Goal)")
    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    t_col1.metric("🏁 Initial Base", f"${init_cap:,.2f}")
    t_col2.metric(
        "📈 Net Compounded Gain",
        f"${progress_data['net_gain_dollars']:+,.2f}",
        delta=f"{((curr_eq - init_cap) / init_cap) * 100.0:+.2f}% Growth",
    )
    t_col3.metric(
        "⏳ Distance to $200k",
        f"${progress_data['goal_dollars_remaining']:,.2f}",
        delta=f"{progress_data['progress_pct']:.1f}% Completed",
    )
    t_col4.metric(
        "🔄 Compound Cycles Left",
        f"~{progress_data['cycles_remaining']} Cycles",
        delta="At avg +4.5% net/cycle",
    )

    # Visual Progress Bar towards $200,000
    st.progress(
        min(1.0, progress_data["progress_pct"] / 100.0),
        text=f"🚀 Doubling Trajectory: {progress_data['progress_pct']:.2f}% of $100k Profit Target Achieved",
    )

    # Controls Row
    universe_count = (
        len(auto_engine.universe_tickers)
        if hasattr(auto_engine, "universe_tickers")
        else 538
    )
    st.markdown(f"#### ⚡ {universe_count}-Universe Multi-Asset Capital Allocator")
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
    with ctrl_col1:
        st.markdown(
            f"""
            <div class="glass-card">
                <b>{universe_count}-Universe Capital Engine:</b>
                <ul>
                    <li><b>Alpha Discovery:</b> Scans all {universe_count} S&P stocks, ranking by 4-Agent Committee Quorum (>60%).</li>
                    <li><b>Kelly Distribution:</b> Allocates available capital proportionally to probability & reward/risk.</li>
                    <li><b>2-Stage Profit Scaling:</b> Takes +50% profit at +2.5 ATR, locks stop at Breakeven, and lets runners target +4.5 ATR.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ctrl_col2:
        max_slots = st.slider(
            "🎯 Max Active Positions",
            min_value=3,
            max_value=15,
            value=8,
            help="Number of concurrent multi-asset positions to hold.",
        )
        auto_pilot = st.toggle(
            "🔴 Live Market Auto-Pilot",
            value=st.session_state.get("auto_pilot_enabled", False),
            help="When enabled during regular market hours (09:30 - 16:00 EDT), continuously scans the universe and executes top setups.",
        )
        st.session_state["auto_pilot_enabled"] = auto_pilot

    with ctrl_col3:
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        btn_col_a, btn_col_b = st.columns([1, 1])
        with btn_col_a:
            if st.button(
                "🚀 Scan & Deploy",
                use_container_width=True,
                type="primary",
                help=f"Scans {universe_count} tickers and deploys Kelly capital into top setups.",
            ):
                with st.spinner(
                    f"Scanning {universe_count} universe assets and executing top Kelly setups (Max: {max_slots} slots)..."
                ):
                    cycle_res = auto_engine.run_autonomous_cycle(
                        max_concurrent_positions=max_slots
                    )
                    status = cycle_res.get("status", "SUCCESS")
                    if status == "SKIPPED_LOCKED":
                        st.warning(
                            "⏳ An autonomous cycle is already in progress in the background. Please wait a few seconds and try again."
                        )
                    else:
                        num_buys = len(cycle_res.get("buys", []))
                        num_tp1 = len(cycle_res.get("take_profits_tp1", []))
                        num_tp2 = len(cycle_res.get("take_profits_tp2", []))
                        num_sl = len(cycle_res.get("stop_losses", []))
                        elapsed = cycle_res.get("elapsed_seconds", 0)

                        if num_buys > 0:
                            bought_tickers = [
                                b.get("ticker") for b in cycle_res.get("buys", [])
                            ]
                            st.success(
                                f"🚀 Executed {num_buys} New Buy Orders: {bought_tickers} in {elapsed}s!"
                            )
                        elif num_tp1 + num_tp2 + num_sl > 0:
                            st.success(
                                f"🎯 Harvested Exits — TP1: {num_tp1} | TP2: {num_tp2} | Stops: {num_sl} in {elapsed}s!"
                            )
                        else:
                            st.info(
                                f"🛡️ Scanned {universe_count} assets in {elapsed}s. 4-Agent Committee preserved capital (No high-conviction setup passed the 2-vote quorum on this candle)."
                            )
                    st.rerun()
        with btn_col_b:
            if st.button(
                "⚡ Fast Price Sync & Discord",
                use_container_width=True,
                help="Sub-second spot price poll for active holdings, checks ATR scale-outs, and dispatches live card to Discord.",
            ):
                with st.spinner(
                    "Fast-polling live quotes for active holdings & notifying Discord..."
                ):
                    from src.realtime_tracker import (
                        update_live_holdings_prices_and_alert_discord,
                    )

                    guard_res = update_live_holdings_prices_and_alert_discord(
                        notify_discord=True
                    )
                    st.success(
                        f"✅ Updated {guard_res.get('updated_positions', 0)} holdings! (Discord alert sent: {guard_res.get('discord_alert_dispatched', False)})"
                    )
                    st.rerun()

    # =========================================================================
    # DEDICATED TICKER SENTINEL SWARM GUARDIANS (1 BOT PER STOCK)
    # =========================================================================
    from src.ticker_sentinel import TickerSentinelSwarm

    open_positions = broker_instance.state.get("open_positions", {})
    if open_positions:
        st.markdown("#### 🛡️ Dedicated Ticker Sentinel Guardians (1 Bot Per Stock)")
        st.caption(
            "Each active position is guarded by a dedicated sub-agent monitoring 15-min volume exhaustion, peak crest tops, and sub-second scale-outs."
        )

        swarm = TickerSentinelSwarm()
        swarm.sync_open_positions(open_positions)
        quotes_map = {
            t: {"price": float(p.get("current_price", p["entry_price"]))}
            for t, p in open_positions.items()
        }
        reports = swarm.audit_all_sentinels(quotes_map)

        num_cols = min(4, len(reports))
        for i in range(0, len(reports), num_cols):
            chunk = reports[i : i + num_cols]
            cols = st.columns(len(chunk))
            for idx, rep in enumerate(chunk):
                t = rep["ticker"]
                p_curr = rep["current_price"]
                pnl = rep["unrealized_pnl"]
                ret = rep["return_pct"]
                crest = rep["crest_analysis"]
                with cols[idx]:
                    st.markdown(
                        f"""
                        <div class="glass-card" style="padding: 12px; margin-bottom: 8px; border-top: 3px solid {'#10B981' if pnl >= 0 else '#EF4444'};">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <b style="font-size: 1.05rem; color: #F3F4F6;">🤖 {t} Sentinel</b>
                                <span style="font-size: 0.75rem; color: #10B981; font-weight: 700;">{rep['status']}</span>
                            </div>
                            <div style="font-size: 0.8rem; color: #94A3B8; margin: 4px 0;">Spot: <b>${p_curr:,.2f}</b> | PnL: <b style="color: {'#10B981' if pnl >= 0 else '#EF4444'};">${pnl:+,.2f} ({ret:+.2f}%)</b></div>
                            <div style="font-size: 0.75rem; color: #64748B;">Peak Seen: <b>${rep['highest_price_seen']:,.2f}</b> | Action: <b style="color: #38BDF8;">{crest['action']}</b></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # Open Positions Table
    st.markdown("#### 📦 Active Open Positions & Scale-Out Status")
    open_df = broker_instance.get_open_positions_df()
    if not open_df.empty:
        st.dataframe(open_df, use_container_width=True)

        # Tactical Position Action Controls
        st.markdown("##### 🎯 Quick Profit Banking & Position Management")
        action_cols = st.columns(len(open_positions))
        for idx, (t, pos) in enumerate(open_positions.items()):
            with action_cols[idx]:
                scaled = pos.get("scaled_out", False)
                entry_p = float(pos.get("entry_price", 0))
                curr_p = float(pos.get("current_price", entry_p))
                pos_pnl = (curr_p - entry_p) * int(pos.get("shares", 0))
                color = "#10B981" if pos_pnl >= 0 else "#EF4444"

                st.markdown(
                    f"<div style='font-size: 0.85rem; font-weight: 700; color: {color};'>● {t} (PnL: ${pos_pnl:+,.2f})</div>",
                    unsafe_allow_html=True,
                )
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if not scaled:
                        if st.button(
                            f"🔒 Bank 50%",
                            key=f"bank_{t}",
                            help=f"Slices 50% shares to lock in cash and moves Stop-Loss to Breakeven (${entry_p:,.2f}).",
                        ):
                            res = broker_instance.execute_manual_scale_out(t)
                            if res.get("success"):
                                st.success(f"Locked 50% profit on {t}!")
                                st.rerun()
                    else:
                        st.caption("🛡️ Risk-Free (Banked)")
                with col_btn2:
                    if st.button(
                        f"🛑 Close All",
                        key=f"close_{t}",
                        help=f"Liquidates 100% of {t} position at spot price (${curr_p:,.2f}).",
                    ):
                        res = broker_instance.execute_manual_sell(
                            t, reason="MANUAL_UI_EXIT"
                        )
                        if res.get("success"):
                            st.warning(f"Closed {t}!")
                            st.rerun()
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
                    "Execution State",
                    (
                        "🛡️ 50% Banked (Risk-Free)"
                        if scaled
                        else "⚡ 100% Active (Phase 1)"
                    ),
                    help="Phase 1 (100% Active): Full position aiming for Target 1 (+2.5 ATR profit). Phase 2 (50% Banked): Sliced half profit, stop moved to Breakeven (0 loss risk), remainder runner aiming for Target 2 (+4.5 ATR).",
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
                "Company Name": st.column_config.TextColumn(
                    "Company Name", width="medium"
                ),
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
