"""
Cockpit 1: Live Trading & Alpha Execution
=========================================
Consolidates:
- Tab 1: 🔮 Live Directional ML Prediction & Microstructure Alpha
- Tab 2: 🏛️ 5-Agent Adversarial Committee War Room
- Tab 3: 🤖 24/7 Autonomous Live Paper Trader
- Tab 4: ⚡ Automated Broker Webhooks & API Dispatcher
- Tab 5: 🎙️ AI Pre-Market Morning Audio & Briefing
"""

import os
import json
import streamlit as st
from src.ui.ws_live_prediction import render_live_prediction_workspace
from src.ui.ws_committee import render_committee_workspace
from src.ui.ws_autonomous_trader import render_autonomous_trader_workspace
from src.ui.ws_broker_webhooks import render_broker_webhooks_workspace
from src.ui.ws_morning_briefing import render_morning_briefing_workspace


def _render_monday_premarket_intel():
    """Renders Monday Opening Intelligence, Overnight Futures Pulse, and Almgren-Chriss Routing."""
    pulse_path = "results/overnight_futures_pulse.json"
    plan_path = "results/monday_opening_execution_plan.json"

    pulse_data = {}
    if os.path.exists(pulse_path):
        try:
            with open(pulse_path, "r", encoding="utf-8") as f:
                pulse_data = json.load(f)
        except Exception:
            pass

    plan_data = {}
    if os.path.exists(plan_path):
        try:
            with open(plan_path, "r", encoding="utf-8") as f:
                plan_data = json.load(f)
        except Exception:
            pass

    if not pulse_data and not plan_data:
        return

    gap_pct = pulse_data.get("composite_gap_pct", 0.50)
    bias = pulse_data.get("market_bias", "🟢 BULLISH_GAP_OPEN")
    proj_impact = pulse_data.get("total_projected_impact_dollar", 219.45)
    proj_eq = pulse_data.get("projected_equity_at_open", 144672.86)
    ticket = plan_data.get("opening_order_ticket", {})
    top_cand = ticket.get("candidate", "MSFT")
    top_conv = ticket.get("conviction_pct", 80.6)

    with st.expander(
        f"🌌 **MONDAY 09:30 EDT PRE-MARKET RADAR** | Implied Open Gap: **{gap_pct:+.2f}%** ({bias}) | Top Setup: **{top_cand} ({top_conv:.1f}%)**",
        expanded=False,
    ):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📊 Implied Gap", f"{gap_pct:+.2f}%", delta="S&P / NQ Composite")
        m2.metric(
            "🛡️ Risk Posture", pulse_data.get("risk_posture", "AGGRESSIVE_RUNNER")
        )
        m3.metric(
            "💵 Projected Open PnL",
            f"${proj_impact:+,.2f}",
            delta=f"Open Eq: ${proj_eq:,.2f}",
        )
        m4.metric(
            "🎯 Top Watchlist Pick",
            f"{top_cand} ({top_conv:.1f}%)",
            delta="Almgren-Chriss Ready",
        )

        st.markdown(
            "<hr style='margin: 8px 0; border-color: rgba(255,255,255,0.08);'>",
            unsafe_allow_html=True,
        )
        col_f, col_w, col_p = st.columns(3)

        with col_f:
            st.markdown("##### 🌐 Global Futures Pulse")
            quotes = pulse_data.get("futures_quotes", {})
            for sym, q in quotes.items():
                st.caption(
                    f"**{sym}** ({q.get('name', '')}): `${q.get('price', 0):,.2f}` (`{q.get('change_pct', 0):+.2f}%`)"
                )
            macro = pulse_data.get("macro_pulse", {})
            for msym, mq in macro.items():
                st.caption(
                    f"**{msym}** ({mq.get('name', '')}): `{mq.get('value', 0):.2f}` (`{mq.get('change_pct', 0):+.2f}%`)"
                )

        with col_w:
            st.markdown("##### 🚀 9:30 AM Opening Order Ticket")
            if ticket:
                st.caption(
                    f"**Candidate**: `{ticket.get('candidate')}` ({ticket.get('conviction_pct')}%)"
                )
                st.caption(
                    f"**Order Action**: `BUY {ticket.get('total_shares')} shares` (~${ticket.get('total_capital_allocation'):,.2f})"
                )
                brackets = ticket.get("brackets", {})
                st.caption(
                    f"**Entry Limit**: `${brackets.get('entry_limit', 0):.2f}` | **SL**: `${brackets.get('stop_loss_target', 0):.2f}`"
                )
                st.caption(
                    f"**TP1 Target**: `${brackets.get('tp1_target', 0):.2f}` | **TP2**: `${brackets.get('tp2_target', 0):.2f}`"
                )
                st.caption(
                    f"**Slicing**: Almgren-Chriss 10-interval optimal execution (09:30 - 10:00 EDT)"
                )

        with col_p:
            st.markdown("##### 🛡️ Position Trailing Stop Status")
            impacts = pulse_data.get("position_impacts", {})
            for t, imp in impacts.items():
                st.caption(
                    f"• **{t}**: ${imp.get('position_value', 0):,.2f} | Implied `{imp.get('projected_change_pct', 0):+.2f}%` [`{imp.get('locked_profit')}`]"
                )


def render_trading_cockpit(selected_ticker: str):
    st.markdown(
        """
        <div style="margin-bottom: 16px;">
            <h1 style="margin: 0; font-size: 2rem; font-weight: 800; letter-spacing: -0.02em;">
                🎯 Live Trading & Alpha Cockpit
            </h1>
            <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.95rem;">
                Autonomous multi-agent consensus, real-time directional signals, and direct broker execution.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_monday_premarket_intel()

    tabs = [
        "🔮 Directional Signals",
        "🏛️ 5-Agent Committee War Room",
        "🤖 Autonomous Paper Trader",
        "⚡ Broker Execution Webhooks",
        "🎙️ Morning AI Audio Briefing",
    ]

    selected_tab = (
        st.segmented_control(
            "Trading Workspace Navigation",
            options=tabs,
            default=tabs[0],
            label_visibility="collapsed",
            key=f"trade_cockpit_subnav_{selected_ticker}",
        )
        or tabs[0]
    )

    st.markdown("<div style='margin-bottom: 14px;'></div>", unsafe_allow_html=True)

    if selected_tab == "🔮 Directional Signals":
        render_live_prediction_workspace(selected_ticker)
    elif selected_tab == "🏛️ 5-Agent Committee War Room":
        render_committee_workspace(selected_ticker)
    elif selected_tab == "🤖 Autonomous Paper Trader":
        render_autonomous_trader_workspace(selected_ticker)
    elif selected_tab == "⚡ Broker Execution Webhooks":
        render_broker_webhooks_workspace(selected_ticker)
    elif selected_tab == "🎙️ Morning AI Audio Briefing":
        render_morning_briefing_workspace(selected_ticker)
