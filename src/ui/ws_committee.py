"""
Workspace 2: 4-Agent Trading Committee Round-Table Deliberations.
"""

import streamlit as st
from src.ui.components import render_workspace_header, render_conviction_gauge
from src.agent_committee import convene_trading_committee


def render_committee_workspace(ticker: str):
    """Renders the 4-Agent Trading Committee deliberation panel."""
    render_workspace_header(
        title=f"🏛️ 4-Agent Trading Committee Round-Table ({ticker})",
        subtitle="Specialized AI Agents: Technicals, NLP Sentiment, Forensic DCF Valuation, and Chief Risk Officer",
        badge_text="CONSENSUS VOTING",
        badge_color="#3B82F6",
    )

    with st.spinner(f"Convening Multi-Agent Committee for {ticker}..."):
        try:
            delib = convene_trading_committee(ticker, save_resolution=False)
        except Exception as e:
            st.error(f"Committee deliberation error: {e}")
            return

    res = delib.get("final_resolution", "HOLD")
    conv = delib.get("consensus_conviction_pct", 65.0)
    action_code = delib.get("action_code", "HOLD")
    votes = delib.get("committee_votes", {})
    cro = delib.get("cro_signoff", {})

    # Top Verdict Bar
    v_col1, v_col2, v_col3 = st.columns([1.5, 1, 1])
    with v_col1:
        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 4px solid #3B82F6;">
                <div style="font-size: 0.8rem; color: #94A3B8; font-weight: 700; font-family: 'JetBrains Mono', monospace;">EXECUTIVE VERDICT</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #F3F4F6; margin-top: 4px;">{res}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with v_col2:
        st.metric("🎯 Action Code", action_code)
    with v_col3:
        st.metric(
            "🛡️ CRO Status",
            cro.get("status", "APPROVED"),
            delta=f"Kelly Sizing: +{cro.get('approved_kelly_pct', 8.0):.1f}%",
        )

    render_conviction_gauge(conv, label=f"COMMITTEE CONSENSUS CONVICTION ({ticker})")

    st.markdown("### 👥 Specialized Agent Deliberations")
    c1, c2 = st.columns(2)

    with c1:
        # Technical Specialist
        tech = votes.get("Technical Specialist", {})
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0;">📈 1. Technical Specialist</h4>
                    <span class="badge-bull">{tech.get('vote', 'HOLD')}</span>
                </div>
                <hr style="border-color: rgba(255,255,255,0.08); margin: 10px 0;">
                <p style="font-size: 0.9rem; color: #94A3B8;"><b>RSI (14-Day):</b> {tech.get('rsi_14', 50.0):.1f}</p>
                <p style="font-size: 0.9rem; color: #94A3B8;"><b>200 SMA Trend:</b> {tech.get('trend_200sma', 'NEUTRAL')}</p>
                <p style="font-size: 0.85rem; color: #CBD5E1;"><i>{tech.get('rationale', 'Standard technical analysis')}</i></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Forensic Valuation Specialist
        fund = votes.get("Forensic Accounting & Valuation Specialist", {})
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0;">🏛️ 3. Forensic DCF Specialist</h4>
                    <span class="badge-bull">{fund.get('vote', 'HOLD')}</span>
                </div>
                <hr style="border-color: rgba(255,255,255,0.08); margin: 10px 0;">
                <p style="font-size: 0.9rem; color: #94A3B8;"><b>Intrinsic DCF Value:</b> ${fund.get('dcf_fair_value', 0.0):.2f}</p>
                <p style="font-size: 0.9rem; color: #94A3B8;"><b>Margin of Safety:</b> {fund.get('margin_of_safety_pct', 0.0):+.1f}%</p>
                <p style="font-size: 0.85rem; color: #CBD5E1;"><i>{fund.get('rationale', 'Valuation assessment')}</i></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        # Sentiment Specialist
        sent = votes.get("Sentiment & Alternative Data Specialist", {})
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0;">🧠 2. NLP Sentiment Specialist</h4>
                    <span class="badge-bull">{sent.get('vote', 'HOLD')}</span>
                </div>
                <hr style="border-color: rgba(255,255,255,0.08); margin: 10px 0;">
                <p style="font-size: 0.9rem; color: #94A3B8;"><b>FinBERT Score:</b> {sent.get('sentiment_score', 60.0):.1f}/100</p>
                <p style="font-size: 0.9rem; color: #94A3B8;"><b>Smart Money Flow:</b> {sent.get('smart_money_score', 50.0):.1f}</p>
                <p style="font-size: 0.85rem; color: #CBD5E1;"><i>{sent.get('rationale', 'Sentiment scoring')}</i></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Chief Risk Officer
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0;">🛡️ 4. Chief Risk Officer (CRO)</h4>
                    <span class="badge-bull">{cro.get('status', 'APPROVED')}</span>
                </div>
                <hr style="border-color: rgba(255,255,255,0.08); margin: 10px 0;">
                <p style="font-size: 0.9rem; color: #94A3B8;"><b>Macro VIX Gate:</b> {cro.get('macro_vix_level', 15.0):.1f} ({cro.get('vix_regime', 'NORMAL')})</p>
                <p style="font-size: 0.9rem; color: #94A3B8;"><b>Approved Kelly Allocation:</b> +{cro.get('approved_kelly_pct', 8.0):.1f}%</p>
                <p style="font-size: 0.85rem; color: #CBD5E1;"><i>{cro.get('risk_assessment', 'Risk cleared')}</i></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
