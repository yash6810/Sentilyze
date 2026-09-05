"""
Interactive Multi-Agent War Room Visualizer Component for Streamlit.

Functions:
- Renders an editorial-styled council deliberation chamber.
- Visualizes individual specialist agent cards, confidence scores, and academic citations.
- Highlights CRO official verdict, Quarter-Kelly sizing, and ATR corridors.
"""

from typing import Dict, Any, List
import streamlit as st
from components.audio_squawk import render_audio_squawk_button


def render_multi_agent_war_room(
    ticker: str, resolution: Dict[str, Any], spot_price: float
):
    """
    Renders the complete 5-Agent War Room Council deliberation chamber.
    """
    st.markdown("### 🏛️ 5-Agent Quantitative War Room Deliberation Council")
    st.caption(
        "Round-table multi-agent consensus synthesizing Technicals, FinBERT NLP, Forensic Fundamentals, Real-Time Tape, and Adversarial Stress-Testing."
    )

    if not resolution or not isinstance(resolution, dict):
        st.info(
            "No active committee deliberation available for this asset. Select a ticker above to convene council."
        )
        return

    cro_info = resolution.get("cro_signoff", {})
    action_code = resolution.get("action_code", "HOLD")
    final_res = resolution.get("final_resolution", "NEUTRAL")
    conviction = float(resolution.get("consensus_conviction_pct", 50.0))
    kelly_pct = float(resolution.get("kelly_allocation_pct", 0.0))
    tp1 = float(resolution.get("tp1_target", spot_price * 1.05))
    tp2 = float(resolution.get("tp2_target", spot_price * 1.10))
    sl = float(resolution.get("stop_loss_target", spot_price * 0.95))

    # 1. CRO Verdict Banner
    if action_code in ["EXECUTE_BUY", "SCALE_IN"]:
        banner_bg = "linear-gradient(135deg, rgba(0,212,170,0.15) 0%, rgba(16,185,129,0.05) 100%)"
        border_color = "#00D4AA"
        badge_color = "#00D4AA"
    elif action_code == "VETO":
        banner_bg = "linear-gradient(135deg, rgba(239,68,68,0.15) 0%, rgba(185,28,28,0.05) 100%)"
        border_color = "#EF4444"
        badge_color = "#EF4444"
    else:
        banner_bg = (
            "linear-gradient(135deg, rgba(234,179,8,0.15) 0%, rgba(161,98,7,0.05) 100%)"
        )
        border_color = "#EAB308"
        badge_color = "#EAB308"

    st.markdown(
        f"""
        <div style="background: {banner_bg}; border: 1.5px solid {border_color}; border-radius: 10px; padding: 16px 20px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 11px; font-weight: 700; color: {badge_color}; text-transform: uppercase; letter-spacing: 1px;">CHIEF RISK OFFICER (CRO) OFFICIAL VERDICT</span>
                    <h3 style="margin: 4px 0px 8px 0px; color: #f0f6fc;">{final_res}</h3>
                    <p style="margin: 0; color: #8b949e; font-size: 13px;">{cro_info.get('cro_thesis', 'Consensus reached across specialist agents.')}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Key Metrics Row
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric("Consensus Conviction", f"{conviction:.1f}%")
    with m_col2:
        st.metric("Quarter-Kelly Sizing", f"{kelly_pct:.1f}%")
    with m_col3:
        st.metric("TP1 Target (+2.5 ATR)", f"${tp1:,.2f}")
    with m_col4:
        st.metric("TP2 Target (+4.5 ATR)", f"${tp2:,.2f}")
    with m_col5:
        st.metric("Stop Loss Target", f"${sl:,.2f}")

    # Audio Squawk Broadcast
    squawk_text = (
        f"Sentilyze Trading Desk. Chief Risk Officer verdict on {ticker}: {final_res}. "
        f"Consensus conviction is {conviction:.1f} percent. Quarter Kelly allocation is {kelly_pct:.1f} percent. "
        f"TP1 target at {tp1:.2f} dollars, stop loss at {sl:.2f} dollars."
    )
    render_audio_squawk_button(
        squawk_text, button_label=f"🎙️ Broadcast {ticker} Audio Squawk"
    )

    st.markdown("---")

    # 3. Specialist Agent Deliberation Cards
    testimonies = resolution.get("agent_testimonies", [])
    st.markdown("#### 🗣️ Specialist Agent Testimonies & Audit Trail")

    card_cols = st.columns(len(testimonies) if testimonies else 1)

    agent_icons = {
        "Technical Momentum Specialist": "📈",
        "Sentiment & Alternative Data Specialist": "📰",
        "Forensic & Valuation Auditor": "🔍",
        "Real-Time Price & Tape Scout": "⏱️",
        "Adversarial Red-Team Specialist": "🛑",
    }

    for idx, report in enumerate(testimonies):
        with card_cols[idx % len(card_cols)]:
            name = report.get("agent_name", "Specialist Agent")
            role = report.get("role", "Domain Auditor")
            vote = report.get("vote", "NEUTRAL")
            c_score = float(report.get("conviction_score", 50.0))
            thesis = report.get("thesis", "No testimony provided.")
            icon = agent_icons.get(name, "🤖")

            if vote in ["BUY", "CLEAR"]:
                v_color = "#00D4AA"
            elif vote in ["VETO", "SELL"]:
                v_color = "#EF4444"
            else:
                v_color = "#EAB308"

            st.markdown(
                f"""
                <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; height: 100%; min-height: 220px;">
                    <div style="font-size: 20px; margin-bottom: 4px;">{icon}</div>
                    <div style="font-size: 13px; font-weight: 700; color: #f0f6fc;">{name}</div>
                    <div style="font-size: 11px; color: #8b949e; margin-bottom: 8px;">{role}</div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="background: rgba(255,255,255,0.05); padding: 2px 6px; border-radius: 4px; font-size: 11px; color: {v_color}; font-weight: 700;">{vote}</span>
                        <span style="font-size: 11px; color: #8b949e;">Conviction: <strong style="color: #f0f6fc;">{c_score:.0f}%</strong></span>
                    </div>
                    <p style="font-size: 11px; color: #c9d1d9; line-height: 1.4; margin: 0;">{thesis}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
