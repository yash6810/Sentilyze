"""
Workspace: AI Pre-Market Audio & Executive Morning Intelligence Briefing.
Generates institutional pre-market podcast audio, Wall Street research memoranda,
and catalyst action matrices before the 9:30 AM market opening bell.
"""

import streamlit as st
import os
import json
from src.morning_briefing import (
    generate_morning_briefing_text,
    synthesize_briefing_audio,
    BRIEFING_AUDIO_PATH,
    BRIEFING_JSON_PATH,
)
from src.config import COMPANY_NAMES


def render_morning_briefing_workspace(selected_ticker: str = "NVDA"):
    st.markdown("### 🎙️ AI Pre-Market Morning Audio & Executive Briefing")
    st.caption(
        "Institutional Pre-Market Intelligence Hub: Synthesizes Macroeconomic Regimes, "
        "Multi-Agent Committee Consensus, Overnight News Catalyst Velocity, and Portfolio Risk Posture "
        "into a Broadcast-Quality Audio Briefing & Executive Research Memorandum before the 9:30 AM Bell."
    )

    comp_name = COMPANY_NAMES.get(selected_ticker, selected_ticker)

    # Top Action Control Bar
    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        st.markdown(
            f"""
            <div class="glass-card" style="padding: 14px; margin-bottom: 12px; border-left: 4px solid #38BDF8;">
                <b style="font-size: 1.1rem; color: #F3F4F6;">🎯 Active Target: {selected_ticker} ({comp_name})</b><br>
                <span style="font-size: 0.82rem; color: #94A3B8;">
                    Click below to generate a real-time synthesized morning audio podcast and institutional research memo.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_col2:
        generate_btn = st.button(
            "🎙️ Generate Morning Audio Brief", type="primary", use_container_width=True
        )

    # State Handling
    if generate_btn:
        with st.spinner(
            "🎙️ Synthesizing Pre-Market Multi-Agent Intelligence & Speech Audio..."
        ):
            memo_data = generate_morning_briefing_text(selected_ticker)
            audio_path = synthesize_briefing_audio(memo_data["audio_script"])
            st.session_state["morning_memo"] = memo_data
            st.success(
                "🎙️ Pre-Market Morning Briefing & Audio Podcast Generated Successfully!"
            )
            st.rerun()

    # Load existing or default memo data
    memo = st.session_state.get("morning_memo")
    if not memo and os.path.exists(BRIEFING_JSON_PATH):
        try:
            with open(BRIEFING_JSON_PATH, "r") as f:
                memo = json.load(f)
        except Exception:
            memo = None

    if not memo:
        memo = generate_morning_briefing_text(selected_ticker)

    # =========================================================================
    # AUDIO BROADCAST PLAYER SECTION
    # =========================================================================
    st.markdown("---")
    st.markdown("#### 🎧 Executive Wall Street Audio Broadcast")

    audio_col1, audio_col2 = st.columns([2, 1])
    with audio_col1:
        if os.path.exists(BRIEFING_AUDIO_PATH):
            st.audio(BRIEFING_AUDIO_PATH, format="audio/mp3")
            st.caption(
                "⚡ Synthesized High-Fidelity Audio Stream | Model: Sentilyze AI Speech Engine"
            )
        else:
            st.info(
                "Click 'Generate Morning Audio Brief' above to synthesize today's audio podcast."
            )

    with audio_col2:
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);">
                <div style="font-size: 0.75rem; color: #64748B;">BROADCAST METADATA</div>
                <div style="font-size: 0.85rem; color: #F3F4F6; margin-top: 4px;">● Generated: <b>{memo.get('generated_at', 'Live')}</b></div>
                <div style="font-size: 0.85rem; color: #38BDF8;">● Target: <b>{selected_ticker}</b></div>
                <div style="font-size: 0.85rem; color: #10B981;">● Audio Quality: <b>320kbps HD Audio</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # EXECUTIVE MEMORANDUM CARDS
    # =========================================================================
    st.markdown("---")
    st.markdown(f"#### 📑 {memo.get('headline', 'Pre-Market Intelligence Memo')}")

    # Top Key Indicator Matrix
    macro_info = memo.get("macro_posture", {})
    focus_info = memo.get("focus_asset", {})
    port_info = memo.get("portfolio_status", {})

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "🌐 Volatility Regime",
        macro_info.get("regime", "BULLISH").replace("_", " ").title(),
        delta=f"VIX: {macro_info.get('vix_level', 15.4):.1f}",
    )
    m2.metric(
        "🏛️ Committee Stance",
        focus_info.get("committee_decision", "APPROVED"),
        delta=f"{focus_info.get('confidence_pct', 72):.0f}% Conviction",
    )
    m3.metric(
        "📈 10-Yr Benchmark Yield",
        macro_info.get("10y_treasury", "4.25%"),
        delta="Treasury Curve",
    )
    m4.metric(
        "💼 Capital Ready",
        f"${port_info.get('cash_reserves', 100000):,.0f}",
        delta=f"{port_info.get('open_positions', 1)} Active Holdings",
    )

    # Executive Summary Glass Card
    st.markdown(
        f"""
        <div class="glass-card" style="padding: 16px; margin: 12px 0; border-left: 4px solid #10B981; background: rgba(16, 185, 129, 0.04);">
            <b style="font-size: 1.05rem; color: #10B981;">📋 Executive Overview:</b><br>
            <div style="font-size: 0.92rem; color: #E2E8F0; line-height: 1.6; margin-top: 6px;">
                {memo.get('executive_summary', '')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Spoken Audio Script Transcript Expander
    with st.expander("📜 View Full Spoken Audio Script & Transcript", expanded=False):
        st.markdown(f"*{memo.get('audio_script', '')}*")
