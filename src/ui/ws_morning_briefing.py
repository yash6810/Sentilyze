"""
Workspace: AI Pre-Market Audio & Executive Morning Intelligence Briefing.
Generates institutional pre-market podcast audio, Wall Street research memoranda,
and catalyst action matrices across the entire S&P 500 universe before the 9:30 AM opening bell.
"""

import streamlit as st
import os
import json
from src.morning_briefing import (
    generate_morning_briefing_text,
    synthesize_briefing_audio,
    VOICE_PROFILES,
    BRIEFING_AUDIO_PATH,
    BRIEFING_JSON_PATH,
)
from src.config import COMPANY_NAMES


def render_morning_briefing_workspace(selected_ticker: str = "NVDA"):
    st.markdown("### 🎙️ AI Pre-Market Morning Audio & Executive Podcast Generator")
    st.caption(
        "Institutional Morning Intelligence Broadcast: Synthesizes Macroeconomic Regimes, "
        "Universe-wide Alpha Scans across stocks.txt, Pre/Post-Market News Catalysts, and Live Portfolio Cash Reserves "
        "into an Audio Podcast & Wall Street Research Memorandum before the 9:30 AM Opening Bell."
    )

    comp_name = COMPANY_NAMES.get(selected_ticker, selected_ticker)

    # Broadcast Mode & Voice Selector Grid
    b1, b2, b3 = st.columns([2, 2, 1])
    with b1:
        broadcast_mode = st.selectbox(
            "📻 Broadcast Episode Mode:",
            [
                "📻 Full Market Master Podcast (Macro + Top Stocks + Portfolio)",
                "🚀 Top Alpha Stocks in Play (Universe Momentum Scan)",
                "💼 Portfolio Holdings & Capital Radar ($152k Cash Update)",
                f"🔍 Single Ticker Deep-Dive ({selected_ticker})",
            ],
            index=0,
        )
    with b2:
        voice_options = {k: v["name"] for k, v in VOICE_PROFILES.items()}
        selected_voice_key = st.selectbox(
            "🎙️ Anchor Voice & Regional Accent:",
            options=list(voice_options.keys()),
            format_func=lambda x: voice_options[x],
            index=0,
            help="Select the AI radio anchor persona, pronunciation, and regional market desk accent.",
        )
    with b3:
        speech_pace = st.selectbox(
            "⏱️ Pacing:",
            ["Standard (1.0x)", "Deliberate (0.85x)"],
            index=0,
        )
        is_slow = "Deliberate" in speech_pace

    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
    generate_btn = st.button(
        "🎙️ Synthesize & Broadcast Morning Audio Brief",
        type="primary",
        use_container_width=True,
    )

    # Map selected mode
    mode_key = (
        "MARKET_MASTER"
        if "Full Market Master" in broadcast_mode
        else (
            "TOP_STOCKS"
            if "Top Alpha Stocks" in broadcast_mode
            else (
                "PORTFOLIO_RADAR"
                if "Portfolio Holdings" in broadcast_mode
                else "SINGLE_TICKER"
            )
        )
    )

    # State Handling
    if generate_btn:
        with st.spinner("🎙️ Synthesizing Multi-Segment Wall Street Audio Broadcast..."):
            memo_data = generate_morning_briefing_text(
                mode=mode_key, ticker=selected_ticker
            )
            synthesize_briefing_audio(
                memo_data["audio_script"],
                voice_key=selected_voice_key,
                slow=is_slow,
            )
            memo_data["active_voice"] = VOICE_PROFILES[selected_voice_key]["name"]
            st.session_state["morning_memo"] = memo_data
            st.success(
                f"🎙️ Morning Podcast Generated with {VOICE_PROFILES[selected_voice_key]['name']}!"
            )

            st.rerun()

    # Load existing or default memo data
    memo = st.session_state.get("morning_memo")
    if not memo and os.path.exists(BRIEFING_JSON_PATH):
        try:
            with open(BRIEFING_JSON_PATH, "r", encoding="utf-8") as f:
                memo = json.load(f)
        except Exception:
            memo = None

    if not memo:
        memo = generate_morning_briefing_text(mode=mode_key, ticker=selected_ticker)

    # =========================================================================
    # AUDIO BROADCAST PLAYER SECTION
    # =========================================================================
    st.markdown("---")
    st.markdown("#### 🎧 Executive Morning Audio Broadcast Player")

    audio_col1, audio_col2 = st.columns([2, 1])
    with audio_col1:
        if os.path.exists(BRIEFING_AUDIO_PATH):
            st.audio(BRIEFING_AUDIO_PATH, format="audio/mp3")
            with open(BRIEFING_AUDIO_PATH, "rb") as af:
                st.download_button(
                    label="⬇️ Download Morning Podcast (.mp3)",
                    data=af.read(),
                    file_name="sentilyze_morning_briefing.mp3",
                    mime="audio/mp3",
                )
        else:
            st.info(
                "Click 'Synthesize & Broadcast Morning Audio Brief' to generate today's podcast."
            )

    with audio_col2:
        mode_label = memo.get("mode", "MARKET_MASTER").replace("_", " ").title()
        active_voice_name = memo.get("active_voice", "🇺🇸 US Financial Anchor")
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);">
                <div style="font-size: 0.75rem; color: #64748B;">BROADCAST METADATA</div>
                <div style="font-size: 0.85rem; color: #F3F4F6; margin-top: 4px;">● Generated: <b>{memo.get('generated_at', 'Live Pre-Market')}</b></div>
                <div style="font-size: 0.85rem; color: #38BDF8;">● Format: <b>{mode_label}</b></div>
                <div style="font-size: 0.85rem; color: #F59E0B;">● Anchor Voice: <b>{active_voice_name}</b></div>
                <div style="font-size: 0.85rem; color: #10B981;">● Audio Quality: <b>320kbps HD Audio</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # EXECUTIVE MEMORANDUM & TOP PICKS
    # =========================================================================
    st.markdown("---")
    st.markdown(f"#### 📑 {memo.get('headline', 'Pre-Market Intelligence Memo')}")

    # Top Key Macro & Portfolio Matrix
    macro_info = memo.get("macro_posture", {})
    port_info = memo.get("portfolio_status", {})
    primary_info = memo.get("primary_focus", {})

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "🌐 Volatility Regime",
        macro_info.get("regime", "BULLISH").replace("_", " ").title(),
        delta=f"VIX: {macro_info.get('vix_level', 16.5):.1f}",
    )
    m2.metric(
        "🏛️ Committee Stance",
        primary_info.get("committee_decision", "APPROVED"),
        delta=f"{primary_info.get('confidence_pct', 78):.0f}% Conviction",
    )
    m3.metric(
        "📈 10-Yr Benchmark Yield",
        macro_info.get("10y_treasury", "4.25%"),
        delta=f"Fed Liq: {macro_info.get('fed_liquidity', '$6.05 T')}",
    )
    m4.metric(
        "💼 Cash Reserves",
        f"${port_info.get('cash_reserves', 152198.09):,.2f}",
        delta=f"{port_info.get('win_rate_pct', 89.66):.1f}% Win Rate",
    )

    # Executive Overview Box
    st.markdown(
        f"""
        <div class="glass-card" style="padding: 16px; margin: 12px 0; border-left: 4px solid #10B981; background: rgba(16, 185, 129, 0.04);">
            <b style="font-size: 1.05rem; color: #10B981;">📋 Executive Morning Summary:</b><br>
            <div style="font-size: 0.92rem; color: #E2E8F0; line-height: 1.6; margin-top: 6px;">
                {memo.get('executive_summary', '')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Alpha Stocks in Play Cards
    top_stocks = memo.get("top_stocks_in_play", [])
    if top_stocks:
        st.markdown("#### 🏆 Top Algorithmic Alpha Picks for Today (`stocks.txt` Scan)")
        t_cols = st.columns(len(top_stocks))
        for idx, (col, stk) in enumerate(zip(t_cols, top_stocks)):
            with col:
                st.markdown(
                    f"""
                    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 10px; padding: 14px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <b style="font-size: 1.15rem; color: #38BDF8;">#{idx+1} {stk.get('ticker')}</b>
                            <span style="background: rgba(16, 185, 129, 0.2); color: #10B981; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: bold;">
                                {stk.get('conviction_pct', 80):.0f}% Conviction
                            </span>
                        </div>
                        <div style="font-size: 1.1rem; font-weight: bold; color: #F3F4F6; margin: 6px 0;">
                            ${stk.get('last_price', 100):,.2f} 
                            <span style="font-size: 0.8rem; color: {'#10B981' if stk.get('day_change_pct', 0) >= 0 else '#EF4444'};">
                                ({stk.get('day_change_pct', 0):+.2f}%)
                            </span>
                        </div>
                        <hr style="margin: 6px 0; border-color: rgba(255,255,255,0.06);">
                        <div style="font-size: 0.78rem; color: #94A3B8;">
                            🎯 <b>TP1 (+2.5 ATR):</b> ${stk.get('tp1_target', 0):,.2f}<br>
                            🏆 <b>TP2 Runner (+4.5 ATR):</b> ${stk.get('tp2_target', 0):,.2f}<br>
                            🛡️ <b>Stop-Loss (-1.5 ATR):</b> ${stk.get('sl_target', 0):,.2f}
                        </div>
                        <div style="font-size: 0.75rem; color: #CBD5E1; margin-top: 8px; font-style: italic;">
                            📰 <i>"{stk.get('headline_catalyst', '')}"</i>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Full Spoken Script Expander
    with st.expander("📜 View Complete Spoken Podcast Transcript", expanded=False):
        st.markdown(f"*{memo.get('audio_script', '')}*")
