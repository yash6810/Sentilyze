"""
Workspace 4: Systematic 4-Station 1-Day-Prior Reddit News & SEC S-1 Pre-IPO Radar.
"""

import streamlit as st
import pandas as pd
from src.ui.components import render_workspace_header, render_conviction_gauge
from src.reddit_premarket_station import fetch_4station_premarket_intelligence
from src.ipo_radar import (
    get_pre_ipo_pipeline_df,
    fetch_sec_edgar_ipo_filings,
    auto_register_ipo_ticker,
)


def render_alternative_data_workspace(selected_ticker: str):
    """Renders the 4-Station Reddit Intelligence and Pre-IPO Valuation Radar."""
    render_workspace_header(
        title=f"📡 Alternative Data & 1-Day-Prior Intelligence ({selected_ticker})",
        subtitle="Systematic 4-Station Reddit Intelligence (WSB, Stocks, Options, Daytrading) + SEC S-1 Pre-IPO Radar",
        badge_text="ALTERNATIVE ALPHA",
        badge_color="#F59E0B",
    )

    t1, t2 = st.tabs(
        [
            "🔥 Systematic 4-Station Reddit Intelligence",
            "🦄 Pre-IPO & SEC S-1 Registration Radar",
        ]
    )

    with t1:
        st.markdown("### 📡 Systematic 4-Station 1-Day-Prior Multi-Feed Radar")
        st.markdown(
            """
            <div class="glass-card">
                <b>1-Day-Prior Multi-Station Consensus:</b> Synthesizes overnight sentiment across 
                <b>r/wallstreetbets (35%)</b>, <b>r/stocks (25%)</b>, <b>r/options (20%)</b>, and <b>r/Daytrading (20%)</b> 
                to gauge retail momentum and detect contrarian exhaustion traps before the 09:30 AM opening bell.
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.spinner(f"Scraping 4 Reddit market stations for {selected_ticker}..."):
            intel = fetch_4station_premarket_intelligence(selected_ticker)

        # Top Composite Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "🎯 1-Day-Prior Composite Score",
            f"{intel['composite_score']:+.3f}",
            delta=f"{intel['composite_conviction_pct']:.1f}% Conviction",
        )
        c2.metric(
            "🏛️ Station Consensus Count",
            f"{intel['positive_stations_count']} / 4 Stations Bullish",
        )
        c3.metric("🏷️ Retail Flow Regime", intel["regime_code"])

        render_conviction_gauge(
            intel["composite_conviction_pct"],
            label=f"4-STATION COMPOSITE CONVICTION ({selected_ticker})",
        )

        # 4 Station Cards Grid
        st.markdown("#### 📊 Station-by-Station Deliberation Breakdown")
        grid_c1, grid_c2 = st.columns(2)

        for idx, st_item in enumerate(intel["stations"]):
            target_col = grid_c1 if idx % 2 == 0 else grid_c2
            with target_col:
                score_col = "#10B981" if st_item["normalized_score"] > 0 else "#EF4444"
                threads_html = (
                    "".join(
                        [
                            f"<li><a href='{t['url']}' target='_blank' style='color: #F3F4F6; text-decoration: none;'>{t['title'][:70]}...</a></li>"
                            for t in st_item.get("threads", [])
                        ]
                    )
                    or "<li>No specific ticker threads detected.</li>"
                )

                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h4 style="margin: 0; font-size: 1rem;">{st_item['station_name']}</h4>
                            <span style="color: {score_col}; font-weight: 800; font-family: 'JetBrains Mono', monospace;">
                                {st_item['normalized_score']:+.3f} ({st_item['bullish_pct']:.1f}% Bull)
                            </span>
                        </div>
                        <p style="margin: 4px 0 8px 0; color: #94A3B8; font-size: 0.8rem;"><b>Cadence:</b> {st_item['cadence']} | <b>Weight:</b> {st_item['weight']*100:.0f}%</p>
                        <ul style="margin: 0; padding-left: 18px; font-size: 0.85rem; color: #CBD5E1;">
                            {threads_html}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with t2:
        st.markdown("### 🦄 Pre-IPO Intelligence & SEC EDGAR S-1 Filings")
        st.markdown(
            """
            <div class="glass-card">
                <b>Curated Private Market Intelligence:</b> Profiles late-stage private companies (OpenAI, Anthropic, SpaceX, Stripe, Databricks) 
                with verified venture rounds, lead backers, and estimated secondary valuations alongside a live <b>SEC EDGAR Form S-1 / S-1/A</b> filing stream.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 🏛️ Late-Stage Private Enterprise Directory")
        ipo_df = get_pre_ipo_pipeline_df()
        st.dataframe(
            ipo_df.style.format(
                {
                    "IPO Readiness (%)": "{:.1f}%",
                }
            ),
            use_container_width=True,
        )

        st.markdown("#### 📜 Live SEC EDGAR Form S-1 IPO Filings Feed")
        with st.spinner("Fetching live SEC EDGAR Form S-1 filings..."):
            sec_filings = fetch_sec_edgar_ipo_filings()

        if sec_filings:
            filings_df = pd.DataFrame(sec_filings)
            st.dataframe(filings_df, use_container_width=True)
        else:
            st.info(
                "ℹ️ No new SEC Form S-1 filings detected in the current EDGAR polling cycle."
            )

        st.markdown("#### ⚡ Register Upcoming IPO Ticker into Model Universe")
        r_col1, r_col2, r_col3 = st.columns([1.5, 2, 1])
        with r_col1:
            new_ticker = st.text_input("Projected Ticker", value="OPENAI").upper()
        with r_col2:
            new_name = st.text_input("Company Name", value="OpenAI, Inc.")
        with r_col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ Auto-Register Ticker", use_container_width=True):
                success = auto_register_ipo_ticker(new_ticker, new_name)
                if success:
                    st.success(
                        f"✅ {new_ticker} successfully registered into `stocks.txt`!"
                    )
                else:
                    st.info(f"ℹ️ {new_ticker} registration complete or already present.")
