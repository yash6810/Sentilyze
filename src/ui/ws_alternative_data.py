"""
Workspace 4: Systematic 9-Station 1-Day-Prior Reddit News & SEC S-1 Pre-IPO Radar.
"""

import streamlit as st
import pandas as pd
from src.ui.components import render_workspace_header, render_conviction_gauge
from src.reddit_premarket_station import fetch_8station_premarket_intelligence
from src.ipo_radar import (
    get_pre_ipo_pipeline_df,
    fetch_sec_edgar_ipo_filings,
    auto_register_ipo_ticker,
)


@st.cache_data(ttl=300)
def _get_cached_reddit_intel(ticker: str):
    return fetch_8station_premarket_intelligence(ticker)


def render_alternative_data_workspace(selected_ticker: str):
    """Renders the 9-Station Reddit Intelligence and Pre-IPO Valuation Radar."""
    render_workspace_header(
        title=f"📡 Alternative Data & 1-Day-Prior Intelligence ({selected_ticker})",
        subtitle="Systematic 9-Station Reddit Intelligence (Unusual Whales, WSB, Stocks, Options, Daytrading, Investing, ValueInvesting, Economics, AlgoTrading) + SEC S-1 Pre-IPO Radar",
        badge_text="ALTERNATIVE ALPHA",
        badge_color="#F59E0B",
    )

    t1, t2, t3 = st.tabs(
        [
            "🔥 Systematic 9-Station Reddit Intelligence",
            "🦄 Pre-IPO & SEC S-1 Registration Radar",
            "🌐 Universal Web Scraper & Intelligence Extractor",
        ]
    )

    with t1:
        st.markdown("### 📡 Systematic 9-Station Multi-Channel Market Radar")
        st.markdown(
            """
            <div class="glass-card">
                <b>9-Station Multi-Channel Market Consensus:</b> Synthesizes real-time flow and sentiment across 
                <b>r/unusual_whales (15%)</b>, <b>r/wallstreetbets (15%)</b>, <b>r/stocks (15%)</b>, <b>r/options (15%)</b>, 
                <b>r/Daytrading (10%)</b>, <b>r/investing (10%)</b>, <b>r/ValueInvesting (10%)</b>, <b>r/Economics (5%)</b>, and <b>r/algotrading (5%)</b> 
                to track institutional dark pool block orders, retail momentum, gamma squeezes, and macro liquidity before and during market sessions.
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.spinner(
            f"Scraping 9 Reddit financial market stations for {selected_ticker}..."
        ):
            intel = _get_cached_reddit_intel(selected_ticker)

        # Top Composite Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "🎯 9-Station Composite Score",
            f"{intel['composite_score']:+.3f}",
            delta=f"{intel['composite_conviction_pct']:.1f}% Conviction",
        )
        c2.metric(
            "🏛️ Station Consensus Count",
            f"{intel['positive_stations_count']} / {intel.get('total_stations_count', 9)} Stations Bullish",
        )
        c3.metric("🏷️ Social Regime", intel["regime_code"])

        render_conviction_gauge(
            intel["composite_conviction_pct"],
            label=f"9-STATION COMPOSITE CONVICTION ({selected_ticker})",
        )

        # 8 Station Cards Grid
        st.markdown("#### 📊 Multi-Channel Deliberation Breakdown")
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
                        <p style="margin: 4px 0 8px 0; color: #94A3B8; font-size: 0.8rem;"><b>Role:</b> {st_item.get('role', 'Market Intelligence')} | <b>Weight:</b> {st_item['weight']*100:.0f}%</p>
                        <p style="margin: 0 0 6px 0; color: #64748B; font-size: 0.75rem;"><b>Cadence:</b> {st_item['cadence']}</p>
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

    with t3:
        st.markdown("### 🌐 Universal Web Scraper & Intelligence Extractor")
        st.markdown(
            """
            <div class="glass-card">
                <b>Adaptive Universal Scraper:</b> Powered by <b>Scrapling</b> (adaptive anti-bot bypass), 
                <b>Trafilatura</b> (academic-grade boilerplate & ad removal), <b>BeautifulSoup4</b> (table extraction), 
                and <b>FinBERT NLP</b>. Scrapes, cleans, parses financial tables, and scores sentiment from <b>any web URL</b> 
                or directly queries live web hubs (Finviz, SEC EDGAR 8-K) when API feeds are unavailable.
            </div>
            """,
            unsafe_allow_html=True,
        )

        from src.universal_web_scraper import scrape_url, scrape_ticker_news_from_web

        scraper_mode = st.radio(
            "Scraper Target Mode:",
            [
                "🔗 Deep Web Page URL Scraper",
                "⚡ Live Ticker Web News & SEC 8-K Multi-Hub",
            ],
            horizontal=True,
        )

        if scraper_mode == "⚡ Live Ticker Web News & SEC 8-K Multi-Hub":
            tk_col1, tk_col2, tk_col3 = st.columns([2, 1, 1])
            with tk_col1:
                scrape_ticker_input = (
                    st.text_input(
                        "Enter Stock Ticker to Scrape Live Web Feeds:",
                        value="NVDA",
                        placeholder="e.g. NVDA, PLTR, AMD, TSLA, AAPL",
                    )
                    .upper()
                    .strip()
                )
            with tk_col2:
                scrape_limit = st.slider(
                    "Max Headlines", min_value=5, max_value=40, value=15
                )
            with tk_col3:
                st.markdown("<br>", unsafe_allow_html=True)
                trigger_tk_scrape = st.button(
                    "⚡ Scrape Live Web Hubs", use_container_width=True
                )

            if trigger_tk_scrape:
                with st.spinner(
                    f"Scraping Finviz news table & SEC EDGAR 8-K Atom feeds for ${scrape_ticker_input}..."
                ):
                    live_news_df = scrape_ticker_news_from_web(
                        scrape_ticker_input, limit=scrape_limit
                    )

                if live_news_df.empty:
                    st.warning(
                        f"⚠️ No web headlines currently found for ${scrape_ticker_input}."
                    )
                else:
                    st.success(
                        f"✅ Successfully scraped **{len(live_news_df)} live headlines** from Finviz and SEC EDGAR!"
                    )

                    # Compute quick FinBERT sentiment summary across scraped titles
                    from src.universal_web_scraper import _score_text_with_finbert

                    scores = []
                    for title in live_news_df["Title"].tolist()[:10]:
                        s_res = _score_text_with_finbert(title)
                        scores.append(s_res.get("sentiment_score", 0.0))

                    avg_score = sum(scores) / len(scores) if scores else 0.0

                    nm1, nm2, nm3 = st.columns(3)
                    nm1.metric("📰 Scraped Headlines", f"{len(live_news_df)} articles")
                    nm2.metric(
                        "🏛️ SEC 8-K Events",
                        f"{sum(1 for s in live_news_df['source'] if s.get('name') == 'SEC EDGAR 8-K')} filings",
                    )
                    nm3.metric(
                        "🧠 Aggregate FinBERT Sentiment",
                        f"{avg_score:+.3f}",
                        delta=(
                            "Bullish"
                            if avg_score > 0.05
                            else ("Bearish" if avg_score < -0.05 else "Neutral")
                        ),
                    )

                    st.markdown("#### 📋 Scraped Articles & Real-Time Disclosures")

                    for _, row in live_news_df.iterrows():
                        src_name = row["source"].get("name", "Web Portal")
                        pub_time = (
                            row["publishedAt"].strftime("%Y-%m-%d %H:%M UTC")
                            if hasattr(row["publishedAt"], "strftime")
                            else str(row["publishedAt"])
                        )
                        badge_color = "#8B5CF6" if "SEC" in src_name else "#0284C7"

                        st.markdown(
                            f"""
                            <div class="glass-card" style="margin-bottom: 8px; padding: 12px;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="background: {badge_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold;">{src_name}</span>
                                    <span style="color: #94A3B8; font-size: 12px;">🕒 {pub_time}</span>
                                </div>
                                <div style="margin-top: 6px; font-size: 14px; font-weight: 600;">
                                    <a href="{row.get('url', '#')}" target="_blank" style="color: #E2E8F0; text-decoration: none;">{row.get('Title')}</a>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        else:
            scrape_col1, scrape_col2 = st.columns([3, 1])
            with scrape_col1:
                target_url = st.text_input(
                    "🔗 Enter Any Web URL to Scrape & Analyze",
                    value="https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-third-quarter-fiscal-2025",
                    placeholder="https://example.com/press-release-or-article",
                )
            with scrape_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                trigger_scrape = st.button(
                    "🚀 Scrape & Analyze Web Page", use_container_width=True
                )

            # Preset Quick Pickers
            st.markdown("##### ⚡ Quick Test Presets:")
            p_c1, p_c2, p_c3, p_c4 = st.columns(4)
            preset_url = None
            if p_c1.button("🟢 Nvidia Newsroom"):
                preset_url = "https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-third-quarter-fiscal-2025"
            if p_c2.button("🔵 Microsoft Press"):
                preset_url = "https://blogs.microsoft.com/blog/2025/01/22/transforming-work-with-ai-agents/"
            if p_c3.button("🟣 SEC EDGAR 8-K"):
                preset_url = "https://www.sec.gov/edgar/searchedgar/companysearch"
            if p_c4.button("🟡 Yahoo Market News"):
                preset_url = "https://finance.yahoo.com/news/"

        active_scrape_url = preset_url if preset_url else target_url

        if trigger_scrape or preset_url:
            with st.spinner(f"Scraping and analyzing {active_scrape_url}..."):
                scrape_res = scrape_url(
                    active_scrape_url, extract_tables=True, score_sentiment=True
                )

            if not scrape_res.get("success"):
                st.error(
                    f"❌ Scraping Failed: {scrape_res.get('error', 'Unknown Error')}"
                )
            else:
                st.success(
                    f"✅ Scraped Successfully via **{scrape_res.get('engine_used', 'Hybrid Engine')}** in real time!"
                )

                # Top Scraped KPI Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("📖 Word Count", f"{scrape_res.get('word_count', 0):,} words")
                m2.metric(
                    "⏱️ Est. Read Time",
                    f"{scrape_res.get('reading_time_mins', 1)} mins",
                )
                m3.metric(
                    "📊 Tables Extracted", f"{scrape_res.get('tables_count', 0)} tables"
                )

                sent_data = scrape_res.get("sentiment", {})
                sent_lbl = sent_data.get("sentiment_label", "neutral").upper()
                sent_score = sent_data.get("sentiment_score", 0.0)
                sent_delta = (
                    f"{sent_data.get('sentiment_confidence', 0.5)*100:.1f}% Conf"
                )
                m4.metric(
                    f"🧠 FinBERT: {sent_lbl}", f"{sent_score:+.3f}", delta=sent_delta
                )

                # Detected Tickers Banner
                tickers = scrape_res.get("detected_tickers", [])
                if tickers:
                    ticker_badges = " ".join(
                        [
                            f"<span style='background: #1E293B; color: #38BDF8; padding: 3px 10px; border-radius: 6px; font-weight: 700; margin-right: 6px; border: 1px solid #0284C7;'>${t}</span>"
                            for t in tickers
                        ]
                    )
                    st.markdown(
                        f"""
                        <div class="glass-card" style="margin: 12px 0;">
                            <b>🏷️ Detected Stock Tickers:</b> {ticker_badges}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Sub Tabs for Extracted Data
                sub_t1, sub_t2, sub_t3, sub_t4 = st.tabs(
                    [
                        "📝 Clean Narrative & Summary",
                        "📊 Extracted Financial Tables",
                        "🔗 Outbound Links & Metadata",
                        "🔍 Raw JSON Inspection",
                    ]
                )

                with sub_t1:
                    st.markdown(f"### {scrape_res.get('title', 'Page Title')}")
                    st.markdown(
                        f"**Author / Source:** {scrape_res.get('author')} | **Published:** {scrape_res.get('published_at')}"
                    )

                    if scrape_res.get("summary"):
                        st.markdown(
                            f"""
                            <div class="glass-card" style="background: rgba(30, 41, 59, 0.7); border-left: 4px solid #10B981; margin: 12px 0;">
                                <b>💡 Executive Narrative Summary:</b><br>{scrape_res['summary']}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("#### Full Cleaned Article Body")
                    st.text_area(
                        "Extracted Text",
                        value=scrape_res.get("clean_text", ""),
                        height=350,
                    )

                with sub_t2:
                    tables = scrape_res.get("tables", [])
                    if tables:
                        st.markdown(
                            f"#### 📊 Extracted Financial Tables ({len(tables)} found)"
                        )
                        for t_idx, tbl_df in enumerate(tables):
                            st.markdown(f"**Table #{t_idx+1}**")
                            st.dataframe(tbl_df, use_container_width=True)
                    else:
                        st.info("ℹ️ No HTML tabular data found on this specific page.")

                with sub_t3:
                    links = scrape_res.get("outbound_links", [])
                    if links:
                        st.markdown(f"#### 🔗 Extracted Outbound Links ({len(links)})")
                        links_df = pd.DataFrame(links)
                        st.dataframe(links_df, use_container_width=True)
                    else:
                        st.info("ℹ️ No outbound hyperlinks extracted.")

                with sub_t4:
                    st.json(
                        {
                            k: v
                            for k, v in scrape_res.items()
                            if k not in ["tables", "clean_text"]
                        }
                    )
