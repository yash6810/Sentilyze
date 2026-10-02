"""
Workspace: Benzinga Live Catalyst Wire & Autonomous News Trader.
Scrapes breaking Benzinga headlines, extracts active tickers, evaluates FinBERT sentiment,
and routes high-conviction catalyst trades through the Sentilyze Multi-Agent Committee.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from src.benzinga_news_trader import (
    scan_and_trade_benzinga_catalysts,
    get_latest_benzinga_scan_report,
)
from src.ui.components import render_workspace_header


def render_benzinga_trader_workspace():
    render_workspace_header(
        title="⚡ Benzinga Breaking News Catalyst Trader",
        subtitle="Real-time Benzinga wire ingestion, ticker extraction, FinBERT sentiment scoring & multi-agent autonomous order routing.",
        badge_text="LIVE BENZINGA CATALYST WIRE",
        badge_color="#F59E0B",
    )

    st.markdown(
        "Monitors breaking financial headlines directly from the **Benzinga Financial Wire**. "
        "When material catalysts break (earnings beats, multi-billion dollar contracts, analyst upgrades, regulatory probes), "
        "the **Multi-Agent Trading Council** convenes to evaluate technical momentum and execute buy/sell orders."
    )

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        max_eval = st.slider(
            "Max Benzinga Stocks to Deliberate",
            min_value=3,
            max_value=15,
            value=6,
            step=1,
        )
    with c2:
        st.write("")
        st.write("")
        scan_btn = st.button(
            "📡 Scan Live Benzinga Wire", type="primary", use_container_width=True
        )
    with c3:
        st.write("")
        st.write("")
        exec_btn = st.button("🚀 Scan & Auto-Execute Trades", use_container_width=True)

    if scan_btn or exec_btn:
        exec_trades = bool(exec_btn)
        action_msg = (
            "Scanning and executing live paper trades..."
            if exec_trades
            else "Scanning live Benzinga headlines..."
        )
        with st.spinner(action_msg):
            try:
                rep = scan_and_trade_benzinga_catalysts(
                    execute_paper=exec_trades,
                    max_stocks_to_evaluate=max_eval,
                )
                st.success(
                    f"Successfully processed {rep.get('total_articles_scraped', 0)} Benzinga articles and {rep.get('total_tickers_identified', 0)} active tickers!"
                )
            except Exception as e:
                st.error(f"Benzinga scan notice: {e}")

    report = get_latest_benzinga_scan_report()

    if not report:
        st.info(
            "No cached Benzinga news scan found. Click 'Scan Live Benzinga Wire' to fetch the latest breaking headlines."
        )
        return

    # Header KPI Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📰 Articles Scraped", f"{report.get('total_articles_scraped', 0)}")
    m2.metric("🎯 Tickers Identified", f"{report.get('total_tickers_identified', 0)}")
    m3.metric("💼 Portfolio Cash", f"${report.get('portfolio_cash', 0.0):,.2f}")
    m4.metric("💰 Total Equity", f"${report.get('portfolio_equity', 0.0):,.2f}")

    st.markdown("---")
    st.subheader("🔥 Top Stocks with Breaking Benzinga Catalysts")

    stocks = report.get("top_evaluated_stocks", [])
    if stocks:
        df_display = []
        for s in stocks:
            sentiment_color = (
                "🟢 BULLISH"
                if s["benzinga_sentiment"] == "BULLISH"
                else (
                    "🔴 BEARISH"
                    if s["benzinga_sentiment"] == "BEARISH"
                    else "⚪ NEUTRAL"
                )
            )
            df_display.append(
                {
                    "Ticker": s["ticker"],
                    "Spot Price": s["current_price"],
                    "24h Change": s["change_pct"],
                    "Benzinga Sentiment": sentiment_color,
                    "Catalyst Score": s["net_catalyst_score"],
                    "Headlines": s["headline_count"],
                    "Council Verdict": s["committee_verdict"],
                    "Conviction": s["committee_conviction_pct"],
                    "Trade Action": s["trade_action"],
                    "Latest Benzinga Headline": s["latest_headline"],
                }
            )

        st.dataframe(
            pd.DataFrame(df_display),
            use_container_width=True,
            column_config={
                "Spot Price": st.column_config.NumberColumn(format="$%.2f"),
                "24h Change": st.column_config.NumberColumn(format="%+.2f%%"),
                "Catalyst Score": st.column_config.NumberColumn(format="%+.2f"),
                "Conviction": st.column_config.NumberColumn(format="%.1f%%"),
                "Trade Action": st.column_config.TextColumn("Trade Action"),
            },
            hide_index=True,
        )

        st.markdown("#### 📋 Detailed Benzinga Catalyst Theses")
        for s in stocks:
            border_color = (
                "#10B981"
                if s["benzinga_sentiment"] == "BULLISH"
                else ("#EF4444" if s["benzinga_sentiment"] == "BEARISH" else "#64748B")
            )
            with st.container():
                st.markdown(
                    f"""
                    <div style="border-left: 4px solid {border_color}; padding-left: 14px; margin-bottom: 12px; background: rgba(255,255,255,0.02); border-radius: 4px; padding: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 700; font-size: 1.1rem; color: #FFFFFF;">
                                {s['ticker']} — ${s['current_price']:.2f} ({s['change_pct']:+.2f}%)
                            </span>
                            <span style="font-size: 0.85rem; font-weight: 600; padding: 3px 8px; border-radius: 4px; background: rgba(255,255,255,0.08);">
                                Council: {s['committee_verdict']} ({s['committee_conviction_pct']:.1f}% Conviction) | Action: {s['trade_action']}
                            </span>
                        </div>
                        <p style="margin: 6px 0 0 0; color: #E2E8F0; font-size: 0.95rem;">
                            📢 <b>Benzinga Headline:</b> {s['latest_headline']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    actions = report.get("trade_actions_executed", [])
    if actions:
        st.markdown("---")
        st.subheader("⚡ Executed Autonomous Trades on Benzinga News")
        st.json(actions)

    st.caption(f"Last Benzinga Wire Scan: {report.get('scan_timestamp', 'N/A')}")
