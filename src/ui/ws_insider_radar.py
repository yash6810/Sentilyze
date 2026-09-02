"""
Workspace: Smart-Money Executive & Institutional Insider Radar.
SEC Form 4 Open-Market Ingestion, C-Suite Cluster Buys & Conviction Scoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

from src.insider_signals import (
    calculate_insider_conviction_score,
    scan_universe_insider_catalysts,
    fetch_insider_transactions,
)
from src.paper_broker import PaperBroker
from src.config import COMPANY_NAMES


def render_insider_radar_workspace(selected_ticker: str = "NVDA"):
    st.markdown("### 🏛️ Smart-Money Executive & Institutional Insider Radar")
    st.caption(
        "Institutional SEC Form 4 Capital Flow Radar: Detects CEO/CFO Open-Market Purchases, "
        "Flags Multi-Officer Cluster Buys ($250k+), and Computes Insider Conviction Index (0 to 100)."
    )

    tab_overview, tab_scanner = st.tabs(
        [
            f"🎯 Deep Dive on Active Asset ({selected_ticker})",
            "📡 Universe Insider Catalyst Scanner",
        ]
    )

    # =========================================================================
    # TAB 1: INDIVIDUAL ASSET DEEP DIVE
    # =========================================================================
    with tab_overview:
        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            st.markdown(
                f"#### SEC Form 4 Capital Flow Telemetry: **{selected_ticker}** ({COMPANY_NAMES.get(selected_ticker, selected_ticker)})"
            )
        with col_t2:
            days_lookback = st.selectbox("Lookback Window:", [30, 60, 90, 180], index=2)

        with st.spinner(f"Ingesting SEC Form 4 filings for {selected_ticker}..."):
            insider_data = calculate_insider_conviction_score(
                selected_ticker, days_back=days_lookback
            )

        score = float(insider_data.get("conviction_score", 50.0))
        signal = insider_data.get("signal", "NEUTRAL")
        cluster_flag = insider_data.get("cluster_buy_detected", False)
        total_buy = float(insider_data.get("total_buy_usd", 0.0))
        total_sale = float(insider_data.get("total_sale_usd", 0.0))
        net_flow = float(insider_data.get("net_purchased_usd", 0.0))

        score_color = (
            "#10B981" if score >= 70 else "#EF4444" if score <= 35 else "#38BDF8"
        )

        # Executive Card Banner
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
                        border: 1px solid {score_color}; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <span style="font-size: 13px; letter-spacing: 2px; text-transform: uppercase; color: #94A3B8;">
                            QUANTITATIVE INSIDER CONVICTION INDEX
                        </span>
                        <div style="font-size: 26px; font-weight: 800; color: #FFFFFF; margin-top: 4px;">
                            {signal}
                        </div>
                        <div style="font-size: 14px; color: #CBD5E1; margin-top: 6px;">
                            {insider_data.get('summary', '')}
                        </div>
                    </div>
                    <div style="background: {score_color}22; border: 2px solid {score_color}; border-radius: 16px;
                                padding: 12px 28px; text-align: center;">
                        <div style="font-size: 38px; font-weight: 900; color: {score_color}; line-height: 1;">
                            {score:.0f} <span style="font-size: 18px; color: #94A3B8;">/100</span>
                        </div>
                        <div style="font-size: 10px; letter-spacing: 1px; color: #E2E8F0; margin-top: 4px;">
                            CONVICTION INDEX
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Cluster Buy Alert Badge
        if cluster_flag:
            st.success(
                f"🚨 **EXECUTIVE CLUSTER BUY DETECTED**: Multiple C-suite officers ({insider_data.get('distinct_buyers_count')} distinct executives) "
                f"have accumulated ${total_buy:,.0f} of shares with personal cash within the last {days_lookback} days."
            )

        # 4 Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "💰 Total Insider Purchases",
            f"${total_buy:,.0f}",
            delta=f"{insider_data.get('buy_count', 0)} Transactions",
        )
        c2.metric(
            "📉 Total Insider Dispositions",
            f"${total_sale:,.0f}",
            delta=f"{insider_data.get('sale_count', 0)} Dispositions",
            delta_color="inverse",
        )
        c3.metric(
            "📊 Net Insider Capital Flow",
            f"${net_flow:+,.0f}",
            delta="Net Inflow" if net_flow > 0 else "Net Outflow",
            delta_color="normal" if net_flow > 0 else "inverse",
        )
        c4.metric(
            "👥 Distinct Buyers",
            f"{insider_data.get('distinct_buyers_count', 0)} Officers",
            delta="Cluster Confirmed" if cluster_flag else "Single/None",
        )

        st.markdown("---")

        # Form 4 Transaction Ledger
        st.markdown("#### 📜 SEC Form 4 Transaction Audit Ledger")
        txs = insider_data.get("transactions", [])
        if txs:
            df_tx = pd.DataFrame(txs)
            df_tx_display = df_tx[
                [
                    "filing_date",
                    "officer_name",
                    "officer_title",
                    "transaction_type",
                    "shares",
                    "price",
                    "value_usd",
                ]
            ].copy()
            df_tx_display.columns = [
                "Filing Date",
                "Officer Name",
                "Officer Title",
                "Transaction Type",
                "Shares",
                "Price ($)",
                "Total Value ($)",
            ]

            st.dataframe(
                df_tx_display.style.format(
                    {
                        "Shares": "{:,.0f}",
                        "Price ($)": "${:,.2f}",
                        "Total Value ($)": "${:,.2f}",
                    }
                ),
                use_container_width=True,
                height=300,
            )
        else:
            st.info(
                f"No Form 4 transactions filed for {selected_ticker} in the selected window."
            )

    # =========================================================================
    # TAB 2: UNIVERSE SCANNER
    # =========================================================================
    with tab_scanner:
        st.markdown("#### 📡 Top 15 Insider Accumulation Setups (S&P 500 Universe)")
        st.caption(
            "Screens high-conviction C-Suite buying across the entire model universe."
        )

        if st.button(
            "🚀 Scan Entire Universe for Insider Cluster Buys", type="primary"
        ):
            universe = [
                "NVDA",
                "AAPL",
                "MSFT",
                "IEX",
                "DE",
                "PLTR",
                "QCOM",
                "AMD",
                "UNH",
                "CRWD",
                "FDX",
                "WFC",
                "VRTX",
                "CL",
                "DIS",
                "CAT",
                "JPM",
                "XOM",
                "LLY",
            ]
            with st.spinner("Scanning SEC filings and evaluating conviction scores..."):
                top_insiders = scan_universe_insider_catalysts(universe, top_n=15)

            scan_rows = []
            for item in top_insiders:
                t = item.get("ticker", "")
                scan_rows.append(
                    {
                        "Ticker": t,
                        "Company": COMPANY_NAMES.get(t, t),
                        "Conviction Score": item.get("conviction_score", 50.0),
                        "Cluster Buy": (
                            "🚨 YES" if item.get("cluster_buy_detected") else "NO"
                        ),
                        "Net Flow ($)": item.get("net_purchased_usd", 0.0),
                        "Purchases ($)": item.get("total_buy_usd", 0.0),
                        "Signal": item.get("signal", ""),
                    }
                )

            df_scan = pd.DataFrame(scan_rows)
            st.dataframe(
                df_scan.style.format(
                    {
                        "Conviction Score": "{:.1f}",
                        "Net Flow ($)": "${:+,.0f}",
                        "Purchases ($)": "${:,.0f}",
                    }
                ),
                use_container_width=True,
                height=450,
            )
