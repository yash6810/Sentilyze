"""
Real-Time Market Anomaly Screener Component for Streamlit.

Functions:
- Renders high-speed multi-condition screener table for 500+ assets.
- Allows real-time filtering by RVOL, Momentum velocity, Range position, and Sector.
"""

from typing import List, Optional
import streamlit as st
import pandas as pd
from src.screener_engine import run_universe_screener
from src.autonomous_trader import load_universe_tickers


def render_live_screener_section():
    """
    Renders the Real-Time Market Anomaly Screener UI.
    """
    st.markdown("### 🌐 Real-Time Market Anomaly Screener")
    st.caption(
        "Live multi-condition scanning for Relative Volume (RVOL) surges, Day Range Breakouts, and Pullback Bounces."
    )

    tickers = load_universe_tickers()

    f_col1, f_col2, f_col3 = st.columns([1, 1, 1])
    with f_col1:
        min_rvol = st.slider("Min Relative Volume (RVOL)", 0.5, 5.0, 1.0, 0.1)
    with f_col2:
        min_score = st.slider("Min Anomaly Score", 40, 95, 50, 5)
    with f_col3:
        scan_count = st.selectbox(
            "Universe Scan Scope", [25, 50, 100, len(tickers)], index=0
        )

    if st.button("⚡ Run Real-Time Screener Scan", use_container_width=True):
        with st.spinner(
            f"Scanning top {scan_count} assets for microstructure anomalies..."
        ):
            screener_df = run_universe_screener(tickers[:scan_count])
            st.session_state["screener_results"] = screener_df

    df = st.session_state.get("screener_results")
    if df is not None and not df.empty:
        filtered = df[
            (df["rvol"] >= min_rvol) & (df["anomaly_score"] >= min_score)
        ].copy()

        st.markdown(f"**Found {len(filtered)} active setups matching criteria:**")

        st.dataframe(
            filtered,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", width="small"),
                "price": st.column_config.NumberColumn("Spot Price", format="$%.2f"),
                "change_pct": st.column_config.NumberColumn(
                    "Change %", format="%+.2f%%"
                ),
                "rvol": st.column_config.NumberColumn("RVOL", format="%.2fx"),
                "range_pos_pct": st.column_config.ProgressColumn(
                    "Day Range Pos", min_value=0, max_value=100, format="%.0f%%"
                ),
                "mom_5d_pct": st.column_config.NumberColumn(
                    "5d Return", format="%+.2f%%"
                ),
                "setup_type": st.column_config.TextColumn(
                    "Setup Anomaly", width="medium"
                ),
                "anomaly_score": st.column_config.NumberColumn("Score", format="%.1f"),
                "sector": st.column_config.TextColumn("Sector", width="medium"),
            },
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info(
            "Click 'Run Real-Time Screener Scan' above to scan active S&P universe setups."
        )
