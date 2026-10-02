import os
import json
import streamlit as st
import pandas as pd
from src.ui.components import render_workspace_header
from components.live_screener import render_live_screener_section
from src.universe_mega_scanner import run_universe_mega_scan


def render_screener_workspace():
    """Renders the Real-Time Market Anomaly Screener Workspace."""
    render_workspace_header(
        title="🌐 Real-Time Market Anomaly Screener & Mega-Scanner",
        subtitle="Sub-second multi-condition scanning for RVOL surges, Range Breakouts, and 1,000+ Stock Institutional Alpha Funnels.",
        badge_text="LIVE MARKET RADAR",
        badge_color="#00D4AA",
    )

    t1, t2 = st.tabs(["📡 Real-Time Anomaly Radar", "🌌 1,000+ Universe Mega-Scanner"])

    with t1:
        render_live_screener_section()

    with t2:
        st.markdown("### 🌌 1,000+ Equities Multi-Factor Alpha Funnel")
        st.caption(
            "Stage 1 screen: XGBoost momentum + 21D Return + SMA200 + Volume Surges | Stage 2: Committee quorum arbitration."
        )

        scan_file = "results/sp500_universe_scan_full.json"

        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            scan_size = st.slider(
                "Universe Scan Breadth",
                min_value=50,
                max_value=1500,
                value=250,
                step=50,
            )
        with c2:
            top_n = st.selectbox("Top Leaders for Committee", [10, 25, 50], index=1)
        with c3:
            st.write("")
            run_btn = st.button(
                "🚀 Run Live Universe Scan", type="primary", use_container_width=True
            )

        if run_btn:
            with st.spinner(
                f"Scanning {scan_size} equities across multi-factor alpha funnel..."
            ):
                try:
                    res = run_universe_mega_scan(
                        max_universe_size=scan_size, top_n_for_committee=top_n
                    )
                    st.success(
                        f"Successfully screened {res.get('valid_screened_stocks', 0)} stocks in {res.get('execution_time_seconds', 0):.2f}s!"
                    )
                except Exception as e:
                    st.error(f"Scan error: {e}")

        if os.path.exists(scan_file):
            try:
                with open(scan_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                rankings = data.get("universe_rankings", [])
                if rankings:
                    df = pd.DataFrame(rankings)
                    st.markdown(
                        f"**Last Scan Timestamp:** `{data.get('timestamp', 'N/A')}` | **Total Screened:** `{data.get('valid_screened_stocks', len(df))}`"
                    )
                    st.dataframe(
                        df,
                        use_container_width=True,
                        column_config={
                            "ticker": st.column_config.TextColumn(
                                "Ticker", width="small"
                            ),
                            "current_price": st.column_config.NumberColumn(
                                "Price ($)", format="$%.2f"
                            ),
                            "composite_alpha_score": st.column_config.ProgressColumn(
                                "Composite Alpha",
                                min_value=0,
                                max_value=100,
                                format="%.1f",
                            ),
                            "model_confidence_pct": st.column_config.NumberColumn(
                                "Model Conf", format="%.1f%%"
                            ),
                            "model_signal": st.column_config.TextColumn("Signal"),
                            "rsi_14": st.column_config.NumberColumn(
                                "RSI (14)", format="%.1f"
                            ),
                            "return_21d_pct": st.column_config.NumberColumn(
                                "21D Return", format="%.2f%%"
                            ),
                            "price_vs_sma200_pct": st.column_config.NumberColumn(
                                "vs SMA200", format="%.2f%%"
                            ),
                            "volume_ratio": st.column_config.NumberColumn(
                                "Vol Ratio", format="%.2fx"
                            ),
                        },
                        hide_index=True,
                    )
            except Exception as e:
                st.warning(f"Could not load scan cache: {e}")
        else:
            st.info(
                "No cached universe scan found. Click 'Run Live Universe Scan' to execute."
            )
