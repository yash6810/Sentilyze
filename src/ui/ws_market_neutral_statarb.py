"""
Workspace: Market-Neutral Cointegration & Statistical Arbitrage Radar.
Engle-Granger Two-Step Tests, Ornstein-Uhlenbeck Half-Life & Live Z-Score Trading Signals.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

from src.statistical_arbitrage import (
    calculate_hedge_ratio_and_spread,
    evaluate_cointegration_adf,
    calculate_half_life,
    calculate_rolling_zscore,
    generate_pairs_trading_signals,
    scan_pairs_universe,
    backtest_pairs_strategy,
)
from src.data_ingestion import get_price_history
from src.config import COMPANY_NAMES


def render_market_neutral_statarb_workspace(selected_ticker: str = "NVDA"):
    st.markdown("### 🔄 Market-Neutral Cointegration & Statistical Arbitrage Radar")
    st.caption(
        "Institutional Market-Neutral Alpha Engine: Uses Engle-Granger Cointegration Tests, "
        "Ornstein-Uhlenbeck Mean-Reversion Half-Life, and Rolling Z-Score Corridors (±2.0σ) "
        "to Extract Long/Short Spread Profits with Zero Directional Market Exposure."
    )

    tab_pairs_radar, tab_chart, tab_backtest = st.tabs(
        [
            "📡 S&P 500 Cointegrated Pairs Radar",
            f"🎯 Live Spread & Z-Score Corridor ({selected_ticker} Pairs)",
            "📈 Market-Neutral Pairs Backtest Simulator",
        ]
    )

    # Preset Institutional Twin Pairs
    INSTITUTIONAL_PAIRS = [
        ("NVDA", "AMD"),
        ("GOOGL", "META"),
        ("V", "MA"),
        ("CAT", "DE"),
        ("XOM", "CVX"),
        ("HD", "LOW"),
        ("UNH", "CVS"),
        ("MSFT", "AAPL"),
        ("GS", "MS"),
        ("FDX", "UPS"),
    ]

    # =========================================================================
    # TAB 1: COINTEGRATED PAIRS RADAR
    # =========================================================================
    with tab_pairs_radar:
        st.markdown("#### 📡 Top Cointegrated Twin Asset Radar")
        st.caption(
            "Scans institutional sector twins for stationary mean-reverting equilibrium spreads (ADF P-Value < 0.05)."
        )

        lookback = st.select_slider(
            "Historical Lookback Window:", options=["6mo", "1y", "2y"], value="1y"
        )

        if st.button(
            "🚀 Scan All Institutional Pairs for Cointegration", type="primary"
        ):
            with st.spinner(
                "Calculating OLS Hedge Ratios, ADF Stationarity Tests, and Ornstein-Uhlenbeck Half-Lives..."
            ):
                scan_results = []
                for tA, tB in INSTITUTIONAL_PAIRS:
                    try:
                        dfA = get_price_history(tA, period=lookback, use_cache=True)
                        dfB = get_price_history(tB, period=lookback, use_cache=True)
                        if (
                            not dfA.empty
                            and not dfB.empty
                            and "Close" in dfA.columns
                            and "Close" in dfB.columns
                        ):
                            beta, alpha, spread = calculate_hedge_ratio_and_spread(
                                dfA["Close"], dfB["Close"]
                            )
                            adf = evaluate_cointegration_adf(spread)
                            half_life = calculate_half_life(spread)
                            zscore = calculate_rolling_zscore(spread, window=20)
                            latest_z = (
                                float(zscore.iloc[-1]) if not zscore.empty else 0.0
                            )

                            # Signal Determination
                            if latest_z <= -2.0:
                                sig = f"🟢 LONG {tA} / SHORT {tB}"
                            elif latest_z >= 2.0:
                                sig = f"🔴 SHORT {tA} / LONG {tB}"
                            elif abs(latest_z) <= 0.5:
                                sig = "⚪ AT EQUILIBRIUM"
                            else:
                                sig = "🟡 DRIFTING"

                            scan_results.append(
                                {
                                    "Pair": f"{tA} - {tB}",
                                    "ADF P-Value": adf.get("p_value", 1.0),
                                    "Cointegrated": (
                                        "✅ CONFIRMED"
                                        if adf.get("is_cointegrated")
                                        else "❌ NO"
                                    ),
                                    "Hedge Ratio (β)": beta,
                                    "Half-Life (Days)": half_life,
                                    "Current Z-Score": latest_z,
                                    "Trading Signal": sig,
                                }
                            )
                    except Exception as e:
                        pass

                if scan_results:
                    df_scan = pd.DataFrame(scan_results)
                    st.dataframe(
                        df_scan.style.format(
                            {
                                "ADF P-Value": "{:.4f}",
                                "Hedge Ratio (β)": "{:.3f}",
                                "Half-Life (Days)": "{:.1f} days",
                                "Current Z-Score": "{:+.2f}σ",
                            }
                        ),
                        use_container_width=True,
                    )

    # =========================================================================
    # TAB 2: LIVE SPREAD & Z-SCORE CORRIDOR
    # =========================================================================
    with tab_chart:
        st.markdown(f"#### 🎯 Live Cointegration Corridor & Z-Score Analysis")

        pair_col1, pair_col2 = st.columns(2)
        with pair_col1:
            asset_A = st.selectbox(
                "Asset A (Dependent Leg):",
                ["NVDA", "GOOGL", "V", "CAT", "XOM", "HD", "UNH", "MSFT", "GS", "FDX"],
                index=0,
            )
        with pair_col2:
            default_b = {
                "NVDA": "AMD",
                "GOOGL": "META",
                "V": "MA",
                "CAT": "DE",
                "XOM": "CVX",
                "HD": "LOW",
                "UNH": "CVS",
                "MSFT": "AAPL",
                "GS": "MS",
                "FDX": "UPS",
            }.get(asset_A, "AMD")
            asset_B = st.selectbox(
                "Asset B (Hedge Leg):",
                ["AMD", "META", "MA", "DE", "CVX", "LOW", "CVS", "AAPL", "MS", "UPS"],
                index=0,
            )

        with st.spinner(
            f"Computing Cointegration & Z-Scores for {asset_A} / {asset_B}..."
        ):
            dfA = get_price_history(asset_A, period="1y", use_cache=True)
            dfB = get_price_history(asset_B, period="1y", use_cache=True)

            if (
                not dfA.empty
                and not dfB.empty
                and "Close" in dfA.columns
                and "Close" in dfB.columns
            ):
                beta, alpha, spread = calculate_hedge_ratio_and_spread(
                    dfA["Close"], dfB["Close"]
                )
                adf = evaluate_cointegration_adf(spread)
                hl = calculate_half_life(spread)
                zscore = calculate_rolling_zscore(spread, window=20)
                curr_z = float(zscore.iloc[-1]) if not zscore.empty else 0.0

                # Top Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(
                    "⚖️ Hedge Ratio (Beta)",
                    f"{beta:.3f}",
                    delta=f"1 {asset_A} : {beta:.2f} {asset_B}",
                )
                m2.metric(
                    "🎯 ADF P-Value",
                    f"{adf.get('p_value', 1.0):.4f}",
                    delta=(
                        "Stationary (< 0.05)"
                        if adf.get("is_cointegrated")
                        else "Non-Cointegrated"
                    ),
                    delta_color="normal" if adf.get("is_cointegrated") else "inverse",
                )
                m3.metric(
                    "⏳ Mean-Reversion Half-Life",
                    f"{hl:.1f} Days",
                    delta="Ornstein-Uhlenbeck",
                )
                m4.metric(
                    "📊 Current Z-Score",
                    f"{curr_z:+.2f}σ",
                    delta=(
                        "Overbought"
                        if curr_z >= 2.0
                        else "Oversold" if curr_z <= -2.0 else "Normal"
                    ),
                    delta_color="inverse" if abs(curr_z) >= 2.0 else "normal",
                )

                st.markdown("---")

                # Z-Score Corridor Plotly Chart
                fig_z = go.Figure()
                fig_z.add_trace(
                    go.Scatter(
                        x=zscore.index,
                        y=zscore.values,
                        name="Spread Z-Score (σ)",
                        line=dict(color="#38BDF8", width=2.0),
                    )
                )

                # Upper & Lower Entry Thresholds (+2.0 and -2.0)
                fig_z.add_hline(
                    y=2.0,
                    line_dash="dash",
                    line_color="#EF4444",
                    annotation_text="Upper Entry (+2.0σ: Short A / Long B)",
                )
                fig_z.add_hline(
                    y=-2.0,
                    line_dash="dash",
                    line_color="#10B981",
                    annotation_text="Lower Entry (-2.0σ: Long A / Short B)",
                )
                fig_z.add_hline(
                    y=0.0,
                    line_dash="dot",
                    line_color="#64748B",
                    annotation_text="Equilibrium Mean (0.0σ)",
                )

                fig_z.update_layout(
                    title=f"Statistical Arbitrage Rolling Z-Score Corridor: {asset_A} vs {asset_B}",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=380,
                    margin=dict(l=20, r=20, t=40, b=20),
                    yaxis_title="Z-Score (Standard Deviations)",
                )
                st.plotly_chart(fig_z, use_container_width=True)

    # =========================================================================
    # TAB 3: BACKTEST SIMULATOR
    # =========================================================================
    with tab_backtest:
        st.markdown(
            f"#### 📈 Market-Neutral Pairs Trading Backtest: **{asset_A} vs {asset_B}**"
        )
        st.caption(
            "Simulates simultaneous Long/Short execution when spread diverges beyond ±2.0 standard deviations."
        )

        if st.button("🚀 Run Market-Neutral Pairs Backtest", type="primary"):
            with st.spinner("Simulating market-neutral Long/Short execution..."):
                bt_res = backtest_pairs_strategy(
                    dfA["Close"], dfB["Close"], entry_z=2.0, exit_z=0.5
                )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "💰 Total Pairs Return", f"{bt_res.get('total_return_pct', 0.0):+.2f}%"
            )
            c2.metric("🛡️ Sharpe Ratio", f"{bt_res.get('sharpe_ratio', 0.0):.2f}")
            c3.metric(
                "🌪️ Max Drawdown",
                f"{bt_res.get('max_drawdown_pct', 0.0):.2f}%",
                delta_color="inverse",
            )
            c4.metric(
                "🎲 Total Trades",
                f"{bt_res.get('total_trades', 0)} Pairs",
                delta=f"{bt_res.get('win_rate_pct', 0.0):.1f}% Win Rate",
            )

            eq_curve = bt_res.get("equity_curve")
            if eq_curve is not None and not eq_curve.empty:
                fig_bt = go.Figure()
                fig_bt.add_trace(
                    go.Scatter(
                        x=eq_curve.index,
                        y=eq_curve.values,
                        name="Market-Neutral Equity ($)",
                        line=dict(color="#10B981", width=2.5),
                    )
                )
                fig_bt.update_layout(
                    title="Market-Neutral Cumulative Equity Growth Curve ($100k Base)",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=320,
                    margin=dict(l=20, r=20, t=35, b=20),
                    yaxis_title="Account Equity ($)",
                )
                st.plotly_chart(fig_bt, use_container_width=True)
