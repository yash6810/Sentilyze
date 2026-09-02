"""
Workspace: Real-Time Macro Liquidity & Treasury Yield Curve Radar.
Visualizes 10Y-2Y Treasury Spread Inversions, Fed Net Liquidity Index,
and Macroeconomic Financial Conditions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.macro_liquidity import calculate_macro_liquidity_metrics
from src.data_ingestion import get_price_history


def render_macro_liquidity_workspace():
    st.markdown("### 🌐 Real-Time Macro Liquidity & Treasury Yield Radar")
    st.caption(
        "Macroeconomic Liquidity Command: Tracks 10Y-2Y Treasury Yield Curve Inversions, "
        "Federal Reserve Net Liquidity ($L = \\text{Fed Assets} - \\text{TGA} - \\text{RRP}$), "
        "and Systemic Financial Conditions Driving Institutional Equity Flows."
    )

    metrics = calculate_macro_liquidity_metrics()

    # Top KPI Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "📈 10-Yr Treasury Yield",
        f"{metrics['10y_yield']:.2f}%",
        delta=f"2Y: {metrics['2y_yield']:.2f}%",
    )
    m2.metric(
        "⚖️ 10Y - 2Y Spread",
        f"{metrics['spread_10_2_bps']:+.1f} bps",
        delta=metrics["yield_regime"],
    )
    m3.metric(
        "🏦 Fed Net Liquidity",
        f"${metrics['net_liquidity_trillions']:.2f}T",
        delta=f"{metrics['net_liquidity_velocity_pct']:+.2f}% 30-Day Velocity",
    )
    m4.metric(
        "🛡️ Systemic Stress Score",
        f"{metrics['financial_stress_score']:.1f}/100",
        delta="Loose Conditions (Bullish)",
        delta_color="normal",
    )

    st.markdown("---")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### 📊 Federal Reserve Balance Sheet Breakdown")
        fig_liq = go.Figure(
            data=[
                go.Bar(
                    name="Fed Total Assets",
                    x=["Fed Total Assets"],
                    y=[metrics["fed_assets_trillions"]],
                    marker_color="#38BDF8",
                ),
                go.Bar(
                    name="Treasury General Account (TGA)",
                    x=["TGA Drain"],
                    y=[metrics["tga_balance_trillions"]],
                    marker_color="#EF4444",
                ),
                go.Bar(
                    name="Overnight Reverse Repo (RRP)",
                    x=["RRP Facility"],
                    y=[metrics["reverse_repo_trillions"]],
                    marker_color="#F59E0B",
                ),
                go.Bar(
                    name="Net Market Liquidity",
                    x=["Net Liquidity Available"],
                    y=[metrics["net_liquidity_trillions"]],
                    marker_color="#10B981",
                ),
            ]
        )
        fig_liq.update_layout(
            title="Federal Reserve Net Liquidity Components ($ Trillions)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
            margin=dict(l=20, r=20, t=35, b=20),
            yaxis_title="Trillions USD ($T)",
        )
        st.plotly_chart(fig_liq, use_container_width=True)

    with col_chart2:
        st.markdown("#### 📉 10-Year Benchmark Treasury Yield History")
        df_tnx = get_price_history("^TNX", period="1y", use_cache=True)
        if not df_tnx.empty and "Close" in df_tnx.columns:
            yield_series = df_tnx["Close"] / 10.0
            fig_tnx = go.Figure()
            fig_tnx.add_trace(
                go.Scatter(
                    x=yield_series.index,
                    y=yield_series.values,
                    name="10Y Yield (%)",
                    line=dict(color="#F59E0B", width=2.0),
                )
            )
            fig_tnx.update_layout(
                title="10-Year US Treasury Yield (% Annualized)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=340,
                margin=dict(l=20, r=20, t=35, b=20),
                yaxis_title="Yield Percentage (%)",
            )
            st.plotly_chart(fig_tnx, use_container_width=True)
