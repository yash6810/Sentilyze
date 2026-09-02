"""
Workspace: Institutional Risk & Alpha Performance Factsheet.
Quantitative Factsheet Analytics: Sortino, Calmar, Monthly Return Grids & Drawdown Plots.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

from src.performance_factsheet import generate_comprehensive_factsheet
from src.paper_broker import PaperBroker


def render_performance_factsheet_workspace():
    st.markdown("### 📊 Institutional Risk & Alpha Performance Factsheet")
    st.caption(
        "Institutional Factsheet Suite: Computes 30+ Advanced Risk Ratios (Sortino, Calmar, Omega, VaR/CVaR), "
        "Monthly Return Heatmap Grids (Jan-Dec), and Underwater Drawdown Duration Telemetry."
    )

    broker = PaperBroker()
    portfolio_summary = broker.get_portfolio_summary()

    with st.spinner("Computing Quantitative Risk & Return Attribution Metrics..."):
        factsheet = generate_comprehensive_factsheet()

    tot_ret = factsheet["total_return_pct"]
    bench_ret = factsheet["benchmark_return_pct"]
    sharpe = factsheet["sharpe_ratio"]
    sortino = factsheet["sortino_ratio"]
    calmar = factsheet["calmar_ratio"]
    max_dd = factsheet["max_drawdown_pct"]
    win_rate = factsheet["win_rate_pct"]
    prof_factor = factsheet["profit_factor"]

    # =========================================================================
    # 1. TOP-LEVEL KPI SCORECARD
    # =========================================================================
    st.markdown("#### 🏆 Top-Level Quantitative Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "📈 Cumulative Total Return",
        f"{tot_ret:+.2f}%",
        delta=f"{tot_ret - bench_ret:+.2f}% vs S&P 500",
    )
    c2.metric(
        "🛡️ Sortino Ratio (Downside Risk)",
        f"{sortino:.2f}",
        delta="Elite (> 2.0)" if sortino > 2.0 else "Good",
    )
    c3.metric(
        "🎯 Calmar Ratio (CAGR / MaxDD)",
        f"{calmar:.2f}",
        delta=f"{factsheet['cagr_pct']:.1f}% CAGR",
    )
    c4.metric(
        "🌪️ Maximum Drawdown",
        f"{max_dd:.2f}%",
        delta=f"{abs(max_dd):.1f}% Depth",
        delta_color="inverse",
    )

    c5, c6, c7, c8 = st.columns(4)
    c5.metric(
        "⚖️ Sharpe Ratio (Rf=4%)",
        f"{sharpe:.2f}",
        delta=f"Vol: {factsheet['annual_volatility_pct']:.1f}%",
    )
    c6.metric(
        "🎲 Win Rate (Daily)",
        f"{win_rate:.1f}%",
        delta=f"Profit Factor: {prof_factor:.2f}",
    )
    c7.metric(
        "📉 Value at Risk (VaR 95%)",
        f"{factsheet['var_95_daily_pct']:.2f}%",
        delta="1-Day 95% Cutoff",
        delta_color="inverse",
    )
    c8.metric(
        "⚡ Expected Shortfall (CVaR)",
        f"{factsheet['cvar_95_daily_pct']:.2f}%",
        delta="Tail Risk Floor",
        delta_color="inverse",
    )

    st.markdown("---")

    # =========================================================================
    # 2. CUMULATIVE EQUITY & UNDERWATER DRAWDOWN PLOTS
    # =========================================================================
    st.markdown("#### 📈 Cumulative Growth & Underwater Drawdown Profiles")
    curves_df = factsheet.get("curves_df")

    if curves_df is not None and not curves_df.empty:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_eq = go.Figure()
            fig_eq.add_trace(
                go.Scatter(
                    x=curves_df.index,
                    y=curves_df["Strategy Cumulative"],
                    name="Sentilyze Multi-Asset Parity",
                    line=dict(color="#10B981", width=2.5),
                )
            )
            fig_eq.add_trace(
                go.Scatter(
                    x=curves_df.index,
                    y=curves_df["Benchmark Cumulative"],
                    name="S&P 500 Benchmark",
                    line=dict(color="#64748B", width=1.5, dash="dash"),
                )
            )
            fig_eq.update_layout(
                title="Cumulative Total Growth Curve ($1.00 Base)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )
            st.plotly_chart(fig_eq, use_container_width=True)

        with col_c2:
            fig_dd = go.Figure()
            fig_dd.add_trace(
                go.Scatter(
                    x=curves_df.index,
                    y=curves_df["Underwater Drawdown"],
                    name="Drawdown Depth (%)",
                    fill="tozeroy",
                    line=dict(color="#EF4444", width=1.5),
                    fillcolor="rgba(239, 68, 68, 0.2)",
                )
            )
            fig_dd.update_layout(
                title="Underwater Drawdown Profile (Peak-to-Trough)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis=dict(ticksuffix="%"),
            )
            st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("---")

    # =========================================================================
    # 3. MONTHLY RETURNS CALENDAR GRID
    # =========================================================================
    st.markdown("#### 📅 Monthly Returns Calendar Grid (Jan - Dec vs YTD)")
    df_monthly = factsheet.get("monthly_grid_df")

    if df_monthly is not None and not df_monthly.empty:
        # Style the dataframe with green/red gradients
        st.dataframe(
            df_monthly.style.format("{:+.2f}%")
            .background_gradient(
                cmap="RdYlGn",
                vmin=-5.0,
                vmax=5.0,
                subset=[c for c in df_monthly.columns if c != "YTD"],
            )
            .background_gradient(cmap="RdYlGn", vmin=-10.0, vmax=20.0, subset=["YTD"]),
            use_container_width=True,
        )

    # =========================================================================
    # 4. INSTITUTIONAL RATIO CHEAT SHEET
    # =========================================================================
    with st.expander("📘 Institutional Ratio Definitions & Formulas"):
        st.markdown(
            """
            * **Sortino Ratio:** $\\frac{R_p - R_f}{\\sigma_{\\text{downside}}}$. Penalizes only downside volatility, ignoring profitable upside swings.
            * **Calmar Ratio:** $\\frac{\\text{CAGR}}{|\\text{Max Drawdown}|}$. Measures return generated per unit of maximum account drawdown.
            * **Omega Ratio:** Ratio of probability-weighted gains to probability-weighted losses relative to benchmark return threshold.
            * **Tail Ratio:** $\\frac{95\\text{th percentile daily gain}}{|5\\text{th percentile daily loss}|}$. Ratios $> 1.0$ indicate positively skewed return profiles.
            * **Expected Shortfall (CVaR 95%):** Average expected loss on days that fall into the worst 5% tail.
            """
        )
