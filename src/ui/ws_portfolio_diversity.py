"""
Workspace: Portfolio Diversity & Correlation Health Grader.
Institutional Multi-Asset Parity, Pairwise Heatmaps & Cluster Analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

from src.portfolio_diversity import calculate_portfolio_diversity_grade
from src.paper_broker import PaperBroker
from src.config import COMPANY_NAMES


def render_portfolio_diversity_workspace():
    st.markdown("### 🧬 Portfolio Diversity & Correlation Health Grader")
    st.caption(
        "Institutional Risk Parity Analytics: Computes N x N Pairwise Return Correlations, "
        "Measures Effective Independent Bets (N_eff), and Grades Basket Health (A+ to D)."
    )

    broker = PaperBroker()
    open_positions = broker.state.get("open_positions", {})
    active_tickers = list(open_positions.keys())

    col_mode_a, col_mode_b = st.columns([2, 1])
    with col_mode_a:
        eval_mode = st.radio(
            "Target Universe for Diversity Grading:",
            ["Active Portfolio Holdings (Live Basket)", "Custom Watchlist Tickers"],
            horizontal=True,
        )

    selected_tickers = []
    if eval_mode == "Active Portfolio Holdings (Live Basket)":
        selected_tickers = active_tickers
        if not selected_tickers:
            st.info(
                "No open positions currently in your paper portfolio. Using active watch candidates."
            )
            selected_tickers = [
                "QCOM",
                "AMD",
                "PLTR",
                "UNH",
                "CRWD",
                "VRTX",
                "CL",
                "DIS",
                "FDX",
                "WFC",
            ]
    else:
        with col_mode_b:
            custom_input = st.text_input(
                "Enter Tickers (comma-separated):",
                value="NVDA, AAPL, MSFT, JPM, XOM, LLY, UNH, COST, CAT, NEE",
            )
            selected_tickers = [
                t.strip().upper() for t in custom_input.split(",") if t.strip()
            ]

    timeframe = st.select_slider(
        "Historical Return Lookback Window:",
        options=["1mo", "3mo", "6mo", "1y", "2y"],
        value="6mo",
        help="Historical lookback window used to calculate daily return correlations.",
    )

    with st.spinner("Calculating Pearson Correlation Matrix & Eigenvalue Entropy..."):
        diversity_res = calculate_portfolio_diversity_grade(
            selected_tickers, period=timeframe
        )

    grade = diversity_res.get("grade", "N/A")
    grade_color = diversity_res.get("grade_color", "#10B981")
    grade_desc = diversity_res.get("grade_description", "")
    avg_corr = float(diversity_res.get("average_correlation", 0.0))
    eff_bets = float(diversity_res.get("effective_bets", 0.0))
    nom_holdings = int(diversity_res.get("nominal_holdings", len(selected_tickers)))

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.05), rgba(15, 23, 42, 0.8));
                    border: 1px solid {grade_color}; border-radius: 12px; padding: 20px; margin-bottom: 24px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 13px; letter-spacing: 2px; text-transform: uppercase; color: #94A3B8;">
                        PORTFOLIO DIVERSITY HEALTH GRADE
                    </span>
                    <div style="font-size: 32px; font-weight: 800; color: #FFFFFF; margin-top: 4px;">
                        {grade_desc}
                    </div>
                </div>
                <div style="background: {grade_color}22; border: 2px solid {grade_color}; border-radius: 16px;
                            padding: 10px 28px; text-align: center;">
                    <div style="font-size: 38px; font-weight: 900; color: {grade_color}; line-height: 1;">
                        {grade}
                    </div>
                    <div style="font-size: 10px; letter-spacing: 1px; color: #E2E8F0; margin-top: 4px;">
                        DIVERSITY SCORE
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(
        "Avg Cross-Asset Corr",
        f"{avg_corr:.3f}",
        delta=(
            "Uncorrelated (< 0.20)"
            if avg_corr < 0.20
            else "Elevated Cluster" if avg_corr > 0.50 else "Moderate"
        ),
        delta_color="normal" if avg_corr < 0.20 else "inverse",
    )
    kpi2.metric(
        "Effective Bets (N_eff)",
        f"{eff_bets:.1f} / {nom_holdings}",
        delta=f"{(eff_bets / max(nom_holdings, 1)) * 100.0:.0f}% Independence Breadth",
    )
    max_p = diversity_res.get("max_correlated_pair")
    min_p = diversity_res.get("min_correlated_pair")
    kpi3.metric(
        "Top Correlated Pair",
        f"{max_p[0]}-{max_p[1]}" if max_p else "N/A",
        delta=f"{max_p[2]:.2f} Corr" if max_p else "None",
        delta_color="inverse",
    )
    kpi4.metric(
        "Best Natural Hedge",
        f"{min_p[0]}-{min_p[1]}" if min_p else "N/A",
        delta=f"{min_p[2]:+.2f} Corr" if min_p else "None",
        delta_color="normal",
    )

    st.markdown("---")
    st.markdown("#### 🗺️ Pairwise Return Correlation Matrix (N x N)")
    corr_df = diversity_res.get("corr_matrix_df")

    if corr_df is not None and not corr_df.empty:
        fig_hm = px.imshow(
            corr_df,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdYlGn_r",
            range_color=[-0.4, 1.0],
            labels=dict(x="Asset", y="Asset", color="Pearson Corr"),
        )
        fig_hm.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=520,
            margin=dict(l=20, r=20, t=30, b=20),
            coloraxis_colorbar=dict(title="Corr (r)", thickness=15),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    st.markdown("#### 🛡️ Cluster Diagnostics & Risk Parity Recommendations")
    diagnostics = diversity_res.get("diagnostics", [])
    for diag in diagnostics:
        st.info(diag)

    with st.expander(
        "📘 Understanding Institutional Diversity Grading (Meucci & Risk Parity)"
    ):
        st.markdown(
            """
            * **Why Low Correlation Matters:** In modern portfolio theory, holding assets with correlations below 0.20 mathematically reduces total portfolio volatility without reducing aggregate returns.
            * **Effective Number of Bets (N_eff):** Derived from Shannon entropy over principal component eigenvalues. A portfolio of 15 stocks with high correlation may only offer 4 *effective* bets.
            * **Grade Thresholds:**
              * **Grade A+ (r < 0.15):** Elite risk parity across decoupled sectors.
              * **Grade A- (0.15 <= r < 0.25):** Strong cross-sector breadth.
              * **Grade B+ (0.25 <= r < 0.40):** Healthy institutional balance.
              * **Grade C/D (r >= 0.55):** Heavy cluster risk — assets move together during market sell-offs.
            """
        )
