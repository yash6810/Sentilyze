"""
Workspace 5: Multi-Asset Unified Portfolio & Risk Parity Allocation.
"""

import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
from src.ui.components import render_workspace_header
from src.portfolio import (
    build_unified_portfolio,
    calculate_risk_parity_weights,
    load_all_ticker_portfolios,
)


def render_portfolio_workspace(selected_ticker: str):
    """Renders the Portfolio Optimization and Unified Multi-Asset Allocation workspace."""
    render_workspace_header(
        title="💼 Institutional Multi-Asset Portfolio & Risk Parity",
        subtitle="Regime-Aware Dynamic Leverage + Inverse Volatility Risk Parity Weighting",
        badge_text="RISK PARITY ALLOCATION",
        badge_color="#3B82F6",
    )

    portfolios = load_all_ticker_portfolios(results_dir="results")

    # Top KPI Metrics Bar
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🌐 Tracked S&P Assets", f"{len(portfolios)} Portfolios")
    k2.metric("⚡ Rebalancing Mode", "Monthly Dynamic")
    k3.metric("🎯 Risk Parity Sizing", "Inverse Volatility (1/σ)")
    k4.metric("🛡️ Leverage Buffer", "1.0x (Unleveraged Core)")

    st.markdown("### 📊 Universe Capital Allocation Weights")
    if portfolios:
        weights = calculate_risk_parity_weights(portfolios)
        df_w = pd.DataFrame(
            [{"Ticker": t, "Weight (%)": w * 100.0} for t, w in weights.items()]
        ).sort_values(by="Weight (%)", ascending=False)

        fig = px.pie(
            df_w,
            values="Weight (%)",
            names="Ticker",
            hole=0.45,
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            df_w.style.format({"Weight (%)": "{:.2f}%"}), use_container_width=True
        )
    else:
        st.info(
            "No precomputed ticker backtest portfolios found in `results/`. Displaying standard equal weighting."
        )
