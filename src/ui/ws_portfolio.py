"""
Workspace 5: Regime-Aware Kelly Allocation & Portfolio Optimization.
"""

import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
from src.ui.components import render_workspace_header
from src.portfolio import optimize_portfolio_kelly, get_macro_regime


def render_portfolio_workspace(selected_ticker: str):
    """Renders the Portfolio Optimization and Regime-Aware Allocation workspace."""
    render_workspace_header(
        title="💼 Institutional Portfolio Optimization & Sizing",
        subtitle="Regime-Aware Dynamic Leverage + Fractional Kelly Criterion Risk Sizing",
        badge_text="KELLY ALLOCATION",
        badge_color="#3B82F6",
    )

    regime_info = get_macro_regime(selected_ticker)

    # Top KPI Metrics Bar
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🌪️ Macro Regime", regime_info.get("regime", "BULLISH"))
    k2.metric(
        "⚡ Dynamic Leverage Cap", f"{regime_info.get('leverage_multiplier', 1.0):.2f}x"
    )
    k3.metric("🎯 Kelly Sizing Fraction", "Quarter Kelly (0.25)")
    k4.metric("🛡️ Max Single Position Cap", "15.0%")

    st.markdown("### 📊 Universe Capital Allocation Weights")
    opt_weights = optimize_portfolio_kelly()
    if opt_weights:
        df_w = pd.DataFrame(
            [{"Ticker": t, "Weight (%)": w * 100.0} for t, w in opt_weights.items()]
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
        st.info("No active universe portfolio weights calculated.")
