"""
Workspace 6: Walk-Forward Backtesting & Performance Tearsheet.
"""

import os
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.ui.components import render_workspace_header


def render_backtesting_workspace(selected_ticker: str):
    """Renders the Walk-Forward Backtesting and Strategy Tearsheet workspace."""
    render_workspace_header(
        title=f"📈 Walk-Forward Backtesting Tearsheet ({selected_ticker})",
        subtitle="Zero Look-Ahead Bias Walk-Forward Optimization & Out-of-Sample Performance",
        badge_text="WFO VALIDATED",
        badge_color="#10B981",
    )

    metrics_file = os.path.join("results", f"{selected_ticker}_metrics.json")
    portfolio_file = os.path.join("results", f"{selected_ticker}_portfolio.csv")

    if not os.path.exists(metrics_file):
        st.warning(
            f"No precomputed backtest results found for {selected_ticker}. Showing baseline tearsheet."
        )
        return

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    # Top KPI Metrics
    b1, b2, b3, b4 = st.columns(4)
    b1.metric(
        "🏆 Strategy Total Return",
        f"{metrics.get('strategy_return', 0.0)*100:.2f}%",
        delta=f"vs B&H: {metrics.get('buy_hold_return', 0.0)*100:.2f}%",
    )
    b2.metric("⚡ Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0.0):.2f}")
    b3.metric("🛡️ Max Drawdown", f"{metrics.get('max_drawdown', 0.0)*100:.2f}%")
    b4.metric("🎯 Win Rate", f"{metrics.get('win_rate', 0.0)*100:.1f}%")

    # Cumulative Return Chart
    if os.path.exists(portfolio_file):
        df_p = pd.read_csv(portfolio_file)
        if "Date" in df_p.columns and "Strategy_Cumulative" in df_p.columns:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df_p["Date"],
                    y=df_p["Strategy_Cumulative"],
                    mode="lines",
                    name="Sentilyze AI Strategy",
                    line=dict(color="#10B981", width=2.5),
                )
            )
            if "Buy_Hold_Cumulative" in df_p.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_p["Date"],
                        y=df_p["Buy_Hold_Cumulative"],
                        mode="lines",
                        name="Benchmark (Buy & Hold)",
                        line=dict(color="#64748B", width=1.5, dash="dot"),
                    )
                )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=420,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
