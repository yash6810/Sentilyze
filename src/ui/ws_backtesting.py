"""
Workspace 6: Walk-Forward Backtesting & Performance Tearsheet.
"""

import os
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.ui.components import render_workspace_header
import train


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

    # Retrain button row
    col_hdr, col_btn = st.columns([3, 1])
    with col_hdr:
        st.markdown(
            f"**Out-of-Sample Performance Audit for `{selected_ticker}`** (Walk-Forward Rolling Window)"
        )
    with col_btn:
        if st.button(
            f"⚡ Run WFO Train for {selected_ticker}", use_container_width=True
        ):
            with st.spinner(
                f"Training Walk-Forward Model & generating tearsheet for {selected_ticker}..."
            ):
                try:
                    train.main(selected_ticker, use_cache=True)
                    st.success(
                        f"✅ Model training and backtesting complete for {selected_ticker}!"
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")

    if not os.path.exists(metrics_file):
        st.info(
            f"ℹ️ No precomputed backtest results found for `{selected_ticker}` yet. "
            f"Click the button above to run Walk-Forward Optimization and generate the tearsheet live!"
        )
        return

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    strat_ret = float(
        metrics.get("strategy_total_return", metrics.get("strategy_return", 0.0))
    )
    bench_ret = float(
        metrics.get("buy_and_hold_total_return", metrics.get("buy_hold_return", 0.0))
    )
    sharpe = float(metrics.get("sharpe_ratio", 0.0))
    max_dd = float(
        metrics.get("strategy_max_drawdown", metrics.get("max_drawdown", 0.0))
    )
    win_rate = float(metrics.get("win_rate", 0.5))

    # Top KPI Metrics
    b1, b2, b3, b4 = st.columns(4)
    b1.metric(
        "🏆 Strategy Total Return",
        f"{strat_ret*100:+.2f}%",
        delta=f"vs Benchmark: {bench_ret*100:+.2f}%",
    )
    b2.metric("⚡ Sharpe Ratio", f"{sharpe:.2f}")
    b3.metric("🛡️ Max Drawdown", f"{max_dd*100:.2f}%")
    b4.metric("🎯 Win Rate", f"{win_rate*100:.1f}%")

    # Cumulative Return Chart
    if os.path.exists(portfolio_file):
        df_p = pd.read_csv(portfolio_file)
        date_col = "Date" if "Date" in df_p.columns else df_p.columns[0]
        x_dates = df_p[date_col]

        fig = go.Figure()

        # Strategy line
        if "total" in df_p.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_dates,
                    y=df_p["total"],
                    mode="lines",
                    name="Sentilyze AI Strategy ($)",
                    line=dict(color="#10B981", width=2.5),
                )
            )
        elif "Strategy_Cumulative" in df_p.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_dates,
                    y=df_p["Strategy_Cumulative"],
                    mode="lines",
                    name="Sentilyze AI Strategy",
                    line=dict(color="#10B981", width=2.5),
                )
            )

        # Benchmark line
        if "benchmark" in df_p.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_dates,
                    y=df_p["benchmark"],
                    mode="lines",
                    name="Benchmark (Buy & Hold)",
                    line=dict(color="#64748B", width=1.5, dash="dot"),
                )
            )
        elif "Buy_Hold_Cumulative" in df_p.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_dates,
                    y=df_p["Buy_Hold_Cumulative"],
                    mode="lines",
                    name="Benchmark (Buy & Hold)",
                    line=dict(color="#64748B", width=1.5, dash="dot"),
                )
            )

        fig.update_layout(
            title=f"Cumulative Portfolio Equity Growth vs Buy & Hold Benchmark ({selected_ticker})",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=460,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap plot if available
        heatmap_path = os.path.join(
            "results", f"{selected_ticker}_monthly_returns_heatmap.png"
        )
        if os.path.exists(heatmap_path):
            st.markdown("### 🗓️ Monthly Returns Distribution Heatmap")
            st.image(heatmap_path, use_container_width=True)
