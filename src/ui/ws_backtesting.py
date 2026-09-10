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
                    import train

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

    with open(metrics_file, "r", encoding="utf-8") as f:
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

    # Model Classification & Directional Accuracy KPIs
    acc = metrics.get("accuracy")
    roc_auc = metrics.get("roc_auc")
    prec = metrics.get("precision")
    rec = metrics.get("recall")
    f1 = metrics.get("f1")
    base_acc = metrics.get("baseline_logistic_accuracy")

    if acc is not None:
        st.markdown(
            f"<div style='margin-top: 14px; margin-bottom: 6px; font-weight: 700; color: #94A3B8; font-size: 0.82rem; letter-spacing: 0.05em; text-transform: uppercase;'>🧠 Machine Learning Validation Metrics ({selected_ticker}) — 5-Fold Purged CPCV</div>",
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(
            "🎯 Model Accuracy",
            f"{float(acc)*100:.2f}%",
            delta=(
                f"vs Baseline: {float(base_acc)*100:.2f}%"
                if base_acc is not None
                else None
            ),
        )
        c2.metric(
            "📊 ROC-AUC",
            f"{float(roc_auc):.4f}" if roc_auc is not None else "N/A",
        )
        c3.metric(
            "🎯 Precision",
            f"{float(prec)*100:.2f}%" if prec is not None else "N/A",
        )
        c4.metric(
            "🔄 Recall",
            f"{float(rec)*100:.2f}%" if rec is not None else "N/A",
        )
        c5.metric("⚖️ F1 Score", f"{float(f1):.4f}" if f1 is not None else "N/A")

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

    # Universe-Wide Accuracy & Model Performance Leaderboard
    with st.expander(
        "📊 Universe-Wide Model Accuracy & Performance Leaderboard (All 535 Tickers)",
        expanded=False,
    ):
        import glob

        @st.cache_data(ttl=600)
        def _get_universe_leaderboard():
            summary_path = os.path.join("results", "universe_summary.json")
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r", encoding="utf-8") as sf:
                        data = json.load(sf)
                    df_lead = pd.DataFrame(data)
                    df_lead.rename(
                        columns={
                            "ticker": "Ticker",
                            "accuracy": "Accuracy (%)",
                            "roc_auc": "ROC-AUC",
                            "sharpe": "Sharpe Ratio",
                            "return": "Strategy Return (%)",
                            "precision": "Precision (%)",
                            "recall": "Recall (%)",
                            "f1": "F1 Score",
                        },
                        inplace=True,
                    )
                    return df_lead
                except Exception:
                    pass

            rows = []
            for f in glob.glob("results/*_metrics.json"):
                t = os.path.basename(f).replace("_metrics.json", "")
                try:
                    with open(f, "r", encoding="utf-8") as jf:
                        d = json.load(jf)
                        if "accuracy" in d and d["accuracy"] is not None:
                            rows.append(
                                {
                                    "Ticker": t,
                                    "Accuracy (%)": round(
                                        float(d.get("accuracy", 0)) * 100, 2
                                    ),
                                    "ROC-AUC": round(float(d.get("roc_auc", 0)), 4),
                                    "Sharpe Ratio": round(
                                        float(d.get("sharpe_ratio", 0)), 2
                                    ),
                                    "Strategy Return (%)": round(
                                        float(d.get("strategy_total_return", 0)) * 100,
                                        2,
                                    ),
                                    "Precision (%)": round(
                                        float(d.get("precision", 0)) * 100, 2
                                    ),
                                    "Recall (%)": round(
                                        float(d.get("recall", 0)) * 100, 2
                                    ),
                                    "F1 Score": round(float(d.get("f1", 0)), 4),
                                }
                            )
                except Exception:
                    pass
            df_lead = pd.DataFrame(rows)
            if not df_lead.empty:
                df_lead = df_lead.sort_values(by="Accuracy (%)", ascending=False)
            return df_lead

        df_leaderboard = _get_universe_leaderboard()
        if not df_leaderboard.empty:
            l_col1, l_col2, l_col3, l_col4 = st.columns(4)
            l_col1.metric("🌐 Universe Evaluated", f"{len(df_leaderboard)} Tickers")
            l_col2.metric(
                "🎯 Mean Accuracy",
                f"{df_leaderboard['Accuracy (%)'].mean():.2f}%",
            )
            l_col3.metric(
                "⚡ Mean Sharpe Ratio",
                f"{df_leaderboard['Sharpe Ratio'].mean():.2f}",
            )
            l_col4.metric(
                "🏆 Top Accuracy Asset",
                f"{df_leaderboard.iloc[0]['Ticker']} ({df_leaderboard.iloc[0]['Accuracy (%)']}%)",
            )

            st.dataframe(
                df_leaderboard,
                use_container_width=True,
                height=350,
                hide_index=True,
            )
