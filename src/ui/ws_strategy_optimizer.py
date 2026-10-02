"""
Workspace: Vectorized Strategy Optimizer Sandbox.
Interactive tuning for leverage, confidence thresholds, and ATR stops.
"""

import streamlit as st
import pandas as pd
from src.strategy_optimizer import simulate_strategy_sandbox
from src.ui.components import render_workspace_header


def render_strategy_optimizer_workspace(selected_ticker: str = "NVDA"):
    render_workspace_header(
        title="🧪 Vectorized Strategy Sandbox & Optimizer",
        subtitle=f"Interactive multi-parameter simulation for {selected_ticker}: leverage, confidence hurdles & ATR risk corridors.",
        badge_text="VECTORIZED QUANT SANDBOX",
        badge_color="#3B82F6",
    )

    st.markdown(
        "Fine-tune strategy parameters in real-time to analyze how dynamic leverage, "
        "decision thresholds, and volatility stops alter historical return, Sharpe ratio, and drawdown profiles."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        leverage = st.slider(
            "Dynamic Leverage", min_value=0.5, max_value=3.0, value=1.5, step=0.1
        )
    with c2:
        conf_thresh = st.slider(
            "Confidence Hurdle", min_value=0.45, max_value=0.75, value=0.52, step=0.01
        )
    with c3:
        sl_mult = st.slider(
            "Stop-Loss (ATR mult)", min_value=0.5, max_value=3.5, value=1.5, step=0.1
        )
    with c4:
        tp_mult = st.slider(
            "Take-Profit (ATR mult)", min_value=1.0, max_value=5.0, value=2.5, step=0.1
        )

    c_cap, c_run = st.columns([3, 1])
    with c_cap:
        init_cap = st.number_input(
            "Starting Capital ($)",
            min_value=1000.0,
            max_value=1000000.0,
            value=10000.0,
            step=1000.0,
        )
    with c_run:
        st.write("")
        st.write("")
        st.button(
            "⚡ Run Vectorized Simulation", type="primary", use_container_width=True
        )

    res = simulate_strategy_sandbox(
        ticker=selected_ticker,
        leverage=leverage,
        confidence_threshold=conf_thresh,
        sl_atr_multiplier=sl_mult,
        tp_atr_multiplier=tp_mult,
        initial_capital=init_cap,
    )

    if "error" in res:
        st.warning(f"Simulation note for {selected_ticker}: {res['error']}")
        return

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("💰 Final Equity", f"${res['final_equity']:,.2f}")
    diff_ret = res["total_return_pct"] - res["benchmark_return_pct"]
    m2.metric(
        "📈 Total Return",
        f"{res['total_return_pct']:+.2f}%",
        delta=f"{diff_ret:+.2f}% vs B&H",
    )
    m3.metric("⚡ Sharpe Ratio", f"{res['sharpe_ratio']:.2f}")
    m4.metric(
        "🛡️ Max Drawdown",
        f"{res['max_drawdown_pct']:.2f}%",
        delta="Inverse" if res["max_drawdown_pct"] < -20 else "Controlled",
        delta_color="inverse",
    )
    m5.metric("💎 Calmar Ratio", f"{res['calmar_ratio']:.2f}")
    m6.metric("🔄 Total Trades", f"{res['total_trades']}")

    st.markdown("---")
    st.markdown("#### 📊 Cumulative Equity Curve: Strategy vs Buy & Hold Benchmark")
    if "chart_df" in res and isinstance(res["chart_df"], pd.DataFrame):
        st.line_chart(res["chart_df"], use_container_width=True)

    with st.expander("ℹ️ Parameter Optimization Mechanics"):
        st.markdown(f"""
            - **Ticker:** `{res['ticker']}`
            - **Leverage Multiplier:** `{res['leverage']}x` (Magnifies directional equity returns)
            - **Confidence Filter:** `>= {res['confidence_threshold']:.2f}` (Only trades with high model probability are executed)
            - **ATR Risk Corridor:** SL = `{res['sl_atr_multiplier']}x ATR`, TP = `{res['tp_atr_multiplier']}x ATR`
            - **Benchmark Return:** `{res['benchmark_return_pct']:+.2f}%`
            """)
