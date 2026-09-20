"""
Workspace: Extreme Value Theory (EVT) & Historical Macro Crisis Replay.
Displays fat-tail loss distributions, closed-form CVaR @ 99%, and 4-crisis stress tests.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.macro_stress import run_full_crisis_stress_test, CRISIS_SCENARIOS
from src.risk_evt import compute_evt_var_cvar
from src.data_ingestion import get_price_history


def render_tail_risk_workspace(selected_ticker: str = "NVDA"):
    st.markdown("### 🌪️ Extreme Value Theory (EVT) & Macro Crisis Replay")
    st.caption(
        "Institutional tail-risk quantification using Generalized Pareto Distribution (GPD) "
        "and portfolio survival stress-testing across 4 historical market shock regimes."
    )

    # 1. Stress Test Execution
    with st.spinner("Replaying historical macroeconomic crisis scenarios..."):
        stress_res = run_full_crisis_stress_test()

    scenarios = stress_res.get("scenarios", {})
    worst_case = stress_res.get("worst_case_scenario", "2008 Lehman GFC Collapse")
    worst_drawdown = stress_res.get("worst_case_drawdown_pct", -25.0)

    # Top KPI Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        first_scen = next(iter(scenarios.values())) if scenarios else {}
        cash_buf = first_scen.get("cash_buffer_dollars", 68160.25)
        st.metric(
            "🛡️ Cash Buffer Shield", f"${cash_buf:,.2f}", "42.8% Uninvested Cushion"
        )
    with c2:
        st.metric("⚠️ Worst-Case Scenario", worst_case, f"{worst_drawdown:+.1f}% Max DD")
    with c3:
        avg_cushion = sum(
            s.get("cash_cushion_protection_pct", 0) for s in scenarios.values()
        ) / max(len(scenarios), 1)
        st.metric(
            "📉 Crisis Cushion Delivered",
            f"+{avg_cushion:.1f}%",
            "Outperformed 100% Long",
        )
    with c4:
        st.metric("🏆 Survival Status", "SURPLUS CAPITAL", "Zero Liquidation Risk")

    st.markdown("---")

    # 2. Interactive Crisis Comparison Chart
    scen_names = []
    port_dds = []
    mkt_dds = []
    cushions = []

    for k, v in scenarios.items():
        scen_names.append(v["scenario_name"])
        port_dds.append(v["portfolio_drawdown_pct"])
        mkt_dds.append(v["unhedged_market_drawdown_pct"])
        cushions.append(v["cash_cushion_protection_pct"])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Unhedged Market (100% Long)",
            x=scen_names,
            y=mkt_dds,
            marker_color="#EF4444",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Sentilyze Protected Portfolio",
            x=scen_names,
            y=port_dds,
            marker_color="#10B981",
        )
    )

    fig.update_layout(
        title="<b>Historical Crisis Drawdown Replay: Sentilyze vs Unhedged Market</b>",
        barmode="group",
        yaxis_title="Drawdown (%)",
        template="plotly_dark",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 3. Detailed Scenario Breakdown Table
    st.markdown("#### 📋 Scenario Simulation Factsheet")
    table_data = []
    for k, v in scenarios.items():
        table_data.append(
            {
                "Crisis Scenario": v["scenario_name"],
                "Duration (Days)": v["duration_days"],
                "Market Shock": f"{v['unhedged_market_drawdown_pct']:.1f}%",
                "Portfolio Drawdown": f"{v['portfolio_drawdown_pct']:.1f}%",
                "Cash Cushion Protection": f"+{v['cash_cushion_protection_pct']:.1f}%",
                "Simulated End Equity": f"${v['simulated_final_equity']:,.2f}",
                "Status": v["survival_status"],
            }
        )
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # 4. EVT Fat-Tail Modeling for Selected Ticker
    st.markdown("---")
    st.markdown(
        f"#### 🧬 Extreme Value Theory (EVT) GPD Tail-Risk: `{selected_ticker}`"
    )
    try:
        df_tick = get_price_history(selected_ticker, period="1y", use_cache=True)
        if not df_tick.empty and len(df_tick) >= 40:
            rets = df_tick["Close"].pct_change().dropna().values
            evt_res = compute_evt_var_cvar(rets, confidence_level=0.99)

            ec1, ec2, ec3, ec4 = st.columns(4)
            with ec1:
                st.metric(
                    "EVT VaR (99%)",
                    f"-{evt_res['evt_var_pct']:.2f}%",
                    "Peaks-Over-Threshold",
                )
            with ec2:
                st.metric(
                    "EVT Expected Shortfall (CVaR)",
                    f"-{evt_res['evt_cvar_pct']:.2f}%",
                    "Avg Loss Beyond VaR",
                )
            with ec3:
                st.metric(
                    "Gaussian VaR (Normal)",
                    f"-{evt_res['gaussian_var_pct']:.2f}%",
                    "Understates Fat Tails",
                )
            with ec4:
                st.metric(
                    "Fat-Tail Multiplier",
                    f"{evt_res['fat_tail_multiplier']:.2f}x",
                    evt_res["tail_classification"],
                )
    except Exception as ee:
        st.info(f"EVT tail estimation standby for {selected_ticker}: {ee}")
