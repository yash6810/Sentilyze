"""
Workspace: Synthetic Crash Diffusion Engine & Black Swan Stress Simulator.
Evaluates portfolio tail risk, VaR 99%, CVaR 99% across historical crisis scenarios (2008, 2020, 2022, 1987).
"""

import streamlit as st
import pandas as pd
from src.stress_simulator import run_portfolio_stress_test
from src.ui.components import render_workspace_header
from src.paper_broker import PaperBroker


def render_stress_simulator_workspace():
    render_workspace_header(
        title="💥 Synthetic Crash & Black Swan Stress Simulator",
        subtitle="Denoising Diffusion Probabilistic Model (DDPM) simulating non-Gaussian fat-tailed crash paths across historical crises.",
        badge_text="DIFFUSION CRISIS REPLAY",
        badge_color="#EF4444",
    )

    broker = PaperBroker()
    current_eq = broker.state.get("total_equity", 145000.0)

    c1, c2 = st.columns([3, 1])
    with c1:
        equity_input = st.number_input(
            "Portfolio Equity Under Stress ($)",
            min_value=10000.0,
            max_value=10000000.0,
            value=float(current_eq),
            step=5000.0,
            help="Simulate severe systemic market shocks against this capital balance.",
        )
    with c2:
        st.write("")
        st.write("")
        st.button(
            "🚨 Run Black Swan Stress Test", type="primary", use_container_width=True
        )

    with st.spinner("Simulating 400+ fat-tailed synthetic diffusion trajectories..."):
        res = run_portfolio_stress_test(equity=equity_input)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "🛡️ Capital Preservation Score",
        f"{res.get('capital_preservation_score', 0.0):.1f}/100",
        delta=(
            "HIGH DEFENSE"
            if res.get("capital_preservation_score", 0) > 70
            else "VULNERABLE"
        ),
    )
    k2.metric(
        "📉 99% VaR (Value-at-Risk)",
        f"${res.get('var_99_dollars', 0.0):,.2f}",
        delta=f"{res.get('var_99_pct', 0.0):.1f}%",
        delta_color="inverse",
    )
    k3.metric(
        "⚡ 99% CVaR (Expected Shortfall)",
        f"${res.get('cvar_99_dollars', 0.0):,.2f}",
        delta=f"{res.get('cvar_99_expected_shortfall_pct', 0.0):.1f}%",
        delta_color="inverse",
    )
    k4.metric("🎲 Crisis Paths Simulated", f"{res.get('simulations_count', 400):,}")

    st.markdown("---")
    st.subheader("📜 Historical Shock Replay Scenarios")

    scenarios = res.get("historical_crisis_scenarios", {})
    if scenarios:
        rows = []
        for name, data in scenarios.items():
            rows.append(
                {
                    "Crisis Scenario": name.replace("_", " ").title(),
                    "Average Terminal Equity ($)": data.get(
                        "average_terminal_equity", 0.0
                    ),
                    "Worst-Case Drawdown (%)": data.get("worst_case_drawdown_pct", 0.0),
                    "Median Drawdown (%)": data.get("median_drawdown_pct", 0.0),
                    "Recovery Resilience": data.get("recovery_resilience", "UNKNOWN"),
                }
            )
        df_scenarios = pd.DataFrame(rows)
        st.dataframe(
            df_scenarios,
            use_container_width=True,
            column_config={
                "Average Terminal Equity ($)": st.column_config.NumberColumn(
                    format="$%.2f"
                ),
                "Worst-Case Drawdown (%)": st.column_config.NumberColumn(
                    format="%.2f%%"
                ),
                "Median Drawdown (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "Recovery Resilience": st.column_config.TextColumn("Resilience"),
            },
            hide_index=True,
        )

    with st.expander("ℹ️ Diffusion Noise & Extreme Value Theory Math"):
        st.markdown("""
            - **Diffusion Model (DDPM):** Lightweight 1D reverse diffusion generating multi-step non-Gaussian jump-diffusion asset trajectories conditioned on historical crisis parameters.
            - **Historical Shocks:**
              * **2008 Lehman Collapse:** -40% equity drawdown with severe liquidity freeze.
              * **2020 COVID Flash Crash:** -34% rapid compression over 23 trading sessions.
              * **2022 Fed Rate Surge:** -28% multiple compression in technology and growth assets.
              * **1987 Black Monday:** -22.6% single-day catastrophic portfolio gap down.
            """)
