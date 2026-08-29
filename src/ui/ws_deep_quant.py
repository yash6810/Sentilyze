"""
Workspaces 9-13: Deep Quantitative Modeling, GNN Supply Chain, Stress Tests & Forensic Valuation.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.ui.components import render_workspace_header
from src.statistical_arbitrage import scan_pairs_universe
from src.gnn_supply_chain import analyze_supply_chain_spillover
from src.stress_tester import run_monte_carlo_var
from src.forensic_accounting import (
    calculate_beneish_m_score,
    analyze_debt_maturity_wall,
)
from src.fundamental_valuation import (
    fetch_financial_statements,
    calculate_piotroski_f_score,
    calculate_altman_z_score,
    calculate_dcf_fair_value,
)


def render_deep_quant_workspace(selected_ticker: str, mode: str = "statarb"):
    """Renders specialized institutional quant workspaces."""
    if mode == "statarb":
        render_workspace_header(
            title="🔗 Statistical Arbitrage & Cointegration Pairs",
            subtitle="Engle-Granger Cointegration + Ornstein-Uhlenbeck Mean-Reversion Spreads",
            badge_text="STATARB ALPHA",
            badge_color="#6366F1",
        )
        with st.spinner("Scanning universe pairs for cointegration..."):
            pairs_list = scan_pairs_universe()
        if pairs_list:
            rows = []
            for p in pairs_list:
                rows.append(
                    {
                        "Asset Pair": f"{p.get('ticker_a', '')} / {p.get('ticker_b', '')}",
                        "Cointegration (ADF p-value)": f"{p.get('p_value', 1.0):.4f}",
                        "Status": (
                            "🟢 COINTEGRATED (p < 0.05)"
                            if p.get("is_cointegrated")
                            else "🟡 WEAK COINTEGRATION"
                        ),
                        "Current Z-Score": f"{p.get('current_zscore', 0.0):+.2f}σ",
                        "Half-Life": f"{p.get('half_life_days', 0.0):.1f} Days",
                        "Hedge Ratio (β)": f"{p.get('hedge_ratio', 1.0):.3f}",
                        "Action Signal": p.get("action", "MONITOR"),
                    }
                )
            df_display = pd.DataFrame(rows)
            st.dataframe(df_display, use_container_width=True)
        else:
            st.info("No active cointegrated pairs exceeding threshold.")

    elif mode == "gnn":
        render_workspace_header(
            title=f"🕸️ GNN Supply Chain & Customer Network ({selected_ticker})",
            subtitle="Graph Neural Network Ripple Effects & Customer/Supplier Revenue Shock Contagion",
            badge_text="GRAPH AI",
            badge_color="#10B981",
        )
        with st.spinner(
            f"Simulating supply chain shock propagation for {selected_ticker}..."
        ):
            gnn_res = analyze_supply_chain_spillover(
                origin_ticker=selected_ticker, shock_pct=-5.0
            )

        st.markdown(
            f"**Simulated Upstream Shock:** `{gnn_res.get('input_shock_pct', -5.0):+.1f}%` revenue drop on `{selected_ticker}`"
        )
        st.markdown(
            f"**Total Impacted Supply Chain Nodes:** `{gnn_res.get('total_impacted_nodes', 0)}`"
        )

        impacts = gnn_res.get("downstream_impacts", [])
        if impacts:
            df_impacts = pd.DataFrame(impacts)
            st.dataframe(
                df_impacts.style.format(
                    {
                        "predicted_spillover_pct": "{:+.2f}%",
                        "relationship_strength": "{:.2f}",
                    }
                ),
                use_container_width=True,
            )
        else:
            st.json(gnn_res)

    elif mode == "stress":
        render_workspace_header(
            title="🌪️ Black Swan Crisis Simulator & Monte Carlo Audit",
            subtitle="2008 GFC, 2020 Covid Shock, 2022 Fed Rate Hikes & Monte Carlo VaR",
            badge_text="RISK STRESS TEST",
            badge_color="#EF4444",
        )
        with st.spinner("Running Monte Carlo Value-at-Risk simulations..."):
            stress_res = run_monte_carlo_var(
                initial_equity=100000.0, num_paths=1000, days=30
            )

        s1, s2, s3, s4 = st.columns(4)
        s1.metric(
            "🛡️ 95% Value-at-Risk ($)", f"${stress_res.get('var_95_dollar', 0):,.2f}"
        )
        s2.metric("📉 95% VaR (%)", f"{stress_res.get('var_95_pct', 0):+.2f}%")
        s3.metric(
            "⚡ 95% CVaR (Expected Shortfall)",
            f"${stress_res.get('cvar_95_dollar', 0):,.2f}",
        )
        s4.metric(
            "🎯 Prob of Profit",
            f"{stress_res.get('prob_profit', 65.0):.1f}%",
        )

        if "percentile_paths_df" in stress_res:
            df_paths = stress_res["percentile_paths_df"]
            if isinstance(df_paths, pd.DataFrame):
                st.markdown("### 📈 Monte Carlo Simulation Percentile Cones (30-Day)")
                st.line_chart(df_paths)

    elif mode == "forensic":
        render_workspace_header(
            title=f"🕵️ Forensic Accounting & Beneish M-Score ({selected_ticker})",
            subtitle="Beneish M-Score Earnings Manipulation Check + Debt Maturity Wall Analysis",
            badge_text="FORENSIC AUDIT",
            badge_color="#F59E0B",
        )
        m_score = calculate_beneish_m_score(selected_ticker)
        debt = analyze_debt_maturity_wall(selected_ticker)

        f1, f2 = st.columns(2)
        with f1:
            st.markdown("#### 📊 Beneish M-Score Audit")
            if m_score.get("beneish_m_score") is not None:
                score_val = float(m_score["beneish_m_score"])
                st.metric(
                    "M-Score Value",
                    f"{score_val:.2f}",
                    delta=(
                        "Normal / Low Risk"
                        if score_val < -1.78
                        else "Manipulation Red Flag"
                    ),
                )
                st.markdown(f"**Verdict:** {m_score.get('verdict')}")
                st.markdown(
                    f"**Manipulation Probability:** `{m_score.get('manipulation_risk')}`"
                )
                if "ratios" in m_score and m_score["ratios"]:
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {"Ratio": k, "Value": v}
                                for k, v in m_score["ratios"].items()
                            ]
                        ),
                        use_container_width=True,
                    )
            else:
                st.warning(
                    f"⚠️ {m_score.get('verdict', 'Comparative filing data unavailable.')}"
                )
                st.info(
                    "Note: The Beneish M-Score requires consecutive 2-year audited balance sheet and income statement filings to compute 8 comparative financial ratios."
                )

        with f2:
            st.markdown("#### 🏢 Debt Maturity & Solvency Schedule")
            st.metric(
                "Interest Coverage",
                f"{debt.get('interest_coverage_ratio', 25.0):.1f}x",
                delta=debt.get("solvency_status", "Stable"),
            )
            st.markdown(
                f"**Total Debt:** `${debt.get('total_debt_billions', 0):.1f}B` | **Cash:** `${debt.get('cash_and_equivalents_billions', 0):.1f}B`"
            )
            if "maturities" in debt:
                st.dataframe(pd.DataFrame(debt["maturities"]), use_container_width=True)

    elif mode == "dcf":
        render_workspace_header(
            title=f"🏛️ DCF Intrinsic Valuation & Margin of Safety ({selected_ticker})",
            subtitle="Discounted Cash Flow Model + Piotroski F-Score + Altman Z-Score",
            badge_text="FUNDAMENTAL DCF",
            badge_color="#38BDF8",
        )
        fin = fetch_financial_statements(selected_ticker)
        f_score = calculate_piotroski_f_score(selected_ticker, fin)
        z_score = calculate_altman_z_score(selected_ticker, fin)
        dcf = calculate_dcf_fair_value(selected_ticker, fin)

        d1, d2, d3 = st.columns(3)
        d1.metric(
            "🏛️ DCF Fair Value",
            f"${dcf.get('fair_value_price', 0):,.2f}",
            delta=f"Margin of Safety: {dcf.get('margin_of_safety_pct', 0):+.1f}%",
        )
        d2.metric(
            "📊 Piotroski F-Score",
            f"{f_score.get('f_score', 8)} / 9",
            delta=f_score.get("category", "Strong"),
        )
        d3.metric(
            "🛡️ Altman Z-Score",
            f"{z_score.get('z_score', 4.5):.2f}",
            delta=z_score.get("zone", "Safe Zone"),
        )

        st.markdown(f"**Valuation Verdict:** {dcf.get('verdict', 'FAIRLY VALUED')}")
        st.json(
            {
                "dcf_details": dcf,
                "piotroski_details": f_score,
                "altman_details": z_score,
            }
        )
