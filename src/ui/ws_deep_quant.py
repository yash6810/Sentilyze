"""
Workspaces 9-13: Deep Quantitative Modeling, GNN Supply Chain, Stress Tests & Forensic Valuation.
"""

import streamlit as st
import pandas as pd
from src.ui.components import render_workspace_header
from src.statistical_arbitrage import (
    scan_pairs_universe,
    generate_pairs_trading_signals,
)
from src.gnn_supply_chain import analyze_supply_chain_spillover
from src.stress_tester import run_monte_carlo_stress_test, run_monte_carlo_var
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
                selected_ticker, shock_magnitude_pct=-0.25
            )
        st.json(gnn_res)

    elif mode == "stress":
        render_workspace_header(
            title="🌪️ Black Swan Crisis Simulator & Monte Carlo Audit",
            subtitle="2008 GFC, 2020 Covid Shock, 2022 Fed Rate Hikes & Monte Carlo VaR",
            badge_text="RISK STRESS TEST",
            badge_color="#EF4444",
        )
        with st.spinner("Running Monte Carlo Value-at-Risk simulations..."):
            stress_res = run_monte_carlo_var(selected_ticker)
        st.json(stress_res)

    elif mode == "forensic":
        render_workspace_header(
            title=f"🕵️ Forensic Accounting & Beneish M-Score ({selected_ticker})",
            subtitle="Beneish M-Score Earnings Manipulation Check + Debt Maturity Wall Analysis",
            badge_text="FORENSIC AUDIT",
            badge_color="#F59E0B",
        )
        fin = fetch_financial_statements(selected_ticker)
        m_score = calculate_beneish_m_score(fin)
        debt = analyze_debt_maturity_wall(selected_ticker)
        st.json({"beneish_m_score": m_score, "debt_wall": debt})

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
        st.json(
            {
                "dcf_valuation": dcf,
                "piotroski_f_score": f_score,
                "altman_z_score": z_score,
            }
        )
