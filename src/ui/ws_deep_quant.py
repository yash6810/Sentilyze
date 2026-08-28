"""
Workspaces 9-15: Deep Quantitative Modeling, GNN Supply Chain, Stress Tests & Forensic Valuation.
"""

import streamlit as st
from src.ui.components import render_workspace_header
from src.statistical_arbitrage import run_pair_trading_scan
from src.gnn_supply_chain import run_gnn_contagion_analysis
from src.stress_tester import run_full_crisis_simulation_suite
from src.forensic_accounting import evaluate_forensic_accounting
from src.fundamental_valuation import run_fundamental_valuation


def render_deep_quant_workspace(selected_ticker: str, mode: str = "statarb"):
    """Renders specialized institutional quant workspaces."""
    if mode == "statarb":
        render_workspace_header(
            title="🔗 Statistical Arbitrage & Cointegration Pairs",
            subtitle="Engle-Granger Cointegration + Ornstein-Uhlenbeck Mean-Reversion Spreads",
            badge_text="STATARB ALPHA",
            badge_color="#6366F1",
        )
        pairs_data = run_pair_trading_scan()
        st.dataframe(pairs_data, use_container_width=True)

    elif mode == "gnn":
        render_workspace_header(
            title=f"🕸️ GNN Supply Chain & Customer Network ({selected_ticker})",
            subtitle="Graph Neural Network Ripple Effects & Customer/Supplier Revenue Shock Contagion",
            badge_text="GRAPH AI",
            badge_color="#10B981",
        )
        gnn_res = run_gnn_contagion_analysis(selected_ticker)
        st.json(gnn_res)

    elif mode == "stress":
        render_workspace_header(
            title="🌪️ Black Swan Crisis Simulator & Monte Carlo Audit",
            subtitle="2008 GFC, 2020 Covid Shock, 2022 Fed Rate Hikes & 50,000 Monte Carlo Paths",
            badge_text="RISK STRESS TEST",
            badge_color="#EF4444",
        )
        stress_res = run_full_crisis_simulation_suite()
        st.json(stress_res)

    elif mode == "forensic":
        render_workspace_header(
            title=f"🕵️ Forensic Accounting & Beneish M-Score ({selected_ticker})",
            subtitle="Beneish M-Score Manipulation Check + Altman Z-Score Bankruptcy Distance",
            badge_text="FORENSIC AUDIT",
            badge_color="#F59E0B",
        )
        forensic = evaluate_forensic_accounting(selected_ticker)
        st.json(forensic)

    else:
        render_workspace_header(
            title=f"🏛️ DCF Intrinsic Valuation & Margin of Safety ({selected_ticker})",
            subtitle="Discounted Cash Flow Model + Monte Carlo Terminal Growth Sensitivity",
            badge_text="FUNDAMENTAL DCF",
            badge_color="#38BDF8",
        )
        val = run_fundamental_valuation(selected_ticker)
        st.json(val)
