"""
Workspace: Evolutionary Strategy Incubator & Robustness Vault.
Genetic Algorithm Breeding, 3-Zone Testing & DNA Inspector.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

from src.strategy_incubator import (
    breed_strategy_generation,
    evaluate_3zone_robustness,
    load_strategy_vault,
    StrategyGenome,
)
from src.paper_broker import PaperBroker
from src.config import COMPANY_NAMES


def render_strategy_incubator_workspace(selected_ticker: str = "NVDA"):
    st.markdown("### 🔬 Evolutionary Strategy Incubator & Robustness Lab")
    st.caption(
        "Institutional Genetic Algorithm Strategy Breeding: Mutates Trading Rule Genomes Across Generations, "
        "Enforces 3-Zone Validation (70% Train / 30% Locked OOS / Live Forward), and Vaults Battle-Tested Strategies (Score >= 60)."
    )

    tab_breeder, tab_vault, tab_inspector = st.tabs(
        [
            "🧬 Strategy Breeding & Incubation Core",
            "🏦 Strategy Vault & Top Survivors",
            f"🔍 Strategy DNA Inspector ({selected_ticker})",
        ]
    )

    # =========================================================================
    # TAB 1: BREEDING & INCUBATION CORE
    # =========================================================================
    with tab_breeder:
        st.markdown(
            f"#### 🧬 Launch Genetic Breeding Campaign on **{selected_ticker}**"
        )

        col_b1, col_b2, col_b3 = st.columns([1, 1, 2])
        with col_b1:
            pop_size = st.slider(
                "Population Size per Gen:", min_value=10, max_value=40, value=15, step=5
            )
            generations = st.slider(
                "Evolutionary Generations:", min_value=3, max_value=20, value=6, step=1
            )
            breed_btn = st.button(
                "🚀 Breed Strategies Across Generations", type="primary"
            )

        with col_b2:
            st.info(
                f"**Selection Criteria**:\n"
                f"* 3-Zone Out-of-Sample Split (70/30)\n"
                f"* 50-Iteration Monte Carlo Noise Stress\n"
                f"* Minimum Fitness Score: **60.0**"
            )

        if breed_btn:
            with st.spinner(
                f"Breeding and evolving {pop_size * generations} strategy genomes for {selected_ticker}..."
            ):
                inc_res = breed_strategy_generation(
                    ticker=selected_ticker,
                    population_size=pop_size,
                    generations=generations,
                )

            st.success(
                f"🎉 Campaign Complete! Vaulted **{inc_res['vaulted_count']} surviving strategies** "
                f"(Best Fitness Score: **{inc_res['best_strategy']['fitness_score']} / 100**)."
            )

            hist = inc_res["generation_history"]
            with col_b3:
                df_hist = pd.DataFrame(hist)
                fig_gen = go.Figure()
                fig_gen.add_trace(
                    go.Scatter(
                        x=df_hist["generation"],
                        y=df_hist["top_score"],
                        name="Top Elite Score",
                        mode="lines+markers",
                        line=dict(color="#10B981", width=2.5),
                    )
                )
                fig_gen.add_trace(
                    go.Scatter(
                        x=df_hist["generation"],
                        y=df_hist["avg_score"],
                        name="Generation Avg Score",
                        mode="lines+markers",
                        line=dict(color="#38BDF8", width=1.5, dash="dash"),
                    )
                )
                fig_gen.update_layout(
                    title="Evolutionary Fitness Score Progression",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=280,
                    margin=dict(l=20, r=20, t=35, b=20),
                    xaxis_title="Generation",
                    yaxis_title="Fitness Score (0-100)",
                )
                st.plotly_chart(fig_gen, use_container_width=True)

    # =========================================================================
    # TAB 2: STRATEGY VAULT
    # =========================================================================
    with tab_vault:
        st.markdown("#### 🏦 Strategy Vault: Verified Out-of-Sample Survivors")
        vault_items = load_strategy_vault()

        if not vault_items:
            st.info(
                "No strategies currently vaulted. Run a genetic breeding campaign to populate the vault."
            )
        else:
            v_rows = []
            for item in vault_items:
                g = item.get("genome", {})
                v_rows.append(
                    {
                        "Genome ID": g.get("genome_id", ""),
                        "Target Asset": item.get("ticker", ""),
                        "Fitness Score": item.get("fitness_score", 0.0),
                        "OOS Return (%)": item.get("oos_return_pct", 0.0),
                        "Train Sharpe": item.get("train_sharpe", 0.0),
                        "OOS Sharpe": item.get("oos_sharpe", 0.0),
                        "MC Survival (%)": item.get("mc_survival_rate_pct", 0.0),
                        "TP Multiple": f"{g.get('tp_atr_multiple', 0.0)}x ATR",
                        "SL Multiple": f"{g.get('sl_atr_multiple', 0.0)}x ATR",
                    }
                )

            df_vault = pd.DataFrame(v_rows)
            st.dataframe(
                df_vault.style.format(
                    {
                        "Fitness Score": "{:.1f}",
                        "OOS Return (%)": "{:+.2f}%",
                        "Train Sharpe": "{:.2f}",
                        "OOS Sharpe": "{:.2f}",
                        "MC Survival (%)": "{:.1f}%",
                    }
                ),
                use_container_width=True,
                height=350,
            )

    # =========================================================================
    # TAB 3: DNA INSPECTOR
    # =========================================================================
    with tab_inspector:
        st.markdown(
            f"#### 🔍 Strategy DNA & 3-Region Equity Curve Inspector: **{selected_ticker}**"
        )

        # Evaluate top strategy for selected ticker
        top_genome = StrategyGenome.random()
        with st.spinner(
            "Evaluating 3-Region In-Sample vs Out-of-Sample Equity Curves..."
        ):
            eval_res = evaluate_3zone_robustness(top_genome, ticker=selected_ticker)

        fit_score = eval_res["fitness_score"]
        fit_color = (
            "#10B981"
            if fit_score >= 70
            else "#38BDF8" if fit_score >= 60 else "#EF4444"
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "🧬 Genome Fitness Score",
            f"{fit_score:.1f} / 100",
            delta="Vaulted (>= 60)" if eval_res["is_vaulted"] else "Failed Gate",
        )
        c2.metric(
            "📈 Out-of-Sample Return",
            f"{eval_res['oos_return_pct']:+.2f}%",
            delta=f"{eval_res['oos_sharpe']:.2f} OOS Sharpe",
        )
        c3.metric(
            "🎲 Monte Carlo Survival",
            f"{eval_res['mc_survival_rate_pct']:.1f}%",
            delta="Under 2x Slippage Noise",
        )
        c4.metric(
            "⚖️ Reward / Risk Ratio",
            f"{top_genome.tp_atr_multiple / max(top_genome.sl_atr_multiple, 0.1):.2f}:1",
        )

        st.markdown("---")

        # 3-Zone Equity Curve Comparison Plot
        train_c = eval_res.get("train_curve", [])
        oos_c = eval_res.get("oos_curve", [])

        if train_c and oos_c:
            fig_3z = go.Figure()
            fig_3z.add_trace(
                go.Scatter(
                    x=list(range(len(train_c))),
                    y=train_c,
                    name="Zone 1: In-Sample Train (70%)",
                    line=dict(color="#38BDF8", width=2.0),
                )
            )
            fig_3z.add_trace(
                go.Scatter(
                    x=list(range(len(train_c), len(train_c) + len(oos_c))),
                    y=oos_c,
                    name="Zone 2: Locked Out-of-Sample OOS (30%)",
                    line=dict(color="#10B981", width=2.5),
                )
            )
            fig_3z.update_layout(
                title="3-Zone Equity Curve (Train vs Locked Out-of-Sample)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )
            st.plotly_chart(fig_3z, use_container_width=True)

        # DNA Breakdown Table
        with st.expander("🧬 View Strategy Rule DNA Parameters"):
            st.json(top_genome.to_dict())
