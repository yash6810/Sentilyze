"""
Workspace: Deep Reinforcement Learning (DRL) Autonomous Policy Agent.
Actor-Critic Neural Policy, Dynamic Leverage Sizing & Volatility Shock Testing.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

from src.drl_policy_agent import (
    evaluate_drl_policy_action,
    train_drl_policy,
)
from src.paper_broker import PaperBroker
from src.config import COMPANY_NAMES


def render_drl_agent_workspace(selected_ticker: str = "NVDA"):
    st.markdown("### 🤖 Deep Reinforcement Learning (DRL) Autonomous Policy Agent")
    st.caption(
        "Autonomous Deep RL Policy Network: Uses PyTorch Continuous Actor-Critic Architecture "
        "and Sortino-Penalized Rewards to Learn Optimal Dynamic Position Sizing & Volatility De-Risking."
    )

    tab_policy, tab_training, tab_simulator = st.tabs(
        [
            f"🎯 Live Policy Telemetry ({selected_ticker})",
            "🧪 Policy Training & Reward Convergence",
            "⚡ Volatility Shock Scenario Simulator",
        ]
    )

    # =========================================================================
    # TAB 1: LIVE POLICY TELEMETRY
    # =========================================================================
    with tab_policy:
        st.markdown(
            f"#### 🧠 Neural Policy Action Recommendation: **{selected_ticker}** ({COMPANY_NAMES.get(selected_ticker, selected_ticker)})"
        )

        with st.spinner(
            "Evaluating 6-dimensional continuous state vector in PyTorch..."
        ):
            policy_res = evaluate_drl_policy_action(selected_ticker)

        act_label = policy_res["action_label"]
        act_color = policy_res["action_color"]
        rec_lev = policy_res["recommended_leverage"]
        target_alloc = policy_res["target_allocation_pct"]
        val_score = policy_res["state_value_score"]

        # Action Scorecard Banner
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
                        border: 1px solid {act_color}; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <span style="font-size: 13px; letter-spacing: 2px; text-transform: uppercase; color: #94A3B8;">
                            ACTOR-CRITIC DRL POLICY DECISION
                        </span>
                        <div style="font-size: 26px; font-weight: 800; color: #FFFFFF; margin-top: 4px;">
                            {act_label}
                        </div>
                        <div style="font-size: 14px; color: #CBD5E1; margin-top: 6px;">
                            {policy_res.get('action_summary', '')}
                        </div>
                    </div>
                    <div style="background: {act_color}22; border: 2px solid {act_color}; border-radius: 16px;
                                padding: 12px 28px; text-align: center;">
                        <div style="font-size: 38px; font-weight: 900; color: {act_color}; line-height: 1;">
                            {rec_lev:.2f}x
                        </div>
                        <div style="font-size: 10px; letter-spacing: 1px; color: #E2E8F0; margin-top: 4px;">
                            RECOMMENDED LEVERAGE
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4 Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "🎯 Target Capital Allocation",
            f"{target_alloc:.1f}%",
            delta="Continuous Sizing",
        )
        m2.metric(
            "🛡️ Critic State Value V(s)",
            f"{val_score:.2f}",
            delta="Favorable State" if val_score > 0.5 else "Challenging State",
        )
        m3.metric(
            "📊 5-Day Momentum Input",
            f"{policy_res['input_state']['momentum_pct']:+.2f}%",
        )
        m4.metric(
            "🌪️ Realized Volatility Input",
            f"{policy_res['input_state']['volatility_pct']:.2f}%",
        )

    # =========================================================================
    # TAB 2: TRAINING & REWARD CONVERGENCE
    # =========================================================================
    with tab_training:
        st.markdown(f"#### 🧪 Retrain Neural Policy Network on {selected_ticker}")
        st.caption(
            "Runs fast CPU policy gradient optimization using asymmetric Sortino-penalized rewards."
        )

        col_tr1, col_tr2 = st.columns([1, 2])
        with col_tr1:
            episodes = st.slider(
                "Training Episodes:", min_value=10, max_value=100, value=30, step=10
            )
            lr = st.select_slider(
                "Learning Rate (Adam):",
                options=[0.0005, 0.001, 0.002, 0.005],
                value=0.002,
            )
            train_btn = st.button("🚀 Train DRL Policy Network", type="primary")

        if train_btn:
            with st.spinner(
                f"Training Actor-Critic network across {episodes} market episodes..."
            ):
                train_res = train_drl_policy(
                    selected_ticker, episodes=episodes, learning_rate=lr
                )

            st.success(
                f"✅ Training Complete! Final Capital: **${train_res['final_capital']:,.2f}** "
                f"(from $100,000 baseline across {episodes} episodes)."
            )

            rewards = train_res["learning_curve_rewards"]
            capitals = train_res["final_capitals"]

            with col_tr2:
                fig_tr = go.Figure()
                fig_tr.add_trace(
                    go.Scatter(
                        y=rewards,
                        mode="lines+markers",
                        name="Episode Reward (Sortino)",
                        line=dict(color="#10B981", width=2.5),
                    )
                )
                fig_tr.update_layout(
                    title="Policy Reward Learning Curve",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=300,
                    margin=dict(l=20, r=20, t=35, b=20),
                    xaxis_title="Episode",
                    yaxis_title="Total Reward",
                )
                st.plotly_chart(fig_tr, use_container_width=True)

    # =========================================================================
    # TAB 3: VOLATILITY SHOCK SIMULATOR
    # =========================================================================
    with tab_simulator:
        st.markdown("#### ⚡ Real-Time Market Shock Simulator")
        st.caption(
            "Move the market state sliders to observe how the neural policy dynamically adapts continuous leverage in real-time."
        )

        sim_c1, sim_c2 = st.columns(2)
        with sim_c1:
            sim_mom = (
                st.slider(
                    "Simulated 5-Day Momentum (%):",
                    min_value=-10.0,
                    max_value=15.0,
                    value=2.5,
                    step=0.5,
                )
                / 100.0
            )
            sim_vol = (
                st.slider(
                    "Simulated Realized Volatility (%):",
                    min_value=0.5,
                    max_value=8.0,
                    value=1.8,
                    step=0.1,
                )
                / 100.0
            )
            sim_sent = st.slider(
                "Simulated News Sentiment:",
                min_value=-1.0,
                max_value=1.0,
                value=0.6,
                step=0.1,
            )
        with sim_c2:
            sim_insider = st.slider(
                "Simulated Insider Conviction:",
                min_value=0.0,
                max_value=1.0,
                value=0.75,
                step=0.05,
            )
            sim_dd = (
                st.slider(
                    "Simulated Portfolio Drawdown (%):",
                    min_value=-15.0,
                    max_value=0.0,
                    value=-2.0,
                    step=0.5,
                )
                / 100.0
            )

        sim_out = evaluate_drl_policy_action(
            ticker=selected_ticker,
            recent_momentum=sim_mom,
            current_volatility=sim_vol,
            sentiment_score=sim_sent,
            insider_score=sim_insider,
            current_drawdown=sim_dd,
        )

        st.markdown("---")
        st.markdown(
            f"**Simulated Policy Decision**: `{sim_out['action_label']}` &nbsp;|&nbsp; **Recommended Leverage**: `{sim_out['recommended_leverage']:.2f}x` ({sim_out['target_allocation_pct']:.1f}% Allocation)"
        )
