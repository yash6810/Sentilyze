"""
Workspace: Institutional Execution Engine, Smart Order Router (SOR) & Microstructure Shield.
Features:
- Almgren-Chriss (2000) optimal trajectory curves.
- Vectorized Intraday VWAP & Anti-Gaming Randomized TWAP.
- Bouchaud et al. (2009) Square-Root Market Impact & Slippage Calculator.
- Live Market Microstructure Health (VPIN Toxicity, Corwin-Schultz Spread, Amihud Illiquidity).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.execution_engine import (
    compute_almgren_chriss_trajectory,
    compute_vwap_schedule,
    compute_twap_schedule,
    compute_bouchaud_market_impact,
)
from src.microstructure import evaluate_microstructure_health
from src.realtime_tracker import fetch_live_quote


def render_execution_sor_workspace(selected_ticker: str = "NVDA"):
    st.markdown("### ⚡ Institutional Execution Engine & Smart Order Router (SOR)")
    st.caption(
        "Almgren-Chriss (2000) optimal liquidation trajectories, Bouchaud (2009) square-root market impact, "
        "and Volume-Synchronized Probability of Toxicity (VPIN) microstructure defense."
    )

    quote = fetch_live_quote(selected_ticker)
    spot_price = float(quote.get("price", 150.0))

    # 1. Market Microstructure Health Audit
    st.markdown(f"#### 🛡️ Microstructure Health & Toxic Flow Audit: `{selected_ticker}`")
    with st.spinner("Analyzing high-frequency volume buckets & bid-ask spreads..."):
        micro_res = evaluate_microstructure_health(selected_ticker)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        score = micro_res.get("microstructure_score", 85.0)
        st.metric(
            "Microstructure Score",
            f"{score:.1f} / 100",
            "Institutional Flow Quality",
            delta_color="normal" if score >= 70 else "inverse",
        )
    with m2:
        vpin = micro_res.get("vpin", 0.32)
        regime = micro_res.get("vpin_regime", "CLEAN_FLOW")
        st.metric("VPIN Flow Toxicity", f"{vpin:.3f}", regime)
    with m3:
        cs_spread = micro_res.get("corwin_schultz_spread_bps", 6.5)
        st.metric(
            "Corwin-Schultz Spread", f"{cs_spread:.1f} bps", "Effective High-Low Spread"
        )
    with m4:
        penalty = micro_res.get("action_penalty_multiplier", 1.0)
        st.metric(
            "Execution Multiplier",
            f"{penalty:.2f}x",
            micro_res.get("execution_recommendation", "OPTIMAL"),
        )

    st.markdown("---")

    # 2. Interactive Execution Simulation Controls
    st.markdown("#### 🎯 Smart Order Routing (SOR) Simulator")
    c_in1, c_in2, c_in3 = st.columns(3)
    with c_in1:
        sim_shares = st.number_input(
            "Target Order Size (Shares)",
            min_value=100,
            max_value=500_000,
            value=5_000,
            step=500,
        )
    with c_in2:
        sim_strategy = st.selectbox(
            "Execution Algorithm",
            ["Almgren-Chriss Optimal", "Intraday VWAP", "Stealth Randomized TWAP"],
        )
    with c_in3:
        sim_adv = st.number_input(
            "Average Daily Volume (ADV)",
            min_value=50_000,
            max_value=100_000_000,
            value=25_000_000,
            step=1_000_000,
        )

    # 3. Market Impact Computation (Bouchaud Law)
    impact = compute_bouchaud_market_impact(
        shares=sim_shares,
        spot_price=spot_price,
        avg_daily_volume=sim_adv,
        daily_volatility=0.022,
    )

    im1, im2, im3, im4 = st.columns(4)
    with im1:
        st.metric(
            "Participation Rate",
            f"{impact['participation_rate_pct']:.3f}%",
            "Order / Daily Volume",
        )
    with im2:
        st.metric(
            "Estimated Slippage",
            f"{impact['impact_bps']:.1f} bps",
            "Bouchaud Square-Root Law",
        )
    with im3:
        st.metric(
            "Dollar Slippage Drag",
            f"${impact['dollar_slippage']:.3f}/sh",
            f"Total: ${impact['dollar_slippage']*sim_shares:,.2f}",
        )
    with im4:
        st.metric(
            "Effective Buy Price",
            f"${impact['effective_buy_price']:,.2f}",
            f"Spot: ${spot_price:,.2f}",
        )

    # 4. Trajectory Plot
    if "Almgren" in sim_strategy:
        plan = compute_almgren_chriss_trajectory(
            total_shares=sim_shares, total_time_hours=6.5, n_intervals=13
        )
        schedule = plan["schedule"]
        x_vals = [f"Hr {p['time_hour']:.1f}" for p in schedule]
        tranche_vals = [p["shares_to_execute"] for p in schedule]
        remaining_vals = [p["remaining_shares"] for p in schedule]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Tranche Shares Executed",
                x=x_vals,
                y=tranche_vals,
                marker_color="#3B82F6",
            )
        )
        fig.add_trace(
            go.Scatter(
                name="Remaining Order (Shares)",
                x=x_vals,
                y=remaining_vals,
                yaxis="y2",
                mode="lines+markers",
                line=dict(color="#10B981", width=3),
            )
        )

        fig.update_layout(
            title=f"<b>Almgren-Chriss Optimal Trajectory ({sim_shares:,} shares over 6.5h)</b>",
            yaxis=dict(title="Tranche Shares"),
            yaxis2=dict(title="Remaining Inventory", overlaying="y", side="right"),
            template="plotly_dark",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    elif "VWAP" in sim_strategy:
        plan = compute_vwap_schedule(total_shares=sim_shares, n_intervals=13)
        schedule = plan["schedule"]
        intervals = [f"Bin {p['interval']} (30m)" for p in schedule]
        shares_slice = [p["shares_to_execute"] for p in schedule]
        pcts = [p["pct_of_order"] for p in schedule]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="VWAP Slices", x=intervals, y=shares_slice, marker_color="#8B5CF6"
            )
        )
        fig.update_layout(
            title=f"<b>Intraday VWAP Canonical U-Curve Volume Allocation</b>",
            yaxis_title="Tranche Shares",
            template="plotly_dark",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        plan = compute_twap_schedule(
            total_shares=sim_shares, n_intervals=13, randomize_pct=0.15
        )
        schedule = plan["schedule"]
        intervals = [f"Slice {p['interval']}" for p in schedule]
        shares_slice = [p["shares_to_execute"] for p in schedule]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Anti-Gaming TWAP Slices",
                x=intervals,
                y=shares_slice,
                marker_color="#EC4899",
            )
        )
        fig.update_layout(
            title=f"<b>Anti-Gaming Stealth TWAP (Randomized Slices ±15%)</b>",
            yaxis_title="Tranche Shares",
            template="plotly_dark",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
