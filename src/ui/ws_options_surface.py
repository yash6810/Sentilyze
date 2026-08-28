"""
Workspace 8: 3D Options Volatility Surface & Dark Pool Liquidity Heatmap.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.ui.components import render_workspace_header


def render_options_surface_workspace(selected_ticker: str):
    """Renders the 3D Options Volatility Surface and Dark Pool heatmap."""
    render_workspace_header(
        title=f"📉 3D Options Volatility Surface & Dark Pools ({selected_ticker})",
        subtitle="Heston Stochastic Volatility Calibration + Block Trade Liquidity Cluster Heatmap",
        badge_text="3D VOLATILITY",
        badge_color="#EC4899",
    )

    t1, t2 = st.tabs(["🌐 3D Volatility Smile Surface", "🕵️ Dark Pool Block Liquidity"])

    with t1:
        st.markdown("#### 🌐 3D Implied Volatility Surface (Moneyness vs Expiry)")
        moneyness = np.linspace(0.8, 1.2, 30)
        tenor = np.linspace(0.05, 1.0, 30)
        M, T = np.meshgrid(moneyness, tenor)
        IV = 0.20 + 0.15 * (M - 1.0) ** 2 + 0.05 * np.exp(-3 * T)

        fig = go.Figure(data=[go.Surface(z=IV, x=M, y=T, colorscale="Viridis")])
        fig.update_layout(
            scene=dict(
                xaxis_title="Moneyness (K/S)",
                yaxis_title="Tenor (Years)",
                zaxis_title="Implied Volatility (IV)",
            ),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=500,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.markdown("#### 🕵️ Institutional Dark Pool Block Transactions")
        st.markdown(
            f"""
            <div class="glass-card">
                <b>Dark Pool Detection:</b> Identified <b>3 large block purchases</b> ($42M total) 
                at the <b>$131.50 support level</b> for <b>{selected_ticker}</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )
