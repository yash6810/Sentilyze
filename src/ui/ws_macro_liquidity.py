"""
Workspace: Real-Time Macro Liquidity & Treasury Yield Curve Radar.
Visualizes 10Y-2Y Treasury Spread Inversions, Fed Net Liquidity Index,
and Macroeconomic Financial Conditions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.macro_liquidity import calculate_macro_liquidity_metrics
from src.cross_asset_matrix import compute_cross_asset_matrix
from src.data_ingestion import get_price_history
from src.utils import get_logger

logger = get_logger(__name__)


@st.cache_data(ttl=300)
def _get_cached_macro_metrics():
    return calculate_macro_liquidity_metrics()


@st.cache_data(ttl=300)
def _get_cached_cross_asset_matrix():
    return compute_cross_asset_matrix()


def render_macro_liquidity_workspace():
    st.markdown("### 🌐 Real-Time Macro Liquidity & Treasury Yield Radar")
    st.caption(
        "Macroeconomic Liquidity Command: Tracks 10Y-2Y Treasury Yield Curve Inversions, "
        "Federal Reserve Net Liquidity ($L = \\text{Fed Assets} - \\text{TGA} - \\text{RRP}$), "
        "and Systemic Financial Conditions Driving Institutional Equity Flows."
    )

    metrics = _get_cached_macro_metrics()

    # Top KPI Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "📈 10-Yr Treasury Yield",
        f"{metrics['10y_yield']:.2f}%",
        delta=f"2Y: {metrics['2y_yield']:.2f}%",
    )
    m2.metric(
        "⚖️ 10Y - 2Y Spread",
        f"{metrics['spread_10_2_bps']:+.1f} bps",
        delta=metrics["yield_regime"],
    )
    m3.metric(
        "🏦 Fed Net Liquidity",
        f"${metrics['net_liquidity_trillions']:.2f}T",
        delta=f"{metrics['net_liquidity_velocity_pct']:+.2f}% 30-Day Velocity",
    )
    m4.metric(
        "🛡️ Systemic Stress Score",
        f"{metrics['financial_stress_score']:.1f}/100",
        delta="Loose Conditions (Bullish)",
        delta_color="normal",
    )

    st.markdown("---")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### 📊 Federal Reserve Balance Sheet Breakdown")
        fig_liq = go.Figure(
            data=[
                go.Bar(
                    name="Fed Total Assets",
                    x=["Fed Total Assets"],
                    y=[metrics["fed_assets_trillions"]],
                    marker_color="#38BDF8",
                ),
                go.Bar(
                    name="Treasury General Account (TGA)",
                    x=["TGA Drain"],
                    y=[metrics["tga_balance_trillions"]],
                    marker_color="#EF4444",
                ),
                go.Bar(
                    name="Overnight Reverse Repo (RRP)",
                    x=["RRP Facility"],
                    y=[metrics["reverse_repo_trillions"]],
                    marker_color="#F59E0B",
                ),
                go.Bar(
                    name="Net Market Liquidity",
                    x=["Net Liquidity Available"],
                    y=[metrics["net_liquidity_trillions"]],
                    marker_color="#10B981",
                ),
            ]
        )
        fig_liq.update_layout(
            title="Federal Reserve Net Liquidity Components ($ Trillions)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
            margin=dict(l=20, r=20, t=35, b=20),
            yaxis_title="Trillions USD ($T)",
        )
        st.plotly_chart(fig_liq, use_container_width=True)

    with col_chart2:
        st.markdown("#### 📉 10-Year Benchmark Treasury Yield History")
        df_tnx = get_price_history("^TNX", period="1y", use_cache=True)
        if not df_tnx.empty and "Close" in df_tnx.columns:
            yield_series = df_tnx["Close"] / 10.0
            fig_tnx = go.Figure()
            fig_tnx.add_trace(
                go.Scatter(
                    x=yield_series.index,
                    y=yield_series.values,
                    name="10Y Yield (%)",
                    line=dict(color="#F59E0B", width=2.0),
                )
            )
            fig_tnx.update_layout(
                title="10-Year US Treasury Yield (% Annualized)",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=340,
                margin=dict(l=20, r=20, t=35, b=20),
                yaxis_title="Yield Percentage (%)",
            )
            st.plotly_chart(fig_tnx, use_container_width=True)

    # =========================================================================
    # CROSS-ASSET CREDIT & MACRO SPILLOVER MATRIX (OPTION 2)
    # =========================================================================
    st.markdown("---")
    st.markdown("### ⚡ Cross-Asset Credit & Macro Spillover Matrix")
    st.caption(
        "Institutional Cross-Asset Lead-Lag Engine: High-Yield vs Investment-Grade credit ratio (HYG/LQD), "
        "30-day rolling multi-asset correlation matrix, stealth credit divergences, and CRO risk multipliers."
    )

    try:
        ca_data = _get_cached_cross_asset_matrix()
        regime_badge = ca_data.get("regime_badge", "⚪ NEUTRAL")
        regime_desc = ca_data.get(
            "regime_description", "Balanced cross-asset relationships."
        )
        cro_mult = ca_data.get("cro_risk_multiplier", 1.0)
        zscore = ca_data.get("credit_ratio_20d_zscore", 0.0)
        ratio_val = ca_data.get("credit_ratio_hyg_lqd", 0.75)
        ratio_chg = ca_data.get("credit_ratio_20d_change_pct", 0.0)

        ca_c1, ca_c2, ca_c3, ca_c4 = st.columns(4)
        ca_c1.metric(
            "🌐 Macro Risk Regime", regime_badge.split()[0], delta=regime_badge
        )
        ca_c2.metric(
            "💳 HYG / LQD Ratio", f"{ratio_val:.4f}", delta=f"{ratio_chg:+.2f}% 20D"
        )
        ca_c3.metric(
            "📊 Credit Ratio Z-Score",
            f"{zscore:+.2f}σ",
            delta="Tight Spreads" if zscore > 0 else "Credit Stress",
            delta_color="normal" if zscore > 0 else "inverse",
        )
        ca_c4.metric(
            "🛡️ CRO Leverage Multiplier", f"{cro_mult:.2f}x", delta="Allocation Scaling"
        )

        st.markdown(
            f"""
            <div style="background-color: rgba(14, 165, 233, 0.08); border-left: 4px solid #0EA5E9; padding: 12px 16px; border-radius: 6px; margin: 10px 0 16px 0;">
                <b>Regime Diagnostic:</b> {regime_badge} (CRO Multiplier: <b>{cro_mult:.2f}x</b>)<br/>
                <span style="font-size: 0.9em; opacity: 0.85;">{regime_desc}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Divergence Alerts
        alerts = ca_data.get("divergence_alerts", [])
        if alerts:
            for alt in alerts:
                sev_color = "#EF4444" if alt.get("severity") == "HIGH" else "#F59E0B"
                st.markdown(
                    f"""
                    <div style="background-color: rgba(239, 68, 68, 0.08); border-left: 4px solid {sev_color}; padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;">
                        <b>⚠️ Macro Divergence Detected:</b> {alt.get('type')}<br/>
                        <span style="font-size: 0.85rem;">{alt.get('message')}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # 30-Day Correlation Heatmap
        corr_dict = ca_data.get("correlation_matrix_30d", {})
        if corr_dict:
            corr_df = pd.DataFrame(corr_dict)
            fig_ca_corr = go.Figure(
                data=go.Heatmap(
                    z=corr_df.values,
                    x=corr_df.columns,
                    y=corr_df.index,
                    colorscale="RdBu_r",
                    zmin=-1.0,
                    zmax=1.0,
                    text=np.round(corr_df.values, 2),
                    texttemplate="%{text}",
                    textfont={"size": 11},
                )
            )
            fig_ca_corr.update_layout(
                title="30-Day Rolling Cross-Asset Pearson Correlation Heatmap",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=380,
                margin=dict(l=20, r=20, t=35, b=20),
            )
            st.plotly_chart(fig_ca_corr, use_container_width=True)

    except Exception as ca_err:
        logger.debug(f"Cross-asset matrix render note: {ca_err}")

    # Source & Attribution Notice
    source_label = (
        "🟢 Live FRED® (St. Louis Fed)"
        if metrics.get("is_live_fred")
        else "🟡 Proxy Feeds"
    )
    st.caption(
        f"Data Feed: **{source_label}** | As of: `{metrics.get('as_of_date', 'N/A')}` | "
        f"*{metrics.get('attribution', 'This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.')}*"
    )
