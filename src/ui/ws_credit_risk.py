"""
Workspace: Merton Structural Credit Risk & Nelson-Siegel-Svensson Yield Curve.
Evaluates corporate solvency, synthetic CDS spreads, equity-credit decoupling,
and 6-parameter continuous Treasury term structure dynamics.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from src.credit_risk import audit_equity_credit_decoupling
from src.macro_yield_curve import evaluate_treasury_yield_curve, svensson_yield
from src.ui.components import render_workspace_header


def render_credit_risk_workspace(selected_ticker: str = "NVDA"):
    render_workspace_header(
        title="🏛️ Merton Structural Credit & NSS Yield Curve",
        subtitle=f"Merton (1974) corporate default radar for {selected_ticker} & Nelson-Siegel-Svensson US Treasury yield decomposition.",
        badge_text="STRUCTURAL CREDIT & TERM STRUCTURE",
        badge_color="#8B5CF6",
    )

    t1, t2 = st.tabs(
        ["🛡️ Corporate Credit Solvency (Merton)", "📈 NSS US Treasury Yield Curve"]
    )

    with t1:
        st.subheader(f"Corporate Solvency & Synthetic CDS Spread: {selected_ticker}")
        st.caption(
            "Solves the Merton (1974) structural option model to extract firm asset value, "
            "asset volatility, distance to default, and decoupling risk."
        )

        credit_res = audit_equity_credit_decoupling(selected_ticker)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "💎 Credit Tier",
            credit_res.get("credit_tier", "INVESTMENT_GRADE").replace("_", " "),
            delta=credit_res.get("credit_health_verdict", "HEALTHY"),
        )
        c2.metric(
            "📏 Distance to Default (DD)",
            f"{credit_res.get('distance_to_default', 0.0):.2f} σ",
            delta=(
                "Fortress"
                if credit_res.get("distance_to_default", 0.0) >= 3.0
                else "Elevated Risk"
            ),
        )
        c3.metric(
            "🏷️ Synthetic CDS Spread",
            f"{credit_res.get('synthetic_cds_spread_bps', 0.0):.1f} bps",
            delta=(
                "Tight (Low Risk)"
                if credit_res.get("synthetic_cds_spread_bps", 0.0) < 100
                else "Widening"
            ),
            delta_color="inverse",
        )
        c4.metric(
            "⚠️ 1-Yr Default Probability",
            f"{credit_res.get('default_probability_1y_pct', 0.0):.3f}%",
        )

        if credit_res.get("is_decoupling_divergence", False):
            st.error(
                "🚨 EQUITY-CREDIT DECOUPLING ALERT: Stock price is rallying while balance sheet credit health deteriorates! Potential bull trap."
            )
        else:
            st.success(
                "✅ Equity and credit signals are harmonious. No structural decoupling detected."
            )

        with st.expander("🔬 Merton Model Solution Details"):
            merton_det = credit_res.get("merton_details", {})
            st.json(merton_det)

    with t2:
        st.subheader("Nelson-Siegel-Svensson (NSS) Continuous Term Structure")
        st.caption(
            "Fits the 6-parameter NSS model to US Treasury benchmarks, yielding continuous spot yield curves and inversion diagnostics."
        )

        curve_res = evaluate_treasury_yield_curve()
        params = curve_res.get("parameters", {})

        y1, y2, y3, y4 = st.columns(4)
        y1.metric(
            "📐 Curve Shape",
            curve_res.get("curve_shape", "NORMAL").replace("_", " "),
        )
        y2.metric(
            "⚖️ 10Y - 2Y Spread",
            f"{curve_res.get('spread_10y_minus_2y', 0.0):+.3f}%",
            delta="Inverted" if curve_res.get("is_curve_inverted", False) else "Normal",
            delta_color=(
                "inverse" if curve_res.get("is_curve_inverted", False) else "normal"
            ),
        )
        y3.metric(
            "⚖️ 10Y - 3M Spread",
            f"{curve_res.get('spread_10y_minus_3m', 0.0):+.3f}%",
        )
        y4.metric(
            "🛡️ CRO Macro Advice",
            curve_res.get("cro_advice", "NEUTRAL").replace("_", " "),
        )

        # Plot continuous fitted yield curve
        b0 = params.get("beta0_level", 4.5)
        b1 = params.get("beta1_slope", -0.5)
        b2 = params.get("beta2_curvature1", -1.0)
        b3 = params.get("beta3_curvature2", 0.5)
        t1_val = params.get("tau1_decay", 1.5)
        t2_val = params.get("tau2_decay", 5.0)

        maturities = np.linspace(0.25, 30.0, 100)
        fitted_yields = svensson_yield(maturities, b0, b1, b2, b3, t1_val, t2_val)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=maturities,
                y=fitted_yields,
                mode="lines",
                name="NSS Fitted Spot Curve",
                line=dict(color="#8B5CF6", width=3),
            )
        )
        fig.update_layout(
            title="Continuous US Treasury Spot Yield Curve y(m)",
            xaxis_title="Maturity (Years)",
            yaxis_title="Yield (%)",
            template="plotly_dark",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"**Fitted Parameters:** $\\beta_0$ (Level) = `{b0:.4f}` | "
            f"$\\beta_1$ (Slope) = `{b1:.4f}` | "
            f"$\\beta_2$ (Curvature 1) = `{b2:.4f}` | "
            f"$\\beta_3$ (Curvature 2) = `{b3:.4f}` | "
            f"Fitting RMSE: `{curve_res.get('fitting_rmse', 0.0):.4f}`"
        )
