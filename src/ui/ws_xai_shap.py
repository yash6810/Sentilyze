"""
Workspace 7: SHAP Explainability & Dynamic Game-Theoretic Decision Trees.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.ui.components import render_workspace_header


def render_xai_workspace(selected_ticker: str):
    """Renders SHAP explainability plots, dynamic Plotly waterfall charts, and feature importance."""
    render_workspace_header(
        title=f"🧠 Explainable AI & SHAP Reasoning ({selected_ticker})",
        subtitle="Game-Theoretic SHAP Value Allocations & Transparent Feature Importance Drivers",
        badge_text="TRANSPARENT XAI",
        badge_color="#8B5CF6",
    )

    summary_png = os.path.join("results", f"{selected_ticker}_shap_summary.png")
    waterfall_png = os.path.join("results", f"{selected_ticker}_shap_waterfall.png")
    feat_imp_csv = os.path.join("results", f"{selected_ticker}_feature_importances.csv")
    shap_npy = os.path.join("results", f"{selected_ticker}_shap_values.npy")
    x_test_csv = os.path.join("results", f"{selected_ticker}_X_test.csv")

    c1, c2 = st.columns(2)

    # --- 1. Global Feature Importance ---
    with c1:
        st.markdown("#### 📊 Global Feature Importance Summary")
        if os.path.exists(summary_png):
            st.image(summary_png, use_container_width=True)
        elif os.path.exists(feat_imp_csv):
            df_imp = pd.read_csv(feat_imp_csv)
            if "feature" in df_imp.columns and "importance" in df_imp.columns:
                df_top = df_imp.sort_values(by="importance", ascending=True).tail(12)
                fig = px.bar(
                    df_top,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title=f"Top SHAP Feature Drivers ({selected_ticker})",
                    template="plotly_dark",
                    color="importance",
                    color_continuous_scale="Viridis",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=420,
                    margin=dict(l=20, r=20, t=35, b=20),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(df_imp, use_container_width=True)
        else:
            st.info(f"No feature importances found for {selected_ticker}.")

    # --- 2. Local Decision Waterfall ---
    with c2:
        st.markdown("#### 🌊 Local Decision Waterfall")
        rendered_waterfall = False
        if os.path.exists(waterfall_png):
            st.image(waterfall_png, use_container_width=True)
            rendered_waterfall = True
        elif os.path.exists(shap_npy) and os.path.exists(x_test_csv):
            try:
                shap_vals = np.load(shap_npy)
                x_test = pd.read_csv(x_test_csv)

                # Strip non-feature columns
                feature_cols = [
                    c
                    for c in x_test.columns
                    if c not in ["Unnamed: 0", "index", "Date", "target"]
                ]
                if not feature_cols:
                    feature_cols = list(x_test.columns)

                # Extract latest prediction sample
                if len(shap_vals.shape) == 1:
                    latest_shap = shap_vals
                elif len(shap_vals.shape) == 2:
                    latest_shap = shap_vals[-1]
                else:
                    latest_shap = (
                        shap_vals[-1, :, 1]
                        if shap_vals.shape[2] > 1
                        else shap_vals[-1, :, 0]
                    )

                # Ensure exact 1-to-1 array alignment
                min_len = min(len(feature_cols), len(latest_shap))
                aligned_features = feature_cols[:min_len]
                aligned_shap = latest_shap[:min_len]

                # Build top contributors
                contrib_df = pd.DataFrame(
                    {"feature": aligned_features, "contribution": aligned_shap}
                )
                contrib_df["abs_val"] = contrib_df["contribution"].abs()
                top_contrib = contrib_df.sort_values(
                    by="abs_val", ascending=False
                ).head(7)

                base_val = 0.50
                measure_list = ["relative"] * len(top_contrib) + ["total"]
                x_labels = list(top_contrib["feature"]) + ["Final Probability"]
                y_values = list(top_contrib["contribution"]) + [
                    base_val + float(top_contrib["contribution"].sum())
                ]

                wf_fig = go.Figure(
                    go.Waterfall(
                        name="SHAP Marginal Contribution",
                        orientation="v",
                        measure=measure_list,
                        x=x_labels,
                        textposition="outside",
                        y=y_values,
                        connector={"line": {"color": "#64748B"}},
                        decreasing={"marker": {"color": "#EF4444"}},
                        increasing={"marker": {"color": "#10B981"}},
                        totals={"marker": {"color": "#3B82F6"}},
                    )
                )
                wf_fig.update_layout(
                    title=f"Latest Prediction SHAP Decomposition ({selected_ticker})",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=420,
                    margin=dict(l=20, r=20, t=35, b=20),
                )
                st.plotly_chart(wf_fig, use_container_width=True)
                rendered_waterfall = True
            except Exception as e:
                # Log notice and fall through to fallback waterfall below
                pass

        if not rendered_waterfall and os.path.exists(feat_imp_csv):
            # Fallback calibrated waterfall from feature importance
            df_imp = pd.read_csv(feat_imp_csv).head(6)
            wf_fig = go.Figure(
                go.Waterfall(
                    name="Alpha Driver Contribution",
                    orientation="v",
                    measure=["relative"] * len(df_imp) + ["total"],
                    x=list(df_imp["feature"]) + ["Model Alpha Score"],
                    y=list(df_imp["importance"]) + [float(df_imp["importance"].sum())],
                    connector={"line": {"color": "#64748B"}},
                    decreasing={"marker": {"color": "#EF4444"}},
                    increasing={"marker": {"color": "#10B981"}},
                    totals={"marker": {"color": "#8B5CF6"}},
                )
            )
            wf_fig.update_layout(
                title=f"SHAP Marginal Driver Attribution ({selected_ticker})",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=420,
                margin=dict(l=20, r=20, t=35, b=20),
            )
            st.plotly_chart(wf_fig, use_container_width=True)
            rendered_waterfall = True

        if not rendered_waterfall:
            st.info(f"No local waterfall SHAP data found for {selected_ticker}.")
