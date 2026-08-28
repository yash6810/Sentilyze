"""
Workspace 7: SHAP Explainability & Transparent Decision Trees.
"""

import os
import streamlit as st
from PIL import Image
from src.ui.components import render_workspace_header


def render_xai_workspace(selected_ticker: str):
    """Renders SHAP explainability plots and decision tree visuals."""
    render_workspace_header(
        title=f"🧠 Explainable AI & SHAP Reasoning ({selected_ticker})",
        subtitle="Game-Theoretic SHAP Value Allocations & Transparent Feature Importance Drivers",
        badge_text="TRANSPARENT XAI",
        badge_color="#8B5CF6",
    )

    summary_png = os.path.join("results", f"{selected_ticker}_shap_summary.png")
    waterfall_png = os.path.join("results", f"{selected_ticker}_shap_waterfall.png")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📊 Global Feature Importance Summary")
        if os.path.exists(summary_png):
            st.image(summary_png, use_container_width=True)
        else:
            st.info(f"No global SHAP summary plot found for {selected_ticker}.")

    with c2:
        st.markdown("#### 🌊 Local Decision Waterfall")
        if os.path.exists(waterfall_png):
            st.image(waterfall_png, use_container_width=True)
        else:
            st.info(f"No local waterfall SHAP plot found for {selected_ticker}.")
