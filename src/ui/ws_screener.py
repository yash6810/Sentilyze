"""
Workspace 24: Real-Time Market Anomaly Screener.
"""

import streamlit as st
from src.ui.components import render_workspace_header
from components.live_screener import render_live_screener_section


def render_screener_workspace():
    """Renders the Real-Time Market Anomaly Screener Workspace."""
    render_workspace_header(
        title="🌐 Real-Time Market Anomaly Screener",
        subtitle="Sub-second multi-condition scanning for Relative Volume (RVOL) surges, Day Range Breakouts, and Pullback Bounces.",
        badge_text="LIVE MARKET RADAR",
        badge_color="#00D4AA",
    )

    render_live_screener_section()
