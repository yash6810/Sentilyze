"""
Shared Institutional UI Components & Widgets for Sentilyze.
"""

import streamlit as st
from typing import Optional, List, Dict, Any


def render_workspace_header(
    title: str,
    subtitle: str,
    badge_text: str = "LIVE PRODUCTION",
    badge_color: str = "#10B981",
):
    """Renders an executive header banner with live status badge."""
    html = f"""
    <div class="glass-card" style="margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div>
                <h1 style="margin: 0; font-size: 1.85rem; font-weight: 800; letter-spacing: -0.02em;">{title}</h1>
                <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.95rem;">{subtitle}</p>
            </div>
            <div>
                <span style="background: rgba(16, 185, 129, 0.15); color: {badge_color}; border: 1px solid {badge_color}; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 0.8rem; letter-spacing: 0.05em; font-family: 'JetBrains Mono', monospace;">
                    ● {badge_text}
                </span>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_glass_card(content_html: str):
    """Wraps HTML content inside an institutional frosted glass container."""
    st.markdown(f'<div class="glass-card">{content_html}</div>', unsafe_allow_html=True)


def render_conviction_gauge(conviction_pct: float, label: str = "AI CONVICTION SCORE"):
    """Renders a progress meter with dynamic color coding."""
    if conviction_pct >= 70.0:
        color = "#10B981"  # Emerald
        tag = "STRONG BUY"
    elif conviction_pct >= 55.0:
        color = "#3B82F6"  # Blue
        tag = "MODERATE BUY"
    elif conviction_pct >= 45.0:
        color = "#64748B"  # Neutral
        tag = "NEUTRAL / HOLD"
    else:
        color = "#EF4444"  # Red
        tag = "STRONG SELL"

    html = f"""
    <div class="glass-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 0.8rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; font-family: 'JetBrains Mono', monospace;">{label}</span>
            <span style="color: {color}; font-weight: 800; font-family: 'JetBrains Mono', monospace;">{conviction_pct:.1f}% ({tag})</span>
        </div>
        <div style="background: rgba(255,255,255,0.08); height: 10px; border-radius: 5px; overflow: hidden;">
            <div style="background: {color}; width: {conviction_pct}%; height: 100%; border-radius: 5px; box-shadow: 0 0 10px {color}; transition: width 0.5s ease;"></div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
