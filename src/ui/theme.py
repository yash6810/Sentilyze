"""
Dynamic Bespoke Theme Engine for Sentilyze.
Supports 3 Institutional Presets:
1. 🌌 Obsidian Terminal (Bloomberg Dark / Citadel Pro)
2. 💎 Cyberpunk Quant (High-Tech Neon Frost & Glassmorphism)
3. 🏛️ Goldman Slate (Executive Minimalist Pro)
"""

import streamlit as st

THEMES = {
    "🌌 Obsidian Terminal": {
        "bg_color": "#0B0F19",
        "card_bg": "rgba(18, 24, 38, 0.75)",
        "border_color": "rgba(16, 185, 129, 0.25)",
        "accent_color": "#10B981",  # Emerald Green
        "accent_secondary": "#059669",
        "text_primary": "#F3F4F6",
        "text_muted": "#9CA3AF",
        "glow": "0 0 15px rgba(16, 185, 129, 0.15)",
        "sidebar_bg": "#070A10",
        "badge_bull": "#10B981",
        "badge_bear": "#EF4444",
    },
    "💎 Cyberpunk Quant": {
        "bg_color": "#070714",
        "card_bg": "rgba(20, 20, 45, 0.7)",
        "border_color": "rgba(99, 102, 241, 0.35)",
        "accent_color": "#06B6D4",  # Cyan
        "accent_secondary": "#6366F1",  # Indigo
        "text_primary": "#FFFFFF",
        "text_muted": "#A5B4FC",
        "glow": "0 0 20px rgba(99, 102, 241, 0.25)",
        "sidebar_bg": "#04040A",
        "badge_bull": "#06B6D4",
        "badge_bear": "#F43F5E",
    },
    "🏛️ Goldman Slate": {
        "bg_color": "#0F172A",
        "card_bg": "rgba(30, 41, 59, 0.8)",
        "border_color": "rgba(56, 189, 248, 0.3)",
        "accent_color": "#38BDF8",  # Sky Blue
        "accent_secondary": "#2563EB",  # Royal Blue
        "text_primary": "#F8FAFC",
        "text_muted": "#94A3B8",
        "glow": "0 0 12px rgba(56, 189, 248, 0.15)",
        "sidebar_bg": "#090D16",
        "badge_bull": "#38BDF8",
        "badge_bear": "#F87171",
    },
}


def inject_custom_theme(theme_name: str = "🌌 Obsidian Terminal"):
    """Injects high-performance, bespoke CSS styling into the Streamlit app."""
    t = THEMES.get(theme_name, THEMES["🌌 Obsidian Terminal"])

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .stApp {{
        background: radial-gradient(circle at 50% 0%, {t['bg_color']} 0%, {t['sidebar_bg']} 100%) !important;
        color: {t['text_primary']} !important;
    }}

    section[data-testid="stSidebar"] {{
        background-color: {t['sidebar_bg']} !important;
        border-right: 1px solid {t['border_color']} !important;
    }}

    /* Institutional Glassmorphism Cards */
    .glass-card {{
        background: {t['card_bg']} !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid {t['border_color']} !important;
        border-radius: 14px !important;
        padding: 20px 24px !important;
        margin-bottom: 18px !important;
        box-shadow: {t['glow']} !important;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    .glass-card:hover {{
        border-color: {t['accent_color']} !important;
        transform: translateY(-2px);
    }}

    /* Metric Cards Custom Styling */
    div[data-testid="stMetric"] {{
        background: {t['card_bg']} !important;
        border: 1px solid {t['border_color']} !important;
        border-radius: 12px !important;
        padding: 16px !important;
        box-shadow: {t['glow']} !important;
    }}
    div[data-testid="stMetricLabel"] p {{
        font-size: 0.82rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: {t['text_muted']} !important;
        font-weight: 600 !important;
    }}
    div[data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        color: {t['text_primary']} !important;
    }}

    /* Buttons */
    .stButton>button {{
        background: linear-gradient(135deg, {t['accent_color']} 0%, {t['accent_secondary']} 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 22px !important;
        font-weight: 700 !important;
        letter-spacing: 0.03em !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
        transition: all 0.25s ease !important;
    }}
    .stButton>button:hover {{
        filter: brightness(1.15) !important;
        transform: scale(1.02) !important;
    }}

    /* Status Badges */
    .badge-bull {{
        background: rgba(16, 185, 129, 0.15) !important;
        color: {t['badge_bull']} !important;
        border: 1px solid {t['badge_bull']} !important;
        padding: 4px 10px !important;
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        display: inline-block;
    }}
    .badge-bear {{
        background: rgba(239, 68, 68, 0.15) !important;
        color: {t['badge_bear']} !important;
        border: 1px solid {t['badge_bear']} !important;
        padding: 4px 10px !important;
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        display: inline-block;
    }}

    /* Tabs Custom Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px !important;
        background: transparent !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: {t['card_bg']} !important;
        border: 1px solid {t['border_color']} !important;
        border-radius: 8px !important;
        color: {t['text_muted']} !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: {t['accent_color']} !important;
        color: #FFFFFF !important;
        border-color: {t['accent_color']} !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
