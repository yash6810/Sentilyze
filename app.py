"""
Sentilyze - Institutional Algorithmic Trading & MLOps Platform.
Modular Master Entry Point & High-Performance Routing Station.
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# --- 1. Streamlit Page Configuration (Must Be First) ---
st.set_page_config(
    layout="wide",
    page_title="Sentilyze | Institutional AI Trading Platform",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# --- 2. Import Modular Theme Engine & Workspaces ---
from src.ui.theme import inject_custom_theme, THEMES
from src.ui.ws_live_prediction import render_live_prediction_workspace
from src.ui.ws_committee import render_committee_workspace
from src.ui.ws_autonomous_trader import render_autonomous_trader_workspace
from src.ui.ws_alternative_data import render_alternative_data_workspace
from src.ui.ws_portfolio import render_portfolio_workspace
from src.ui.ws_backtesting import render_backtesting_workspace
from src.ui.ws_xai_shap import render_xai_workspace
from src.ui.ws_options_surface import render_options_surface_workspace
from src.ui.ws_deep_quant import render_deep_quant_workspace
from src.ui.ws_quantum_tournament import render_quantum_tournament_workspace
from src.ui.ws_portfolio_diversity import render_portfolio_diversity_workspace
from src.ui.ws_insider_radar import render_insider_radar_workspace
from src.ui.ws_performance_factsheet import render_performance_factsheet_workspace
from src.ui.ws_drl_agent import render_drl_agent_workspace
from src.ui.ws_strategy_incubator import render_strategy_incubator_workspace
from src.ui.components import get_market_status
from src.config import COMPANY_NAMES

STOCKS_FILE = "stocks.txt"


def load_universe_tickers():
    """Loads active S&P 100 universe tickers."""
    if os.path.exists(STOCKS_FILE):
        with open(STOCKS_FILE, "r") as f:
            tickers = [
                line.strip().upper()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
        if tickers:
            return tickers
    return ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA"]


def main():
    # --- Sidebar Controls ---
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding-bottom: 8px;">
            <h2 style="margin: 0; font-weight: 800; letter-spacing: -0.02em;">📈 SENTILYZE</h2>
            <p style="margin: 2px 0 0 0; color: #94A3B8; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace;">INSTITUTIONAL QUANT OS</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Live Market Clock & Status Widget
    mkt = get_market_status()
    st.sidebar.markdown(
        f"""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.75rem; font-weight: 700; color: #94A3B8; text-transform: uppercase;">NYSE / NASDAQ</span>
                <span style="font-size: 0.72rem; font-weight: 800; color: {mkt['badge_color']}; font-family: 'JetBrains Mono', monospace;">{mkt['icon']} {mkt['status']}</span>
            </div>
            <div style="font-size: 0.85rem; font-weight: 600; color: #F8FAFC; margin-top: 4px; font-family: 'JetBrains Mono', monospace;">
                🕒 {mkt['time_str']}
            </div>
            <div style="font-size: 0.7rem; color: #64748B; margin-top: 2px;">
                {mkt['description']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Bespoke Theme Selector
    theme_choice = st.sidebar.selectbox(
        "🎨 UI Theme Preset",
        list(THEMES.keys()),
        index=0,
        help="Select a bespoke institutional visual theme.",
    )
    inject_custom_theme(theme_choice)

    st.sidebar.markdown("---")

    # 2. Universe Ticker Selector
    tickers = load_universe_tickers()
    selected_ticker = st.sidebar.selectbox(
        "🎯 Select Asset Ticker",
        tickers,
        index=0 if "NVDA" in tickers else 0,
        format_func=lambda t: f"{t} — {COMPANY_NAMES.get(t, t)}",
    )

    st.sidebar.markdown("---")

    # 3. Workspace Navigation
    workspaces = {
        "🔮 1. Live Directional Prediction": "prediction",
        "🏛️ 2. 4-Agent Trading Committee": "committee",
        "🤖 3. 24/7 Autonomous Live Trader": "auto_trader",
        "📡 4. 4-Station Reddit News & Pre-IPO": "alternative",
        "💼 5. Portfolio Kelly Sizing": "portfolio",
        "📈 6. Walk-Forward Backtesting": "backtesting",
        "🧠 7. SHAP Explainability & Trees": "xai",
        "📉 8. 3D Volatility & Dark Pools": "options",
        "🔗 9. Statistical Arbitrage Pairs": "statarb",
        "🕸️ 10. GNN Supply Chain Contagion": "gnn",
        "🌪️ 11. Black Swan Crisis Simulator": "stress",
        "🕵️ 12. Forensic Beneish M-Score": "forensic",
        "🏛️ 13. DCF Intrinsic Valuation": "dcf",
        "👑 14. 25-Paper Tournament & Deep Learning Shield": "quantum_tournament",
        "🧬 15. Portfolio Diversity & Correlation Grader": "portfolio_diversity",
        "🏛️ 16. Smart-Money Executive & Insider Radar": "insider_radar",
        "📊 17. Institutional Risk & Alpha Factsheet": "performance_factsheet",
        "🤖 18. Deep RL Autonomous Policy Agent": "drl_agent",
        "🔬 19. Evolutionary Strategy Incubator": "strategy_incubator",
    }

    selected_ws_label = st.sidebar.radio(
        "📂 Mission Control Workspaces",
        list(workspaces.keys()),
        index=0,
    )
    ws_key = workspaces[selected_ws_label]

    # Quick Status in Sidebar
    comp_name = COMPANY_NAMES.get(selected_ticker, selected_ticker)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div style="font-size: 0.75rem; color: #64748B; font-family: 'JetBrains Mono', monospace;">
            ● Model Universe: {len(tickers)} S&P Assets<br>
            ● Active Asset: <b>{selected_ticker}</b><br>
            ● Company: {comp_name}<br>
            ● Status: 🟢 All Systems Operational
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Dispatch Workspace Router ---
    if ws_key == "prediction":
        render_live_prediction_workspace(selected_ticker)
    elif ws_key == "committee":
        render_committee_workspace(selected_ticker)
    elif ws_key == "auto_trader":
        render_autonomous_trader_workspace(selected_ticker)
    elif ws_key == "alternative":
        render_alternative_data_workspace(selected_ticker)
    elif ws_key == "portfolio":
        render_portfolio_workspace(selected_ticker)
    elif ws_key == "backtesting":
        render_backtesting_workspace(selected_ticker)
    elif ws_key == "xai":
        render_xai_workspace(selected_ticker)
    elif ws_key == "options":
        render_options_surface_workspace(selected_ticker)
    elif ws_key in ["statarb", "gnn", "stress", "forensic", "dcf"]:
        render_deep_quant_workspace(selected_ticker, mode=ws_key)
    elif ws_key == "quantum_tournament":
        render_quantum_tournament_workspace(selected_ticker)
    elif ws_key == "portfolio_diversity":
        render_portfolio_diversity_workspace()
    elif ws_key == "insider_radar":
        render_insider_radar_workspace(selected_ticker)
    elif ws_key == "performance_factsheet":
        render_performance_factsheet_workspace()
    elif ws_key == "drl_agent":
        render_drl_agent_workspace(selected_ticker)
    elif ws_key == "strategy_incubator":
        render_strategy_incubator_workspace(selected_ticker)


if __name__ == "__main__":
    main()
