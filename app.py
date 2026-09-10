"""
Sentilyze - Institutional Algorithmic Trading & MLOps Platform.
High-Speed 5-Cockpit Mission Control Station.
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# --- 1. Streamlit Page Configuration (Must Be First) ---
st.set_page_config(
    layout="wide",
    page_title="Sentilyze | Institutional Quant OS",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# --- 2. Import Modular Theme Engine & Components ---
from src.ui.theme import inject_custom_theme
from src.ui.components import get_market_status
from src.config import COMPANY_NAMES

STOCKS_FILE = "stocks.txt"


@st.cache_data(ttl=3600)
def load_universe_tickers():
    """Loads active S&P 100 universe tickers."""
    if os.path.exists(STOCKS_FILE):
        with open(STOCKS_FILE, "r", encoding="utf-8") as f:
            tickers = [
                line.strip().upper()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
        if tickers:
            return tickers
    return ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA"]


@st.cache_data(ttl=300)
def get_model_universe_indicator():
    """Computes universe model training coverage and mean accuracy using compiled summary."""
    import glob
    import json

    summary_file = os.path.join("results", "universe_summary.json")
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            accs = [row["accuracy"] for row in data if "accuracy" in row]
            mean_acc = (sum(accs) / len(accs)) if accs else 0.0
            model_count = len(data)
            total_valid = 535
            pct = min(100.0, (model_count / total_valid) * 100.0)
            return {
                "models_count": model_count,
                "total_valid": total_valid,
                "pct_trained": pct,
                "mean_accuracy": mean_acc,
                "metrics_count": len(accs),
            }
        except Exception:
            pass

    model_files = glob.glob("models/*_model.json")
    model_count = len(model_files)
    metrics_files = glob.glob("results/*_metrics.json")
    accuracies = []
    for f in metrics_files:
        try:
            with open(f, "r", encoding="utf-8") as jf:
                d = json.load(jf)
                a = d.get("accuracy")
                if a is not None and isinstance(a, (int, float)):
                    accuracies.append(float(a))
        except Exception:
            pass
    mean_acc = (sum(accuracies) / len(accuracies) * 100.0) if accuracies else 0.0
    total_valid = 535
    pct = min(100.0, (model_count / total_valid) * 100.0) if total_valid else 0.0
    return {
        "models_count": model_count,
        "total_valid": total_valid,
        "pct_trained": pct,
        "mean_accuracy": mean_acc,
        "metrics_count": len(accuracies),
    }


def main():
    # Ensure permanent 24/7 autonomous trading daemon is continuously running
    try:
        from src.autonomous_trader import (
            ensure_background_daemon_thread_running,
        )

        ensure_background_daemon_thread_running(interval_seconds=60)
    except Exception:
        pass

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

    # Model Universe Training Indicator Widget
    u_stats = get_model_universe_indicator()
    st.sidebar.markdown(
        f"""
        <div style="background: rgba(16, 185, 129, 0.06); border: 1px solid rgba(16, 185, 129, 0.22); border-radius: 8px; padding: 10px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.75rem; font-weight: 700; color: #10B981; text-transform: uppercase;">🧠 TRAINED UNIVERSE</span>
                <span style="font-size: 0.72rem; font-weight: 800; color: #10B981; font-family: 'JetBrains Mono', monospace;">{u_stats['pct_trained']:.1f}% READY</span>
            </div>
            <div style="font-size: 1.05rem; font-weight: 800; color: #F8FAFC; margin-top: 4px; font-family: 'JetBrains Mono', monospace;">
                {u_stats['models_count']} / {u_stats['total_valid']} Assets Trained
            </div>
            <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 2px;">
                Mean Accuracy: <b style="color: #38BDF8;">{u_stats['mean_accuracy']:.2f}%</b> (5-Fold Purged CV)
            </div>
            <div style="width: 100%; background: rgba(255,255,255,0.1); border-radius: 4px; height: 5px; margin-top: 6px;">
                <div style="width: {u_stats['pct_trained']}%; background: #10B981; height: 5px; border-radius: 4px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Institutional Theme Engine (🏛️ Goldman Slate)
    inject_custom_theme("🏛️ Goldman Slate")

    # 2. Universe Ticker Selector
    tickers = load_universe_tickers()
    selected_ticker = st.sidebar.selectbox(
        "🎯 Select Asset Ticker",
        tickers,
        index=0 if "NVDA" in tickers else 0,
        format_func=lambda t: f"{t} — {COMPANY_NAMES.get(t, t)}",
    )

    st.sidebar.markdown("---")

    # 3. Streamlined 5 Master Cockpits
    cockpits = {
        "🎯 1. Live Trading & Alpha Cockpit": "trading",
        "🌊 2. Smart Money & Alternative Data": "smart_money",
        "💼 3. Portfolio & Risk Engine": "portfolio",
        "📈 4. Backtest & Performance Factsheet": "backtest",
        "🧠 5. Deep Quant & Explainability (XAI)": "deep_quant",
    }

    selected_cockpit_label = st.sidebar.radio(
        "📂 Mission Control Cockpits",
        list(cockpits.keys()),
        index=0,
    )
    cockpit_key = cockpits[selected_cockpit_label]

    # Quick Status in Sidebar
    comp_name = COMPANY_NAMES.get(selected_ticker, selected_ticker)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div style="font-size: 0.75rem; color: #64748B; font-family: 'JetBrains Mono', monospace;">
            ● Model Universe: {len(tickers)} S&P Assets<br>
            ● Active Asset: <b>{selected_ticker}</b><br>
            ● Company: {comp_name}<br>
            ● Status: 🟢 5 Cockpits Operational
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Dispatch Cockpit Router (Lazy Loaded for Instant Startup) ---
    if cockpit_key == "trading":
        from src.ui.cockpit_trading import render_trading_cockpit

        render_trading_cockpit(selected_ticker)
    elif cockpit_key == "smart_money":
        from src.ui.cockpit_smart_money import render_smart_money_cockpit

        render_smart_money_cockpit(selected_ticker)
    elif cockpit_key == "portfolio":
        from src.ui.cockpit_portfolio import render_portfolio_cockpit

        render_portfolio_cockpit(selected_ticker)
    elif cockpit_key == "backtest":
        from src.ui.cockpit_backtest import render_backtest_cockpit

        render_backtest_cockpit(selected_ticker)
    elif cockpit_key == "deep_quant":
        from src.ui.cockpit_deep_quant import render_deep_quant_cockpit

        render_deep_quant_cockpit(selected_ticker)


if __name__ == "__main__":
    main()
