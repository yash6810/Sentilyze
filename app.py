import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# --- Streamlit Page Config MUST BE FIRST ---
st.set_page_config(
    layout="wide",
    page_title="Sentilyze | Institutional AI Trading Platform",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

from src.utils import get_logger
from src.config import FEATURES
from src.preprocessing import preprocess_data
from src.modeling import load_model, get_prediction_on_latest_data
from src.paper_broker import PaperBroker
from src.realtime_tracker import fetch_live_quote, fetch_universe_live_quotes, evaluate_intraday_execution
from src.portfolio import build_unified_portfolio, load_all_ticker_portfolios, calculate_risk_parity_weights
from src.alerts import format_signal_card, send_discord_alert, send_telegram_alert
from src.rebalancer import calculate_custom_rebalance
from src.tearsheet import generate_executive_pdf_tearsheet
from src.stress_tester import run_monte_carlo_var
from src.correlation_matrix import compute_correlation_matrix
from src.strategy_optimizer import simulate_strategy_sandbox

logger = get_logger(__name__)


# --- Helper: Dynamic Ticker Universe ---
def get_universe_tickers() -> List[str]:
    stocks_file = "stocks.txt"
    if os.path.exists(stocks_file):
        with open(stocks_file, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]
            if tickers:
                return tickers
    return [
        "NVDA", "AAPL", "MSFT", "GOOGL", "META", "TSLA", "AMZN",
        "AVGO", "AMD", "PLTR", "LLY", "QQQ", "SPY", "JPM", "COST", "NFLX", "TSM"
    ]


UNIVERSE_TICKERS = get_universe_tickers()


# --- Premium Custom Styling ---
def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
        }
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        .hero-banner {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
            border: 1px solid rgba(0, 212, 170, 0.2);
            border-radius: 14px;
            padding: 1.2rem 1.8rem;
            margin-bottom: 1.2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .metric-card {
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
        }
        .section-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #F8FAFC;
            margin: 1.2rem 0 0.8rem 0;
            border-left: 4px solid #00D4AA;
            padding-left: 0.6rem;
        }
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 🎯 WORKSPACE 1: AI TRADING COMMAND CENTER (LIVE SIGNAL & REAL-TIME RADAR)
# ==============================================================================
def render_command_center(ticker: str):
    st.markdown('<div class="section-title">⚡ Live AI Momentum Signal & Market Radar</div>', unsafe_allow_html=True)

    # 1. Real-Time Price & AI Prediction
    col1, col2 = st.columns([1.2, 1.8])

    with col1:
        quote = fetch_live_quote(ticker)
        curr_p = float(quote.get("price", 0))
        chg = float(quote.get("change_pct", 0))

        st.markdown(
            f"""
            <div class="metric-card" style="text-align: left; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-size: 1.6rem; font-weight: 800; color: #00D4AA;">{ticker}</span>
                    <span style="font-size: 0.9rem; font-weight: bold; background: {'#065F46' if chg>=0 else '#991B1B'}; padding: 3px 8px; border-radius: 4px;">{chg:+.2f}%</span>
                </div>
                <div style="font-size: 2rem; font-weight: 900; margin: 0.4rem 0;">${curr_p:,.2f}</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">Day High: ${quote.get('day_high', 0):,.2f} | Day Low: ${quote.get('day_low', 0):,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # AI Prediction
        model_path = f"models/{ticker}_model.json"
        if os.path.exists(model_path) or os.path.exists(model_path.replace(".json", ".joblib")):
            with st.spinner(f"Computing XGBoost + FinBERT inference for {ticker}..."):
                try:
                    df = preprocess_data(ticker, use_cache=True)
                    model = load_model(model_path)
                    pred, conf = get_prediction_on_latest_data(model, df)

                    signal = "BUY" if pred == 1 and conf >= 0.50 else "HOLD"
                    sig_color = "#10B981" if signal == "BUY" else "#F59E0B"

                    atr = float(df["atr"].iloc[-1]) if "atr" in df.columns else curr_p * 0.03
                    tp1 = curr_p + (2.5 * atr)
                    tp2 = curr_p + (4.5 * atr)
                    sl = curr_p - (1.5 * atr)

                    st.markdown(
                        f"""
                        <div class="metric-card" style="text-align: left; border: 1px solid {sig_color}55;">
                            <div style="font-size: 0.85rem; color: #94A3B8;">AI MODEL VERDICT</div>
                            <div style="font-size: 1.8rem; font-weight: 900; color: {sig_color};">{signal} ({conf*100:.1f}% Confidence)</div>
                            <div style="font-size: 0.85rem; margin-top: 0.5rem;">
                                • <b>TP1 (+2.5 ATR):</b> ${tp1:,.2f} <br>
                                • <b>TP2 (+4.5 ATR):</b> ${tp2:,.2f} <br>
                                • <b>Stop-Loss:</b> ${sl:,.2f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Inference error: {e}")
        else:
            st.warning(f"No trained model found for {ticker}.")

    with col2:
        # Intraday Price & 5-Minute Proximity Radar
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-weight: 700; color: #E2E8F0;">📡 5-Minute Proximity Radar (Open Positions)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        broker = PaperBroker()
        open_pos = broker.state.get("open_positions", {})

        if open_pos:
            for t_sym, pos in open_pos.items():
                q_sym = fetch_live_quote(t_sym)
                live_p = float(q_sym.get("price", pos.get("current_price", 100)))
                entry_p = float(pos.get("entry_price", live_p))
                tp1_val = float(pos.get("tp1_target", entry_p * 1.06))
                sl_val = float(pos.get("sl_target", entry_p * 0.95))
                pnl_ret = ((live_p - entry_p) / entry_p) * 100.0 if entry_p > 0 else 0.0

                span = max(0.01, tp1_val - entry_p)
                prog = max(0.0, min(1.0, (live_p - entry_p) / span))

                st.markdown(
                    f"""
                    <div style="background: #0F172A; border: 1px solid #334155; border-radius: 8px; padding: 0.8rem; margin-bottom: 0.6rem;">
                        <div style="display: flex; justify-content: space-between;">
                            <b>{t_sym}</b>
                            <span style="color: {'#10B981' if pnl_ret>=0 else '#EF4444'}; font-weight: bold;">${live_p:,.2f} ({pnl_ret:+.2f}%)</span>
                        </div>
                        <div style="font-size: 0.75rem; color: #94A3B8;">Target TP1: ${tp1_val:,.2f} | Stop-Loss: ${sl_val:,.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(prog, text=f"Progress to Take-Profit: {prog*100:.1f}%")
        else:
            st.info("No active open positions. Cash is liquid awaiting morning signals.")

        if st.button("⚡ Trigger 5-Minute Intraday Scan & Exit Check", use_container_width=True):
            with st.spinner("Checking live quotes against TP/SL triggers..."):
                res = evaluate_intraday_execution(broker=broker)
                trades = res.get("executed_trades", [])
                if trades:
                    st.success(f"Executed {len(trades)} exit trades on live market prices!")
                else:
                    st.info("All open positions are within target bands. No exit thresholds triggered.")
                st.rerun()

    # 2. Universe Live Board
    st.markdown('<div class="section-title">📊 17-Stock Universe Real-Time Quotes</div>', unsafe_allow_html=True)
    with st.spinner("Streaming universe quotes..."):
        all_q = fetch_universe_live_quotes(UNIVERSE_TICKERS)
        rows = []
        for t, q in all_q.items():
            price = q.get("price", 0)
            rows.append(
                {
                    "Ticker": t,
                    "Live Price": f"${price:,.2f}" if price > 0 else "N/A",
                    "Today's Return": f"{q.get('change_pct', 0):+.2f}%",
                    "Day High": f"${q.get('day_high', 0):,.2f}",
                    "Day Low": f"${q.get('day_low', 0):,.2f}",
                    "Status": "🟢 Live" if q.get("status") == "LIVE" else "⚪ Offline",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ==============================================================================
# 💼 WORKSPACE 2: PORTFOLIO & PAPER TRADING BROKER ($100k ACCOUNT)
# ==============================================================================
def render_portfolio_workspace():
    st.markdown('<div class="section-title">💼 Virtual Paper Trading Broker ($100,000 Portfolio)</div>', unsafe_allow_html=True)

    broker = PaperBroker()
    summary = broker.get_portfolio_summary()

    # Top KPI Bar
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Equity", f"${summary['total_equity']:,.2f}", f"{summary['total_return_pct']:+.2f}%")
    with kpi2:
        st.metric("Available Cash", f"${summary['cash']:,.2f}")
    with kpi3:
        st.metric("Unrealized PnL", f"${summary['unrealized_pnl']:+,.2f}")
    with kpi4:
        st.metric("Win Rate", f"{summary['win_rate']:.1f}% ({summary['winning_trades']}/{summary['total_trades']})")

    # PDF Download Button
    col_pdf, col_act = st.columns([1, 2])
    with col_pdf:
        pdf_bytes = generate_executive_pdf_tearsheet(
            portfolio_summary=summary,
            open_positions=list(broker.state.get("open_positions", {}).values()),
            equity_history_df=broker.get_equity_curve_df(),
        )
        st.download_button(
            "📄 Download 2-Page PDF Tearsheet",
            data=pdf_bytes,
            file_name=f"Sentilyze_Factsheet_{datetime.now(timezone.utc).strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    # Active Holdings Table
    st.markdown('<div class="section-title">📦 Active Open Holdings (50/50 Scale-Out Model)</div>', unsafe_allow_html=True)
    open_df = broker.get_open_positions_df()
    if not open_df.empty:
        st.dataframe(open_df, use_container_width=True, hide_index=True)
    else:
        st.info("No open positions. Ready to deploy cash into morning high-conviction signals.")

    # Equity Curve & Closed Trades
    col_eq, col_jrnl = st.columns([1.5, 1.5])
    with col_eq:
        st.markdown('<div class="section-title">📈 Equity Growth Curve</div>', unsafe_allow_html=True)
        eq_df = broker.get_equity_curve_df()
        if not eq_df.empty and "total_equity" in eq_df.columns:
            st.line_chart(eq_df["total_equity"], use_container_width=True)
        else:
            st.info("Equity curve will update after daily scans.")

    with col_jrnl:
        st.markdown('<div class="section-title">📜 Closed Trade History Journal</div>', unsafe_allow_html=True)
        closed_df = broker.get_closed_trades_df()
        if not closed_df.empty:
            st.dataframe(closed_df, use_container_width=True, hide_index=True)
        else:
            st.info("No closed trades yet. Trades appear here once Take-Profit or Stop-Loss is reached.")


# ==============================================================================
# 📊 WORKSPACE 3: MULTI-ASSET FUND & RISK ANALYTICS
# ==============================================================================
def render_fund_and_risk():
    st.markdown('<div class="section-title">📊 17-Asset Fund Allocation, Rebalancer & Stress Testing</div>', unsafe_allow_html=True)

    tab_fund, tab_var, tab_corr = st.tabs(
        ["💼 Fund Allocation & Rebalancer", "🎲 Monte Carlo Stress Test & VaR", "🔗 17-Asset Correlation Matrix"]
    )

    with tab_fund:
        col_reb1, col_reb2 = st.columns([1, 2])
        with col_reb1:
            st.markdown("### 🧮 Custom Capital Share Calculator")
            budget = st.number_input("Total Investment Budget ($)", min_value=1000.0, max_value=1000000.0, value=25000.0, step=1000.0)
            model_type = st.selectbox("Allocation Model", ["Risk Parity (Inverse Vol)", "Equal Weight", "Conviction Weight"])
            model_key = "risk_parity" if "Risk Parity" in model_type else ("equal_weight" if "Equal Weight" in model_type else "conviction")
            reb_res = calculate_custom_rebalance(total_capital=budget, method=model_key)

        with col_reb2:
            st.markdown(f"### 📋 Exact Whole-Share Buy Orders (${budget:,.2f})")
            if "allocation_table" in reb_res:
                st.dataframe(pd.DataFrame(reb_res["allocation_table"]), use_container_width=True, hide_index=True)

    with tab_var:
        st.markdown("### 🎲 Monte Carlo Forward Simulation & VaR")
        st.caption("Simulates 1,000 future forward market paths to compute Value-at-Risk (VaR) and Expected Shortfall.")
        if st.button("🚀 Run Monte Carlo Simulation", use_container_width=True):
            with st.spinner("Simulating 1,000 Geometric Brownian Motion paths..."):
                sim_res = run_monte_carlo_var(initial_equity=100000.0, num_paths=1000, days=45)
                st.success(f"95% Value-at-Risk: ${sim_res['var_95_dollar']:,.2f} ({sim_res['var_95_pct']:.2f}%) | Probability of Profit: {sim_res['prob_profit_pct']:.1f}%")

    with tab_corr:
        st.markdown("### 🔗 17-Asset Cross-Correlation Heatmap")
        corr_res = compute_correlation_matrix()
        if "matrix" in corr_res:
            st.dataframe(corr_res["matrix"], use_container_width=True)


# ==============================================================================
# 🔬 WORKSPACE 4: QUANTITATIVE RESEARCH & STRATEGY SANDBOX
# ==============================================================================
def render_research_workspace(ticker: str):
    st.markdown('<div class="section-title">🔬 Quantitative Sandbox & Strategy Optimizer</div>', unsafe_allow_html=True)

    col_ctrl, col_chart = st.columns([1, 2])
    with col_ctrl:
        st.markdown("### ⚙️ Strategy Sandbox Controls")
        lev = st.slider("Account Leverage", min_value=1.0, max_value=2.0, value=1.0, step=0.1)
        conf_thresh = st.slider("Confidence Filter (%)", min_value=50, max_value=75, value=55, step=5) / 100.0
        tp_mult = st.slider("Take-Profit ATR Multiplier", min_value=1.5, max_value=4.5, value=2.5, step=0.5)

    with col_chart:
        st.markdown(f"### 📈 Live Strategy Sandbox Simulation ({ticker})")
        res = simulate_strategy_sandbox(
            ticker=ticker,
            leverage=lev,
            confidence_threshold=conf_thresh,
            tp_atr_multiplier=tp_mult,
        )
        if "total_return_pct" in res:
            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("Strategy Return", f"{res['total_return_pct']:+.2f}%", f"Benchmark: {res['benchmark_return_pct']:+.1f}%")
            with k2:
                st.metric("Sharpe Ratio", f"{res['sharpe_ratio']:.2f}")
            with k3:
                st.metric("Max Drawdown", f"{res['max_drawdown_pct']:.2f}%")

            if "chart_df" in res and not res["chart_df"].empty:
                st.line_chart(res["chart_df"], use_container_width=True)
        else:
            st.error(res.get("error", "Simulation error"))


# ==============================================================================
# 🚀 MAIN APPLICATION CONTROLLER
# ==============================================================================
def main():
    inject_custom_css()

    # --- Sidebar ---
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
                <div style="font-size: 2.2rem;">📈</div>
                <div style="font-size: 1.2rem; font-weight: 800;
                    background: linear-gradient(135deg, #00D4AA, #7C3AED);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    Sentilyze
                </div>
                <div style="color: #64748B; font-size: 0.75rem;">AI Trading Intelligence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # 1. Navigation Mode Selector
        nav_mode = st.radio(
            "Navigation Workspace",
            [
                "⚡ AI Command Center",
                "💼 Portfolio & Broker",
                "📊 Multi-Asset Fund & Risk",
                "🔬 Quantitative Research",
            ],
            index=0,
        )

        st.markdown("---")

        # 2. Specialist Ticker Selector
        selected_ticker = st.selectbox(
            "Specialist Asset",
            UNIVERSE_TICKERS,
            index=0,
            key="main_specialist_ticker",
            help="Select from the 17-stock universe",
        )

        st.markdown("---")

        # 3. Quick Multi-Channel Alert Trigger
        with st.expander("🔔 Alert Dispatchers", expanded=False):
            discord_url = st.text_input("Discord Webhook", type="password", placeholder="https://discord.com/api/webhooks/...")
            if st.button("📨 Send Discord Test"):
                if discord_url:
                    payload = format_signal_card(ticker=selected_ticker, signal="BUY", confidence=0.82, current_price=120.5, stop_loss=115.0, regime="BULLISH")
                    send_discord_alert(payload, webhook_url=discord_url)
                    st.success("Sent to Discord!")
                else:
                    st.warning("Enter Discord URL")

    # --- Top Hero Banner ---
    st.markdown(
        f"""
        <div class="hero-banner">
            <div>
                <h3 style="margin: 0; color: #F8FAFC; font-weight: 800;">Sentilyze Platform</h3>
                <p style="margin: 0.2rem 0 0 0; color: #94A3B8; font-size: 0.85rem;">
                    Active Mode: <b>{nav_mode}</b> &nbsp;·&nbsp; Analyzing <b>{selected_ticker}</b>
                </p>
            </div>
            <div style="font-size: 0.85rem; color: #00D4AA; font-weight: 600;">
                🟢 17 Models Active
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Workspace Routing ---
    if nav_mode == "⚡ AI Command Center":
        render_command_center(selected_ticker)
    elif nav_mode == "💼 Portfolio & Broker":
        render_portfolio_workspace()
    elif nav_mode == "📊 Multi-Asset Fund & Risk":
        render_fund_and_risk()
    elif nav_mode == "🔬 Quantitative Research":
        render_research_workspace(selected_ticker)


if __name__ == "__main__":
    main()
