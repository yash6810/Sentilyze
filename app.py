import os
import json
import numpy as np
import pandas as pd
import streamlit as st
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
from src.alpaca_broker import AlpacaBrokerBridge
from src.audio_briefing import synthesize_morning_audio
from src.discord_bot import handle_bot_command, send_bot_command_reply

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


# --- Luxury Glassmorphic Styling ---
def inject_luxury_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
            background-color: #090D16;
            color: #F1F5F9;
        }
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        
        /* Glassmorphic Luxury Header */
        .luxury-header {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(0, 212, 170, 0.25);
            border-radius: 16px;
            padding: 1.2rem 2rem;
            margin-bottom: 1.2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
        
        /* Glass Card */
        .glass-card {
            background: rgba(30, 41, 59, 0.5);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.25);
        }
        
        .section-badge {
            font-size: 1.15rem;
            font-weight: 800;
            color: #F8FAFC;
            margin: 1.2rem 0 0.8rem 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .section-badge::before {
            content: '';
            display: inline-block;
            width: 4px;
            height: 18px;
            background: #00D4AA;
            border-radius: 2px;
        }
        
        .stButton>button {
            border-radius: 8px;
            font-weight: 700;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 14px rgba(0, 212, 170, 0.3);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --- Interactive Candlestick / Price Chart with ATR Risk Bands ---
def render_plotly_candlestick(ticker: str, df: pd.DataFrame, curr_p: float, tp1: float, tp2: float, sl: float):
    """Renders high-frequency interactive Candlestick chart with ATR channels."""
    if df.empty or len(df) < 30:
        return
    recent_df = df.tail(60).copy()

    try:
        import plotly.graph_objects as go

        fig = go.Figure()

        # 1. Candlestick
        fig.add_trace(
            go.Candlestick(
                x=recent_df.index if isinstance(recent_df.index, pd.DatetimeIndex) else pd.to_datetime(recent_df.index),
                open=recent_df["Open"],
                high=recent_df["High"],
                low=recent_df["Low"],
                close=recent_df["Close"],
                name="Price",
                increasing_line_color="#10B981",
                decreasing_line_color="#EF4444",
            )
        )

        # 2. 7 MA & 21 MA
        if "ma7" in recent_df.columns:
            fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df["ma7"], line=dict(color="#38BDF8", width=1.5), name="7 MA"))
        if "ma21" in recent_df.columns:
            fig.add_trace(go.Scatter(x=recent_df.index, y=recent_df["ma21"], line=dict(color="#F59E0B", width=1.5), name="21 MA"))

        # 3. Take-Profit & Stop-Loss Target Lines
        fig.add_hline(y=tp1, line_dash="dash", line_color="#00D4AA", annotation_text=f"TP1 (+2.5 ATR): ${tp1:,.2f}", annotation_position="top right")
        fig.add_hline(y=tp2, line_dash="dot", line_color="#10B981", annotation_text=f"TP2 (+4.5 ATR): ${tp2:,.2f}", annotation_position="top right")
        fig.add_hline(y=sl, line_dash="dash", line_color="#EF4444", annotation_text=f"Stop-Loss: ${sl:,.2f}", annotation_position="bottom right")

        fig.update_layout(
            template="plotly_dark",
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.4)",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        # Fallback to Streamlit native line chart if plotly not yet installed
        chart_data = recent_df[["Close"]].copy()
        if "ma21" in recent_df.columns:
            chart_data["21 MA"] = recent_df["ma21"]
        st.line_chart(chart_data, use_container_width=True)



# ==============================================================================
# 🎯 WORKSPACE 1: AI TRADING COMMAND CENTER
# ==============================================================================
def render_command_center(ticker: str):
    st.markdown('<div class="section-badge">AI Momentum Inference & Intraday Market Radar</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1.8])

    df = pd.DataFrame()
    quote = fetch_live_quote(ticker)
    curr_p = float(quote.get("price", 0))
    chg = float(quote.get("change_pct", 0))
    tp1 = curr_p * 1.06
    tp2 = curr_p * 1.12
    sl = curr_p * 0.95

    with col1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.8rem; font-weight: 900; color: #00D4AA;">{ticker}</span>
                    <span style="font-size: 0.9rem; font-weight: bold; background: {'rgba(16, 185, 129, 0.2)' if chg>=0 else 'rgba(239, 68, 68, 0.2)'}; color: {'#10B981' if chg>=0 else '#EF4444'}; padding: 4px 10px; border-radius: 6px;">{chg:+.2f}%</span>
                </div>
                <div style="font-size: 2.2rem; font-weight: 900; margin: 0.3rem 0; font-family: 'JetBrains Mono', monospace;">${curr_p:,.2f}</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">High: ${quote.get('day_high', 0):,.2f} &nbsp;|&nbsp; Low: ${quote.get('day_low', 0):,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # AI Model Inference
        model_path = f"models/{ticker}_model.json"
        if os.path.exists(model_path) or os.path.exists(model_path.replace(".json", ".joblib")):
            with st.spinner(f"Running XGBoost + FinBERT for {ticker}..."):
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
                        <div class="glass-card" style="border: 1px solid {sig_color}55;">
                            <div style="font-size: 0.8rem; color: #94A3B8; letter-spacing: 0.05em;">QUANTITATIVE VERDICT</div>
                            <div style="font-size: 1.8rem; font-weight: 900; color: {sig_color};">{signal} &nbsp;<span style="font-size: 1rem; color: #E2E8F0;">({conf*100:.1f}% Confidence)</span></div>
                            <div style="font-size: 0.85rem; margin-top: 0.6rem; line-height: 1.6;">
                                • <b>TP1 (50% Scale-Out):</b> <span style="color:#00D4AA; font-family:monospace;">${tp1:,.2f}</span><br>
                                • <b>TP2 (Runner Target):</b> <span style="color:#10B981; font-family:monospace;">${tp2:,.2f}</span><br>
                                • <b>Stop-Loss:</b> <span style="color:#EF4444; font-family:monospace;">${sl:,.2f}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Inference error: {e}")

    with col2:
        # Interactive Candlestick Chart
        if not df.empty:
            render_plotly_candlestick(ticker, df, curr_p, tp1, tp2, sl)

    # 5-Minute Proximity Radar
    st.markdown('<div class="section-badge">📡 5-Minute Active Position Guardian & Proximity Radar</div>', unsafe_allow_html=True)
    broker = PaperBroker()
    open_pos = broker.state.get("open_positions", {})

    if open_pos:
        radar_cols = st.columns(min(len(open_pos), 3))
        for idx, (t_sym, pos) in enumerate(open_pos.items()):
            col_target = radar_cols[idx % len(radar_cols)]
            with col_target:
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
                    <div class="glass-card" style="margin-bottom: 0.5rem;">
                        <div style="display: flex; justify-content: space-between;">
                            <b style="color: #00D4AA;">{t_sym}</b>
                            <span style="color: {'#10B981' if pnl_ret>=0 else '#EF4444'}; font-weight: bold;">${live_p:,.2f} ({pnl_ret:+.2f}%)</span>
                        </div>
                        <div style="font-size: 0.75rem; color: #94A3B8; margin: 0.2rem 0;">TP1: ${tp1_val:,.2f} | SL: ${sl_val:,.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(prog, text=f"Progress to Take-Profit: {prog*100:.1f}%")
    else:
        st.info("No active open positions. Cash is liquid awaiting morning signals.")

    col_btn1, col_btn2 = st.columns([1, 2])
    with col_btn1:
        if st.button("⚡ Run 5-Minute Intraday Scan Now", use_container_width=True):
            with st.spinner("Checking live quotes against TP/SL triggers..."):
                res = evaluate_intraday_execution(broker=broker)
                trades = res.get("executed_trades", [])
                if trades:
                    st.success(f"Executed {len(trades)} exit trades on live market prices!")
                else:
                    st.info("All open positions are within target bands.")
                st.rerun()


# ==============================================================================
# 💼 WORKSPACE 2: PORTFOLIO & BROKER ($100k ACCOUNT)
# ==============================================================================
def render_portfolio_workspace():
    st.markdown('<div class="section-badge">Virtual Paper Trading Broker ($100,000 Portfolio)</div>', unsafe_allow_html=True)

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

    # Alpaca Connection Card & PDF Tearsheet
    col_alpaca, col_pdf = st.columns([1.5, 1.5])
    with col_alpaca:
        alpaca = AlpacaBrokerBridge()
        alp_acc = alpaca.get_account_summary()
        alp_status = "🟢 Connected (Alpaca Paper)" if alpaca.is_connected() else "⚪ Simulated Local Mode"
        st.markdown(
            f"""
            <div class="glass-card" style="padding: 0.8rem 1.2rem;">
                <div style="font-size: 0.8rem; color: #94A3B8;">BROKER EXECUTION BRIDGE</div>
                <div style="font-size: 1.1rem; font-weight: bold; color: #38BDF8;">{alp_status}</div>
                <div style="font-size: 0.8rem; color: #64748B;">Buying Power: ${alp_acc.get('buying_power', 200000.0):,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_pdf:
        pdf_bytes = generate_executive_pdf_tearsheet(
            portfolio_summary=summary,
            open_positions=list(broker.state.get("open_positions", {}).values()),
            equity_history_df=broker.get_equity_curve_df(),
        )
        st.download_button(
            "📄 Download 2-Page Executive Factsheet (PDF)",
            data=pdf_bytes,
            file_name=f"Sentilyze_Factsheet_{datetime.now(timezone.utc).strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    # Active Holdings Table
    st.markdown('<div class="section-badge">📦 Active Open Holdings (50/50 Scale-Out Model)</div>', unsafe_allow_html=True)
    open_df = broker.get_open_positions_df()
    if not open_df.empty:
        st.dataframe(open_df, use_container_width=True, hide_index=True)
    else:
        st.info("No open positions. Ready to deploy cash into morning high-conviction signals.")

    # Equity Curve & Closed Trades
    col_eq, col_jrnl = st.columns([1.5, 1.5])
    with col_eq:
        st.markdown('<div class="section-badge">📈 Equity Growth Curve</div>', unsafe_allow_html=True)
        eq_df = broker.get_equity_curve_df()
        if not eq_df.empty and "total_equity" in eq_df.columns:
            st.line_chart(eq_df["total_equity"], use_container_width=True)

    with col_jrnl:
        st.markdown('<div class="section-badge">📜 Closed Trade History Journal</div>', unsafe_allow_html=True)
        closed_df = broker.get_closed_trades_df()
        if not closed_df.empty:
            st.dataframe(closed_df, use_container_width=True, hide_index=True)


# ==============================================================================
# 📊 WORKSPACE 3: MULTI-ASSET FUND & RISK ANALYTICS
# ==============================================================================
def render_fund_and_risk():
    st.markdown('<div class="section-badge">17-Asset Fund Allocation, Rebalancer & Stress Testing</div>', unsafe_allow_html=True)

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
# 🔬 WORKSPACE 4: QUANTITATIVE RESEARCH & DISCORD BOT
# ==============================================================================
def render_research_workspace(ticker: str):
    st.markdown('<div class="section-badge">Quantitative Sandbox & Interactive Bot Console</div>', unsafe_allow_html=True)

    tab_sand, tab_bot = st.tabs(["⚙️ Strategy Optimizer & Sandbox", "🤖 Interactive Discord AI Bot Console"])

    with tab_sand:
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

    with tab_bot:
        st.markdown("### 🤖 Interactive Discord AI Bot Console")
        st.caption("Test slash commands live from your browser before or alongside mobile Discord execution.")

        col_cmd, col_exec = st.columns([2, 1])
        with col_cmd:
            user_cmd = st.text_input("Discord Command", value=f"/signal {ticker}", placeholder="/signal AMD, /portfolio, /execute, /var")
        with col_exec:
            st.write("")
            st.write("")
            run_cmd_btn = st.button("🚀 Test Execute Command", use_container_width=True)

        if run_cmd_btn and user_cmd:
            reply = handle_bot_command(user_cmd)
            st.markdown(
                f"""
                <div class="glass-card" style="border-left: 4px solid #00D4AA;">
                    <h4 style="margin: 0; color: #00D4AA;">{reply.get('title')}</h4>
                    <p style="margin-top: 0.5rem; white-space: pre-wrap; font-size: 0.9rem;">{reply.get('description')}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# 🚀 MAIN APPLICATION CONTROLLER
# ==============================================================================
def main():
    inject_luxury_css()

    # --- Sidebar ---
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
                <div style="font-size: 2.4rem;">📈</div>
                <div style="font-size: 1.3rem; font-weight: 900;
                    background: linear-gradient(135deg, #00D4AA, #7C3AED);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    Sentilyze
                </div>
                <div style="color: #64748B; font-size: 0.75rem; font-weight: 500;">AI Trading Intelligence</div>
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

        # 2. Specialist Ticker Selector (All 17 Assets)
        selected_ticker = st.selectbox(
            "Specialist Asset",
            UNIVERSE_TICKERS,
            index=0,
            key="main_specialist_ticker",
            help="Choose from the 17-stock pre-trained universe",
        )

        st.markdown("---")

        # 3. Audio Briefing Trigger
        st.markdown("**🎙️ AI Morning Audio Podcast**")
        if st.button("📻 Generate & Play Audio Briefing"):
            with st.spinner("Synthesizing audio briefing..."):
                audio_path = synthesize_morning_audio()
                if audio_path and os.path.exists(audio_path):
                    st.audio(audio_path, format="audio/mp3")
                else:
                    st.info("Audio generated in text briefing format.")

    # --- Top Luxury Header ---
    st.markdown(
        f"""
        <div class="luxury-header">
            <div>
                <h3 style="margin: 0; color: #F8FAFC; font-weight: 900; letter-spacing: -0.02em;">Sentilyze Terminal</h3>
                <p style="margin: 0.2rem 0 0 0; color: #94A3B8; font-size: 0.85rem;">
                    Active Workspace: <b style="color: #F1F5F9;">{nav_mode}</b> &nbsp;·&nbsp; Specialist: <b style="color: #00D4AA;">{selected_ticker}</b>
                </p>
            </div>
            <div style="font-size: 0.85rem; color: #00D4AA; font-weight: 700; background: rgba(0, 212, 170, 0.1); border: 1px solid rgba(0, 212, 170, 0.3); padding: 4px 12px; border-radius: 20px;">
                🟢 17 Specialist Models Live
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
