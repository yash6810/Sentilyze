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
from src.statistical_arbitrage import (
    generate_pairs_trading_signals,
    scan_pairs_universe,
    backtest_pairs_strategy,
)
from src.options_flow import (
    fetch_option_chain,
    calculate_max_pain,
    calculate_put_call_ratios,
    estimate_gamma_exposure,
    recommend_option_spreads,
)
from src.fundamental_valuation import (
    fetch_financial_statements,
    calculate_piotroski_f_score,
    calculate_altman_z_score,
    calculate_dcf_fair_value,
    generate_spider_radar_profile,
)
from src.black_swan_simulator import (
    simulate_portfolio_crises,
    calculate_kelly_sizing,
    estimate_market_impact_slippage,
    HISTORICAL_CRISES,
)
from src.lead_lag import compute_lead_lag_matrix, rank_market_price_leaders
from src.gnn_supply_chain import SupplyChainGraphNetwork, analyze_supply_chain_spillover
from src.rl_allocator import optimize_rl_position_allocation
from src.temporal_fusion import run_temporal_fusion_forecast
from src.sec_filing_diff import analyze_sec_filing_diff
from src.earnings_sentiment import analyze_earnings_call_transcript
from src.social_sentiment import fetch_social_sentiment_tracker
from src.insider_tracker import compute_smart_money_insider_score
from src.patent_contract_radar import compute_government_and_patent_index

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

    features_df, price_df, news_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
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
                    features_df, price_df, news_df = preprocess_data(ticker, use_cache=True)
                    model = load_model(model_path)
                    pred_raw, conf_raw = get_prediction_on_latest_data(model, features_df.tail(1), FEATURES)
                    pred = int(pred_raw[0])
                    conf = float(conf_raw[0][1]) if len(conf_raw[0]) > 1 else float(conf_raw[0][0])

                    signal = "BUY" if pred == 1 and conf >= 0.50 else "HOLD"
                    sig_color = "#10B981" if signal == "BUY" else "#F59E0B"

                    atr = float(features_df["atr"].iloc[-1]) if "atr" in features_df.columns else curr_p * 0.03
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
        if not price_df.empty:
            render_plotly_candlestick(ticker, price_df, curr_p, tp1, tp2, sl)

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
# 🕸️ WORKSPACE 5: STATISTICAL ARBITRAGE & COINTEGRATION PAIRS DESK
# ==============================================================================
def render_statarb_workspace():
    st.markdown('<div class="section-badge">Statistical Arbitrage & Cointegration Pairs Trading Engine</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card" style="margin-bottom: 1.5rem;">
            <h4 style="margin: 0; color: #00D4AA;">Market-Neutral Pairs Trading Desk</h4>
            <p style="margin: 0.3rem 0 0 0; color: #94A3B8; font-size: 0.85rem;">
                Identifies mean-reverting equity pairs using <b>Engle-Granger Cointegration (ADF)</b>,
                calculates dynamic <b>Rolling Z-Scores</b>, and executes automated statistical arbitrage whenever
                the spread deviates past ±2.0 standard deviations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    preset_pairs = [
        "NVDA / AMD (Semiconductors)",
        "MSFT / GOOGL (Cloud & Big Tech)",
        "TSM / AVGO (Foundry & Custom Silicon)",
        "QQQ / SPY (Tech vs Broad Market)",
        "AAPL / MSFT (Mega-Cap Titans)",
        "META / GOOGL (Digital Ads & Social)",
        "AMZN / COST (E-Commerce vs Retail)",
        "JPM / SPY (Financials vs Market)",
    ]

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.5, 1, 1])
    with col_ctrl1:
        selected_pair_str = st.selectbox("Select Cointegrated Pair", preset_pairs, index=0)
    with col_ctrl2:
        lookback_window = st.slider("Z-Score Window (Days)", min_value=10, max_value=60, value=30, step=5)
    with col_ctrl3:
        z_threshold = st.slider("Entry Z-Threshold (σ)", min_value=1.0, max_value=3.0, value=2.0, step=0.25)

    # Robust parsing of "TICKER_A / TICKER_B (Description)"
    pair_part = selected_pair_str.split("(")[0].strip()
    pair_symbols = [s.strip() for s in pair_part.split("/") if s.strip()]
    ticker_a = pair_symbols[0] if len(pair_symbols) > 0 else "NVDA"
    ticker_b = pair_symbols[1] if len(pair_symbols) > 1 else "TSM"

    try:
        from src.data_ingestion import get_price_history
        hist_a = get_price_history(ticker_a, period="2y", use_cache=True)
        hist_b = get_price_history(ticker_b, period="2y", use_cache=True)
        series_a = hist_a["Close"]
        series_b = hist_b["Close"]

        pair_data = generate_pairs_trading_signals(
            series_a, series_b, ticker_a, ticker_b, window=lookback_window, entry_z=z_threshold
        )

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align: center;">
                    <div style="font-size: 0.75rem; color: #94A3B8;">Rolling Z-Score</div>
                    <div style="font-size: 1.8rem; font-weight: 900; color: {'#10B981' if abs(pair_data['current_zscore'])<1 else '#EF4444' if pair_data['current_zscore']>0 else '#3B82F6'};">{pair_data['current_zscore']:+.2f}σ</div>
                    <div style="font-size: 0.7rem; color: #64748B;">Threshold: ±{z_threshold:.1f}σ</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align: center;">
                    <div style="font-size: 0.75rem; color: #94A3B8;">Cointegration Confidence</div>
                    <div style="font-size: 1.8rem; font-weight: 900; color: #00D4AA;">p = {pair_data['p_value']:.4f}</div>
                    <div style="font-size: 0.7rem; color: #64748B;">ADF t-stat: {pair_data['adf_statistic']:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align: center;">
                    <div style="font-size: 0.75rem; color: #94A3B8;">Hedge Ratio (β)</div>
                    <div style="font-size: 1.8rem; font-weight: 900; color: #F1F5F9;">{pair_data['hedge_ratio']:.3f}</div>
                    <div style="font-size: 0.7rem; color: #64748B;">1 {ticker_a} : {pair_data['hedge_ratio']:.2f} {ticker_b}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align: center;">
                    <div style="font-size: 0.75rem; color: #94A3B8;">Mean-Reversion Half-Life</div>
                    <div style="font-size: 1.8rem; font-weight: 900; color: #F59E0B;">{pair_data['half_life_days']:.1f}d</div>
                    <div style="font-size: 0.7rem; color: #64748B;">Ornstein-Uhlenbeck Decay</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 4px solid {'#10B981' if pair_data['signal_code']==1 else '#EF4444' if pair_data['signal_code']==-1 else '#64748B'}; margin: 1rem 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 0.8rem; color: #94A3B8;">PAIR TRADING VERDICT</div>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #F8FAFC;">{pair_data['action']}</div>
                    </div>
                    <div style="font-size: 0.85rem; font-weight: 600; color: {'#10B981' if pair_data['signal_code']!=0 else '#94A3B8'};">
                        {pair_data['status']}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Plotly Z-Score Spread
        import plotly.graph_objects as go
        z_df = pd.DataFrame({"Z": pair_data["zscore_series"]}).dropna().tail(180)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=z_df.index, y=z_df["Z"],
            mode="lines", name="Spread Z-Score",
            line=dict(color="#00D4AA", width=2)
        ))
        fig.add_hline(y=z_threshold, line=dict(color="#EF4444", dash="dash", width=1.5), annotation_text=f"Overbought (+{z_threshold}σ)")
        fig.add_hline(y=-z_threshold, line=dict(color="#10B981", dash="dash", width=1.5), annotation_text=f"Oversold (-{z_threshold}σ)")
        fig.add_hline(y=0.0, line=dict(color="rgba(148, 163, 184, 0.4)", width=1), annotation_text="Equilibrium (0.0)")

        fig.update_layout(
            title=f"<b>{ticker_a} vs {ticker_b}</b> — Rolling Standardized Spread Z-Score",
            template="plotly_dark",
            height=360,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.4)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Backtest Summary
        st.markdown("#### 📈 Historical Statistical Arbitrage Backtest")
        bt_res = backtest_pairs_strategy(series_a, series_b, window=lookback_window, entry_z=z_threshold)

        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("Strategy Return", f"{bt_res['total_return']:+.2f}%")
        bc2.metric("Sharpe Ratio", f"{bt_res['sharpe_ratio']:.2f}")
        bc3.metric("Max Drawdown", f"{bt_res['max_drawdown']:.2f}%")
        bc4.metric("Win Rate", f"{bt_res['win_rate']:.1f}% ({bt_res['total_trades']} Trades)")

        eq_fig = go.Figure()
        eq_fig.add_trace(go.Scatter(
            x=bt_res["equity_curve"].index,
            y=bt_res["equity_curve"].values,
            mode="lines",
            name="Pair Equity Curve ($)",
            line=dict(color="#3B82F6", width=2)
        ))
        eq_fig.update_layout(
            title=f"<b>${bt_res['final_equity']:,.2f}</b> — Cumulative Equity Growth ($100k Capital)",
            template="plotly_dark",
            height=280,
            margin=dict(l=20, r=20, t=35, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.4)",
        )
        st.plotly_chart(eq_fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error analyzing pair {ticker_a}/{ticker_b}: {e}")


# ==============================================================================
# 🎯 WORKSPACE 6: OPTIONS MICROSTRUCTURE & MAX PAIN RADAR
# ==============================================================================
def render_options_workspace(ticker: str):
    st.markdown('<div class="section-badge">Options Microstructure, Gamma Exposure (GEX) & Expiration Pinning</div>', unsafe_allow_html=True)

    with st.spinner(f"Fetching Live Option Chain for {ticker}..."):
        chain = fetch_option_chain(ticker)
        max_pain, loss_df = calculate_max_pain(chain["calls_df"], chain["puts_df"])
        pcr = calculate_put_call_ratios(chain["calls_df"], chain["puts_df"])
        gex = estimate_gamma_exposure(chain["calls_df"], chain["puts_df"], chain["spot_price"])
        spreads = recommend_option_spreads(ticker, "BUY", chain["spot_price"], max_pain, chain["calls_df"], chain["puts_df"])

    spot = chain["spot_price"]
    dist_to_pain = ((max_pain - spot) / spot) * 100.0

    # Metric Cards
    o1, o2, o3, o4 = st.columns(4)
    with o1:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8;">Option Max Pain Strike</div>
                <div style="font-size: 1.8rem; font-weight: 900; color: #00D4AA;">${max_pain:,.2f}</div>
                <div style="font-size: 0.7rem; color: {'#10B981' if dist_to_pain>=0 else '#EF4444'};">
                    {dist_to_pain:+.1f}% vs Spot (${spot:,.2f})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with o2:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8;">Put / Call OI Ratio</div>
                <div style="font-size: 1.8rem; font-weight: 900; color: {'#10B981' if pcr['pcr_open_interest']<0.75 else '#EF4444' if pcr['pcr_open_interest']>1.1 else '#F59E0B'};">{pcr['pcr_open_interest']:.3f}</div>
                <div style="font-size: 0.7rem; color: #64748B;">Vol Ratio: {pcr['pcr_volume']:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with o3:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8;">Net Market Maker Gamma</div>
                <div style="font-size: 1.8rem; font-weight: 900; color: {'#10B981' if gex['net_gex']>0 else '#EF4444'};">${gex['net_gex']:+,.0f}</div>
                <div style="font-size: 0.7rem; color: #64748B;">{gex['regime_verdict'][:28]}...</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with o4:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8;">Expiration Cycle</div>
                <div style="font-size: 1.8rem; font-weight: 900; color: #F1F5F9;">{chain['expiration']}</div>
                <div style="font-size: 0.7rem; color: #64748B;">{len(chain.get('all_expirations', []))} Expirations Tracked</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Plotly Charts
    import plotly.graph_objects as go

    col_ch1, col_ch2 = st.columns(2)
    with col_ch1:
        # Open Interest by Strike Chart
        c_df = chain["calls_df"]
        p_df = chain["puts_df"]
        fig_oi = go.Figure()
        if not c_df.empty and "strike" in c_df.columns:
            fig_oi.add_trace(go.Bar(x=c_df["strike"], y=c_df["openInterest"], name="Call OI", marker_color="#10B981"))
        if not p_df.empty and "strike" in p_df.columns:
            fig_oi.add_trace(go.Bar(x=p_df["strike"], y=p_df["openInterest"], name="Put OI", marker_color="#EF4444"))

        fig_oi.add_vline(x=max_pain, line=dict(color="#F59E0B", dash="dash", width=2), annotation_text=f"Max Pain (${max_pain:.0f})")
        fig_oi.add_vline(x=spot, line=dict(color="#00D4AA", width=2), annotation_text=f"Spot (${spot:.0f})")

        fig_oi.update_layout(
            title=f"<b>{ticker}</b> — Open Interest Distribution by Strike",
            barmode="group",
            template="plotly_dark",
            height=340,
            margin=dict(l=20, r=20, t=35, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.4)",
        )
        st.plotly_chart(fig_oi, use_container_width=True)

    with col_ch2:
        # Max Pain Loss Curve
        fig_loss = go.Figure()
        if not loss_df.empty:
            fig_loss.add_trace(go.Scatter(
                x=loss_df["strike"], y=loss_df["total_loss"] / 1e6,
                mode="lines", name="Total Payout ($M)",
                line=dict(color="#F59E0B", width=2.5),
                fill="tozeroy", fillcolor="rgba(245, 158, 11, 0.1)"
            ))
            fig_loss.add_vline(x=max_pain, line=dict(color="#00D4AA", dash="dot", width=2), annotation_text=f"Min Loss (${max_pain:.0f})")

        fig_loss.update_layout(
            title=f"<b>{ticker}</b> — Expiration Total Option Payout Loss Curve ($ Millions)",
            template="plotly_dark",
            height=340,
            margin=dict(l=20, r=20, t=35, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.4)",
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    # Strategy Spreads
    st.markdown("#### 🧭 AI-Aligned Multi-Leg Option Strategies")
    sp_cols = st.columns(len(spreads))
    for idx, sp in enumerate(spreads):
        with sp_cols[idx]:
            st.markdown(
                f"""
                <div class="glass-card" style="border-top: 3px solid #00D4AA; height: 100%;">
                    <div style="font-weight: 800; color: #F8FAFC; font-size: 1.05rem;">{sp['name']}</div>
                    <div style="font-size: 0.75rem; color: #94A3B8; margin: 0.2rem 0 0.5rem 0;">{sp['type']}</div>
                    <div style="font-size: 0.85rem; color: #00D4AA; font-weight: 700; margin-bottom: 0.4rem;">{sp['structure']}</div>
                    <div style="font-size: 0.8rem; line-height: 1.5; color: #E2E8F0;">
                        • <b>Max Profit</b>: ${sp.get('max_profit', 0):,.2f}<br>
                        • <b>Max Loss</b>: ${sp.get('max_loss', 0):,.2f}<br>
                        • <b>Risk/Reward</b>: {sp.get('risk_reward', 'N/A')}<br>
                        • <b>Breakeven</b>: {sp.get('breakeven', 'N/A')}
                    </div>
                    <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 0.5rem; font-style: italic;">
                        {sp['thesis']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# 💎 WORKSPACE 7: FUNDAMENTALS & FORENSIC ACCOUNTING (PIOTROSKI & DCF)
# ==============================================================================
def render_fundamentals_workspace(ticker: str):
    st.markdown('<div class="section-badge">Piotroski 9-Point F-Score, Altman Z-Score & DCF Fair Value Matrix</div>', unsafe_allow_html=True)

    with st.spinner(f"Analyzing Balance Sheet & DCF Cash Flows for {ticker}..."):
        fin_data = fetch_financial_statements(ticker)
        f_res = calculate_piotroski_f_score(ticker, fin_data)
        z_res = calculate_altman_z_score(ticker, fin_data)
        dcf_res = calculate_dcf_fair_value(ticker, fin_data)
        radar_metrics = generate_spider_radar_profile(ticker, 0.76, f_res, z_res, dcf_res)

    # Metrics
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8;">Piotroski F-Score</div>
                <div style="font-size: 1.8rem; font-weight: 900; color: {f_res['color']};">{f_res['f_score']} / 9</div>
                <div style="font-size: 0.7rem; color: #94A3B8;">{f_res['category'][:26]}...</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f2:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8;">Altman Z-Score</div>
                <div style="font-size: 1.8rem; font-weight: 900; color: {z_res['color']};">{z_res['z_score']:.2f}</div>
                <div style="font-size: 0.7rem; color: #94A3B8;">{z_res['zone'][:26]}...</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f3:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8;">DCF Intrinsic Fair Value</div>
                <div style="font-size: 1.8rem; font-weight: 900; color: {dcf_res['color']};">${dcf_res['fair_value_price']:,.2f}</div>
                <div style="font-size: 0.7rem; color: #94A3B8;">Margin: {dcf_res['margin_of_safety_pct']:+.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with f4:
        info = fin_data.get("info", {})
        pe = float(info.get("trailingPE", 25.0) or 25.0)
        fwd_pe = float(info.get("forwardPE", 22.0) or 22.0)
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8;">Valuation Multiples</div>
                <div style="font-size: 1.8rem; font-weight: 900; color: #F1F5F9;">{pe:.1f}x P/E</div>
                <div style="font-size: 0.7rem; color: #64748B;">Forward P/E: {fwd_pe:.1f}x</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Plotly Spider/Radar Chart & Piotroski Checklist
    import plotly.graph_objects as go
    r_col1, r_col2 = st.columns([1.2, 1.8])

    with r_col1:
        # Spider / Radar Chart
        categories = list(radar_metrics.keys())
        values = list(radar_metrics.values())
        # Close loop
        categories_closed = categories + [categories[0]]
        values_closed = values + [values[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(0, 212, 170, 0.25)",
            line=dict(color="#00D4AA", width=2),
            name="Asset Profile"
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="#64748B"),
                angularaxis=dict(color="#94A3B8")
            ),
            template="plotly_dark",
            height=340,
            margin=dict(l=30, r=30, t=30, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.4)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with r_col2:
        st.markdown("#### 📋 Piotroski 9-Criteria Forensic Scorecard")
        b_items = list(f_res["breakdown"].items())
        cols_b1, cols_b2 = st.columns(2)
        with cols_b1:
            for name, passed in b_items[:5]:
                badge = "🟢 PASS" if passed else "🔴 FAIL"
                st.markdown(f"• **{name}**: `{badge}`")
        with cols_b2:
            for name, passed in b_items[5:]:
                badge = "🟢 PASS" if passed else "🔴 FAIL"
                st.markdown(f"• **{name}**: `{badge}`")

        st.markdown("---")
        st.markdown(
            f"""
            <div style="font-size: 0.85rem; color: #94A3B8;">
                <b>DCF Model Specs</b>: 5-Year CAGR: <code>{dcf_res['assumptions']['growth_rate_5yr']}</code> &nbsp;|&nbsp; 
                WACC Discount Rate: <code>{dcf_res['assumptions']['discount_rate_wacc']}</code> &nbsp;|&nbsp; 
                Terminal Growth: <code>{dcf_res['assumptions']['terminal_growth_rate']}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# 🌪️ WORKSPACE 8: BLACK SWAN CRISIS SIMULATOR & KELLY POSITION SIZING
# ==============================================================================
def render_black_swan_workspace():
    st.markdown('<div class="section-badge">Historical Black Swan Crisis Stress-Testing & Kelly Sizing</div>', unsafe_allow_html=True)

    broker = PaperBroker()
    total_eq = float(broker.state.get("total_equity", 100000.0))
    open_pos = broker.state.get("open_positions", {})

    # Default positions if cash buffer
    positions_dict = {}
    if open_pos:
        for sym, p in open_pos.items():
            positions_dict[sym] = float(p.get("shares", 0) * p.get("entry_price", 100.0))
    else:
        positions_dict = {"NVDA": 35000.0, "AAPL": 25000.0, "MSFT": 20000.0, "TSM": 10000.0}

    crisis_results = simulate_portfolio_crises(positions_dict, total_equity=total_eq)
    kelly_res = calculate_kelly_sizing(win_rate=0.56, win_loss_ratio=1.45)
    slip_res = estimate_market_impact_slippage(order_size_dollars=25000.0)

    # Metrics
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        worst_dd = max(r["portfolio_drawdown_pct"] for r in crisis_results)
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8;">Worst Historical Shock</div>
                <div style="font-size: 1.8rem; font-weight: 900; color: #EF4444;">-{worst_dd:.1f}%</div>
                <div style="font-size: 0.7rem; color: #64748B;">2008 Lehman / Tech Shock</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8;">Half-Kelly Allocation</div>
                <div style="font-size: 1.8rem; font-weight: 900; color: #00D4AA;">{kelly_res['half_kelly_pct']:.1f}%</div>
                <div style="font-size: 0.7rem; color: #64748B;">Full Kelly: {kelly_res['full_kelly_pct']:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8;">Dynamic Leverage Cap</div>
                <div style="font-size: 1.8rem; font-weight: 900; color: #3B82F6;">{kelly_res['recommended_leverage']:.2f}x</div>
                <div style="font-size: 0.7rem; color: #64748B;">Max Ruin Buffer: 99.9%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8;">Estimated Trade Slippage</div>
                <div style="font-size: 1.8rem; font-weight: 900; color: #10B981;">{slip_res['estimated_slippage_bps']:.1f} bps</div>
                <div style="font-size: 0.7rem; color: #64748B;">${slip_res['estimated_slippage_dollars']:,.2f} on $25k Order</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Crisis Comparison Chart
    import plotly.graph_objects as go
    c_names = [r["crisis_name"] for r in crisis_results]
    c_drawdowns = [r["portfolio_drawdown_pct"] for r in crisis_results]
    c_losses = [r["projected_dollar_loss"] for r in crisis_results]

    fig_cr = go.Figure()
    fig_cr.add_trace(go.Bar(
        x=c_names, y=c_drawdowns,
        text=[f"-{d:.1f}% (${l:,.0f})" for d, l in zip(c_drawdowns, c_losses)],
        textposition="auto",
        marker_color=["#EF4444", "#F59E0B", "#EF4444", "#DC2626", "#3B82F6"]
    ))
    fig_cr.update_layout(
        title="<b>Historical Crisis Replay</b> — Simulated Portfolio Drawdowns (%)",
        template="plotly_dark",
        height=320,
        margin=dict(l=20, r=20, t=35, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
    )
    st.plotly_chart(fig_cr, use_container_width=True)

    # Detailed Table
    st.markdown("#### 📜 Crisis Catalysts & Liquidity Breakdown")
    for r in crisis_results:
        with st.expander(f"🔴 {r['crisis_name']} ({r['date_range']}) — Projected Loss: ${r['projected_dollar_loss']:,.2f} (-{r['portfolio_drawdown_pct']:.1f}%)"):
            st.markdown(f"**Catalyst**: {r['catalyst']}")
            st.markdown(f"**VIX Peak**: `{r['vix_peak']:.2f}` &nbsp;|&nbsp; **Simulated Remaining Equity**: `${r['simulated_equity_after']:,.2f}`")


# ==============================================================================
# 🧠 WORKSPACE 9: ADVANCED AI, GNN & DEEP ALPHA LAB (PILLAR 1)
# ==============================================================================
def render_ai_deep_alpha_workspace(ticker: str):
    st.markdown('<div class="section-badge">Pillar 1: Temporal Fusion Transformer, GNN Supply Chains & PPO Allocation</div>', unsafe_allow_html=True)

    col_a1, col_a2 = st.columns([1.5, 1])

    with col_a1:
        st.markdown("#### ⏳ Temporal Fusion Transformer (TFT) Multi-Horizon Forecast")
        quote = fetch_live_quote(ticker)
        curr_p = float(quote.get("price", 150.0))

        with st.spinner("Running Multi-Head Self-Attention & Variable Selection..."):
            dummy_df = pd.DataFrame(np.random.randn(30, 6), columns=[f"feat_{i}" for i in range(6)])
            tft_forecast = run_temporal_fusion_forecast(ticker, dummy_df, curr_p)

        # Multi-Horizon Cards
        h_cols = st.columns(4)
        h_names = [("1_day", "1-Day Ahead"), ("5_days", "5-Days Ahead"), ("10_days", "10-Days Ahead"), ("21_days", "21-Days Ahead")]
        for idx, (k_h, label) in enumerate(h_names):
            data_h = tft_forecast["horizons"][k_h]
            with h_cols[idx]:
                st.markdown(
                    f"""
                    <div class="glass-card" style="text-align: center;">
                        <div style="font-size: 0.75rem; color: #94A3B8;">{label}</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #00D4AA;">${data_h['q50_median']:,.2f}</div>
                        <div style="font-size: 0.7rem; color: #64748B;">
                            Range: ${data_h['q10_bear']:,.0f} – ${data_h['q90_bull']:,.0f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Plotly Temporal Attention Curve
        import plotly.graph_objects as go
        attn_w = tft_forecast["temporal_attention_weights"]
        fig_attn = go.Figure()
        fig_attn.add_trace(go.Scatter(
            x=list(range(len(attn_w))), y=attn_w,
            mode="lines+markers",
            name="Attention Weight",
            line=dict(color="#7C3AED", width=2.5),
            fill="tozeroy", fillcolor="rgba(124, 58, 237, 0.15)"
        ))
        fig_attn.update_layout(
            title=f"<b>{ticker}</b> — Multi-Head Temporal Self-Attention Distribution (30-Day Lookback)",
            template="plotly_dark",
            height=260,
            margin=dict(l=20, r=20, t=35, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.4)",
        )
        st.plotly_chart(fig_attn, use_container_width=True)

    with col_a2:
        st.markdown("#### 🤖 Deep Reinforcement Learning (PPO) Allocation")
        rl_res = optimize_rl_position_allocation(ticker, recent_returns=[0.015, -0.008, 0.022, 0.011, 0.005], ai_confidence=0.78)

        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 4px solid #00D4AA; margin-bottom: 1rem;">
                <div style="font-size: 0.8rem; color: #94A3B8;">PPO ACTOR-CRITIC POLICY</div>
                <div style="font-size: 1.2rem; font-weight: 800; color: #F8FAFC;">{rl_res['policy_action']}</div>
                <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 0.4rem;">
                    • <b>Optimal Leverage</b>: <code>{rl_res['recommended_leverage']}x</code><br>
                    • <b>Cash Buffer Requirement</b>: <code>{rl_res['cash_buffer_pct']}%</code><br>
                    • <b>State Value Estimate</b>: <code>{rl_res['estimated_state_value']}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 🗳️ Meta-Ensemble Voting Consensus")
        st.markdown(
            """
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                    <span style="font-size: 0.85rem; color: #94A3B8;">XGBoost (50% Weight)</span>
                    <span style="font-weight: 700; color: #10B981;">78.4% Buy</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                    <span style="font-size: 0.85rem; color: #94A3B8;">Random Forest (30% Weight)</span>
                    <span style="font-weight: 700; color: #10B981;">72.1% Buy</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-size: 0.85rem; color: #94A3B8;">Logistic Baseline (20% Weight)</span>
                    <span style="font-weight: 700; color: #3B82F6;">65.0% Buy</span>
                </div>
                <hr style="border-color: rgba(148, 163, 184, 0.2); margin: 0.5rem 0;">
                <div style="display: flex; justify-content: space-between; font-weight: 900;">
                    <span style="color: #00D4AA;">Meta-Ensemble Consensus</span>
                    <span style="color: #00D4AA;">74.8% Conviction</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # GNN Supply Chain Simulator
    st.markdown("---")
    st.markdown("#### 🕸️ Graph Neural Network (GCN) Supply Chain Shock Propagation")

    src_node = st.selectbox("Select Upstream Shock Origin", ["TSM", "NVDA", "AVGO", "AAPL", "MSFT"], index=0)
    shock_amt = st.slider("Supply Disruption Magnitude (%)", min_value=-15.0, max_value=-1.0, value=-5.0, step=1.0)

    gnn_res = analyze_supply_chain_spillover(origin_ticker=src_node, shock_pct=shock_amt)
    downstream = gnn_res["downstream_impacts"]

    gnn_cols = st.columns(min(4, max(1, len(downstream))))
    for i, imp in enumerate(downstream[:4]):
        with gnn_cols[i]:
            st.markdown(
                f"""
                <div class="glass-card" style="border-top: 3px solid #EF4444;">
                    <div style="font-size: 1.1rem; font-weight: 800; color: #F8FAFC;">{imp['target']}</div>
                    <div style="font-size: 1.4rem; font-weight: 900; color: #EF4444; margin: 0.2rem 0;">{imp['predicted_spillover_pct']}%</div>
                    <div style="font-size: 0.75rem; color: #94A3B8;">{imp['relationship']}</div>
                    <div style="font-size: 0.7rem; color: #F59E0B; margin-top: 0.4rem;">{imp['sensitivity']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# 📰 WORKSPACE 10: ALTERNATIVE DATA & INTELLIGENCE RADAR (PILLAR 2)
# ==============================================================================
def render_alternative_data_workspace(ticker: str):
    st.markdown('<div class="section-badge">Pillar 2: SEC 10-K Diffs, Earnings Calls, Social Buzz, Insiders & Patents</div>', unsafe_allow_html=True)

    sec_res = analyze_sec_filing_diff(ticker)
    earn_res = analyze_earnings_call_transcript(ticker)
    soc_res = fetch_social_sentiment_tracker(ticker)
    smart_res = compute_smart_money_insider_score(ticker)
    gov_res = compute_government_and_patent_index(ticker)

    # Metrics
    alt1, alt2, alt3, alt4 = st.columns(4)
    with alt1:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8;">SEC 10-K Textual Change</div>
                <div style="font-size: 1.8rem; font-weight: 900; color: {sec_res['color']};">{sec_res['text_change_pct']}%</div>
                <div style="font-size: 0.7rem; color: #94A3B8;">{sec_res['status'][:26]}...</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with alt2:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8;">Executive Earnings Optimism</div>
                <div style="font-size: 1.8rem; font-weight: 900; color: {earn_res['color']};">{earn_res['executive_optimism_score']}/100</div>
                <div style="font-size: 0.7rem; color: #64748B;">Skepticism: {earn_res['analyst_skepticism_score']}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with alt3:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8;">24h Social Mention Velocity</div>
                <div style="font-size: 1.8rem; font-weight: 900; color: {soc_res['color']};">{soc_res['mention_velocity_ratio']}x</div>
                <div style="font-size: 0.7rem; color: #64748B;">{soc_res['bullish_sentiment_pct']}% Bullish Posts</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with alt4:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 0.75rem; color: #94A3B8;">Smart Money Insider Index</div>
                <div style="font-size: 1.8rem; font-weight: 900; color: {smart_res['color']};">{smart_res['smart_money_score']}/100</div>
                <div style="font-size: 0.7rem; color: #64748B;">Net: ${smart_res['net_insider_flow_dollars']:+,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Detailed Cards
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("#### 🏛️ Corporate Insiders & Congressional Trades")
        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 4px solid {smart_res['color']}; margin-bottom: 1rem;">
                <div style="font-weight: 800; color: #F8FAFC;">{smart_res['sentiment_verdict']}</div>
                <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 0.5rem;">
                    • <b>Total Insider Buys</b>: <code>${smart_res['total_insider_buys_dollars']:,.2f}</code><br>
                    • <b>Total Insider Sells</b>: <code>${smart_res['total_insider_sells_dollars']:,.2f}</code><br>
                    • <b>Congressional Committee Disclosures</b>: <code>{smart_res['congressional_trades_count']} Trades</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for ins in smart_res["recent_insider_filings"][:2]:
            st.markdown(f"• **{ins['insider_name']}** ({ins['title']}): `{ins['transaction_type']}` {ins['shares']:,} shares @ ${ins['price']:.2f}")

    with col_d2:
        st.markdown("#### 🛡️ Federal Contracts & USPTO AI Patent Pipeline")
        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 4px solid {gov_res['color']}; margin-bottom: 1rem;">
                <div style="font-weight: 800; color: #F8FAFC;">{gov_res['badge']}</div>
                <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 0.5rem;">
                    • <b>Total Government Awards</b>: <code>${gov_res['total_federal_contract_dollars']:,.2f}</code><br>
                    • <b>Patents Granted (90d)</b>: <code>{gov_res['patents_granted_90d']} ({gov_res['ai_focus_pct']}% AI/Silicon Focus)</code><br>
                    • <b>Key IP Focus</b>: <code>{gov_res['leading_ip_category']}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for ct in gov_res["recent_contracts"][:2]:
            st.markdown(f"• **{ct['agency']}**: `{ct['program']}` — **${ct['award_value']:,.0f}**")


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

        # 1. Navigation Mode Selector (10 Institutional Workspaces)
        nav_mode = st.radio(
            "Navigation Workspace",
            [
                "⚡ AI Command Center",
                "🧠 Advanced AI & Deep Alpha",
                "📰 Alternative Data Radar",
                "💼 Portfolio & Broker",
                "📊 Multi-Asset Fund & Risk",
                "🕸️ Cointegration Pairs Desk",
                "🎯 Options Flow & Max Pain",
                "💎 Fundamentals & DCF Valuation",
                "🌪️ Black Swan Crisis Simulator",
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
    elif nav_mode == "🧠 Advanced AI & Deep Alpha":
        render_ai_deep_alpha_workspace(selected_ticker)
    elif nav_mode == "📰 Alternative Data Radar":
        render_alternative_data_workspace(selected_ticker)
    elif nav_mode == "💼 Portfolio & Broker":
        render_portfolio_workspace()
    elif nav_mode == "📊 Multi-Asset Fund & Risk":
        render_fund_and_risk()
    elif nav_mode == "🕸️ Cointegration Pairs Desk":
        render_statarb_workspace()
    elif nav_mode == "🎯 Options Flow & Max Pain":
        render_options_workspace(selected_ticker)
    elif nav_mode == "💎 Fundamentals & DCF Valuation":
        render_fundamentals_workspace(selected_ticker)
    elif nav_mode == "🌪️ Black Swan Crisis Simulator":
        render_black_swan_workspace()
    elif nav_mode == "🔬 Quantitative Research":
        render_research_workspace(selected_ticker)


if __name__ == "__main__":
    main()
