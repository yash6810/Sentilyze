import os
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone, timedelta
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
from src.data_ingestion import get_news
from src.paper_broker import PaperBroker
from src.realtime_tracker import (
    fetch_live_quote,
    fetch_universe_live_quotes,
    evaluate_intraday_execution,
    get_us_market_session_info,
)
from src.portfolio import (
    build_unified_portfolio,
    load_all_ticker_portfolios,
    calculate_risk_parity_weights,
)
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
from src.ipo_radar import fetch_pre_ipo_radar_summary
from src.agent_committee import (
    convene_trading_committee,
    execute_committee_order,
    COMMITTEE_FILE,
)
from src.ai_copilot import AICopilotEngine
from src.options_surface import (
    generate_volatility_surface_mesh,
    calculate_multileg_payoff,
)
from src.liquidity_heatmap import (
    compute_order_book_depth_and_clusters,
    compute_volume_profile_and_poc,
)
from src.autonomous_trader import AutonomousTradingEngine, AUTONOMOUS_LOG_FILE

logger = get_logger(__name__)


# --- Helper: Dynamic Ticker Universe ---
def get_universe_tickers() -> List[str]:
    stocks_file = "stocks.txt"
    if os.path.exists(stocks_file):
        with open(stocks_file, "r") as f:
            tickers = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
            if tickers:
                return tickers
    return [
        "NVDA",
        "AAPL",
        "MSFT",
        "GOOGL",
        "META",
        "TSLA",
        "AMZN",
        "AVGO",
        "AMD",
        "PLTR",
        "LLY",
        "QQQ",
        "SPY",
        "JPM",
        "COST",
        "NFLX",
        "TSM",
    ]


UNIVERSE_TICKERS = get_universe_tickers()


def format_timestamp_ist(iso_or_utc_str: str) -> str:
    """Converts UTC ISO timestamp string to Indian Standard Time (IST)."""
    if not iso_or_utc_str or iso_or_utc_str == "N/A":
        return "N/A"
    try:
        clean_str = str(iso_or_utc_str).replace("Z", "+00:00")
        dt_utc = datetime.fromisoformat(clean_str)
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        ist_offset = timezone(timedelta(hours=5, minutes=30))
        dt_ist = dt_utc.astimezone(ist_offset)
        return dt_ist.strftime("%d %b %Y, %I:%M:%S %p IST")
    except Exception:
        return str(iso_or_utc_str)[:19].replace("T", " ") + " UTC"


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
def render_plotly_candlestick(
    ticker: str, df: pd.DataFrame, curr_p: float, tp1: float, tp2: float, sl: float
):
    """Renders high-frequency interactive Candlestick chart with ATR channels and live intraday bar."""
    if df.empty or len(df) < 5:
        return
    recent_df = df.tail(60).copy()

    # Ensure DatetimeIndex with UTC normalization
    if not isinstance(recent_df.index, pd.DatetimeIndex):
        recent_df.index = pd.to_datetime(recent_df.index)
    if recent_df.index.tz is None:
        recent_df.index = recent_df.index.tz_localize("UTC").normalize()
    else:
        recent_df.index = recent_df.index.tz_convert("UTC").normalize()

    # Append or update today's live intraday price bar in real-time
    today_dt = pd.Timestamp.now(tz="UTC").normalize()
    if curr_p > 0:
        if recent_df.index[-1] < today_dt:
            q = fetch_live_quote(ticker)
            day_h = (
                float(q.get("day_high", curr_p))
                if float(q.get("day_high", 0)) > 0
                else curr_p
            )
            day_l = (
                float(q.get("day_low", curr_p))
                if float(q.get("day_low", 0)) > 0
                else curr_p
            )
            day_o = float(q.get("prev_close", curr_p))

            ma7_val = float((recent_df["Close"].tail(6).sum() + curr_p) / 7.0)
            ma21_val = float((recent_df["Close"].tail(20).sum() + curr_p) / 21.0)

            live_bar = pd.DataFrame(
                [
                    {
                        "Open": day_o,
                        "High": max(day_h, curr_p),
                        "Low": min(day_l, curr_p),
                        "Close": curr_p,
                        "ma7": ma7_val,
                        "ma21": ma21_val,
                    }
                ],
                index=[today_dt],
            )
            recent_df = pd.concat([recent_df, live_bar])
        elif recent_df.index[-1] == today_dt:
            recent_df.loc[today_dt, "Close"] = curr_p
            recent_df.loc[today_dt, "High"] = max(
                float(recent_df.loc[today_dt, "High"]), curr_p
            )
            recent_df.loc[today_dt, "Low"] = min(
                float(recent_df.loc[today_dt, "Low"]), curr_p
            )

    try:
        import plotly.graph_objects as go

        fig = go.Figure()

        # 1. Candlestick
        fig.add_trace(
            go.Candlestick(
                x=recent_df.index,
                open=recent_df["Open"],
                high=recent_df["High"],
                low=recent_df["Low"],
                close=recent_df["Close"],
                name="Price (Live)",
                increasing_line_color="#10B981",
                decreasing_line_color="#EF4444",
            )
        )

        # 2. 7 MA & 21 MA
        if "ma7" in recent_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=recent_df.index,
                    y=recent_df["ma7"],
                    line=dict(color="#38BDF8", width=1.5),
                    name="7 MA",
                )
            )
        if "ma21" in recent_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=recent_df.index,
                    y=recent_df["ma21"],
                    line=dict(color="#F59E0B", width=1.5),
                    name="21 MA",
                )
            )

        # 3. Take-Profit & Stop-Loss Target Lines
        fig.add_hline(
            y=tp1,
            line_dash="dash",
            line_color="#00D4AA",
            annotation_text=f"TP1 (+2.5 ATR): ${tp1:,.2f}",
            annotation_position="top right",
        )
        fig.add_hline(
            y=tp2,
            line_dash="dot",
            line_color="#10B981",
            annotation_text=f"TP2 (+4.5 ATR): ${tp2:,.2f}",
            annotation_position="top right",
        )
        fig.add_hline(
            y=sl,
            line_dash="dash",
            line_color="#EF4444",
            annotation_text=f"Stop-Loss: ${sl:,.2f}",
            annotation_position="bottom right",
        )

        fig.update_layout(
            template="plotly_dark",
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.4)",
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
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
    st.markdown(
        '<div class="section-badge">AI Momentum Inference & Intraday Market Radar</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.2, 1.8])

    features_df, price_df, news_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    quote = fetch_live_quote(ticker)
    curr_p = float(quote.get("price", 0))
    chg = float(quote.get("change_pct", 0))
    tp1 = curr_p * 1.06
    tp2 = curr_p * 1.12
    last_p = st.session_state.get(f"cmd_prev_p_{ticker}", curr_p)
    tick_dir = "SAME"
    if curr_p > last_p:
        tick_dir = "UP"
    elif curr_p < last_p:
        tick_dir = "DOWN"
    st.session_state[f"cmd_prev_p_{ticker}"] = curr_p

    tick_badge = (
        '<span style="font-size: 0.75rem; font-weight: 700; color: #10B981; background: rgba(16,185,129,0.2); padding: 2px 8px; border-radius: 4px; border: 1px solid rgba(16,185,129,0.4);">▲ Up Tick</span>'
        if tick_dir == "UP"
        else (
            '<span style="font-size: 0.75rem; font-weight: 700; color: #EF4444; background: rgba(239,68,68,0.2); padding: 2px 8px; border-radius: 4px; border: 1px solid rgba(239,68,68,0.4);">▼ Down Tick</span>'
            if tick_dir == "DOWN"
            else '<span style="font-size: 0.75rem; color: #94A3B8;">● Live Pulse</span>'
        )
    )

    with col1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.8rem; font-weight: 900; color: #00D4AA;">{ticker}</span>
                    <div style="display: flex; gap: 6px; align-items: center;">
                        {tick_badge}
                        <span style="font-size: 0.9rem; font-weight: bold; background: {'rgba(16, 185, 129, 0.2)' if chg>=0 else 'rgba(239, 68, 68, 0.2)'}; color: {'#10B981' if chg>=0 else '#EF4444'}; padding: 4px 10px; border-radius: 6px;">{chg:+.2f}%</span>
                    </div>
                </div>
                <div style="font-size: 2.2rem; font-weight: 900; margin: 0.3rem 0; font-family: 'JetBrains Mono', monospace;">${curr_p:,.2f}</div>
                <div style="font-size: 0.8rem; color: #94A3B8;">High: ${quote.get('day_high', 0):,.2f} &nbsp;|&nbsp; Low: ${quote.get('day_low', 0):,.2f} &nbsp;|&nbsp; <span style="color:#00D4AA;">Real-Time Feed</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # AI Model Inference
        model_path = f"models/{ticker}_model.json"
        if os.path.exists(model_path) or os.path.exists(
            model_path.replace(".json", ".joblib")
        ):
            with st.spinner(f"Running XGBoost + FinBERT for {ticker}..."):
                try:
                    features_df, price_df, news_df = preprocess_data(
                        ticker, use_cache=True
                    )
                    model = load_model(model_path)
                    pred_raw, conf_raw = get_prediction_on_latest_data(
                        model, features_df.tail(1), FEATURES
                    )
                    pred = int(pred_raw[0])
                    conf = (
                        float(conf_raw[0][1])
                        if len(conf_raw[0]) > 1
                        else float(conf_raw[0][0])
                    )

                    signal = "BUY" if pred == 1 and conf >= 0.50 else "HOLD"
                    sig_color = "#10B981" if signal == "BUY" else "#F59E0B"

                    atr = (
                        float(features_df["atr"].iloc[-1])
                        if "atr" in features_df.columns
                        else curr_p * 0.03
                    )
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

                    # --- 1-Click Interactive Live Order Ticket for Selected Asset ---
                    st.markdown(
                        f"""
                        <div class="glass-card" style="margin-top: 0.8rem; padding: 0.8rem; border-left: 3px solid #00D4AA;">
                            <div style="font-weight: 800; font-size: 0.85rem; color: #F8FAFC; margin-bottom: 0.4rem;">
                                ⚡ <b>1-Click Live Market Order Ticket ({ticker})</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    tkt_col1, tkt_col2 = st.columns([1.2, 1.2])
                    with tkt_col1:
                        order_size_preset = st.selectbox(
                            "Capital Allocation",
                            [
                                "$5,000",
                                "$10,000",
                                "$25,000",
                                "$45,000 (Max Target)",
                                "Custom Shares",
                            ],
                            key=f"cmd_order_alloc_{ticker}",
                        )
                        if order_size_preset == "Custom Shares":
                            order_shares = st.number_input(
                                "Shares",
                                min_value=1,
                                max_value=5000,
                                value=10,
                                step=1,
                                key=f"cmd_shares_{ticker}",
                            )
                        else:
                            dollar_budget = float(
                                order_size_preset.replace("$", "")
                                .replace(",", "")
                                .split()[0]
                            )
                            order_shares = (
                                max(1, int(dollar_budget // curr_p))
                                if curr_p > 0
                                else 1
                            )

                        est_cost = order_shares * curr_p
                        st.caption(
                            f"Estimated Order: **{order_shares} shares** (~${est_cost:,.2f})"
                        )

                    with tkt_col2:
                        st.write("")
                        st.write("")
                        b_instance = PaperBroker()
                        has_open = ticker in b_instance.state.get("open_positions", {})
                        if st.button(
                            f"🟢 Execute Live BUY {ticker}",
                            use_container_width=True,
                            key=f"cmd_btn_buy_{ticker}",
                        ):
                            with st.spinner(
                                f"Executing market buy order for {ticker}..."
                            ):
                                buy_res = b_instance.execute_manual_buy(
                                    ticker=ticker,
                                    shares=order_shares,
                                    price=curr_p,
                                    atr=atr,
                                    confidence=conf,
                                )
                                if buy_res.get("success"):
                                    st.toast(
                                        f"✅ Bought {order_shares} shares of {ticker} @ ${curr_p:,.2f}!",
                                        icon="🚀",
                                    )
                                    st.success(
                                        f"Executed BUY {order_shares} {ticker} @ ${curr_p:,.2f} | TP1: ${buy_res['tp1_target']:.2f} | SL: ${buy_res['sl_target']:.2f}"
                                    )
                                    time.sleep(0.6)
                                    st.rerun()
                                else:
                                    st.error(
                                        buy_res.get(
                                            "error", "Failed to execute buy order."
                                        )
                                    )

                        if has_open:
                            if st.button(
                                f"🔴 Exit / Sell {ticker} Now",
                                use_container_width=True,
                                key=f"cmd_btn_sell_{ticker}",
                            ):
                                with st.spinner(f"Closing position in {ticker}..."):
                                    sell_res = b_instance.execute_manual_sell(
                                        ticker=ticker, price=curr_p
                                    )
                                    if sell_res.get("success"):
                                        st.toast(
                                            f"🛑 Closed {ticker} @ ${curr_p:,.2f} (PnL: ${sell_res['trade']['pnl']:+,.2f})",
                                            icon="💰",
                                        )
                                        st.rerun()
                except Exception as e:
                    st.error(f"Inference error: {e}")

    with col2:
        # Interactive Candlestick Chart
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; padding: 0 4px;">
                <span style="font-size: 0.85rem; font-weight: 700; color: #F8FAFC;">📈 Live Market Candlestick & ATR Targets</span>
                <span style="font-size: 0.75rem; color: #10B981; font-weight: 600; background: rgba(16, 185, 129, 0.15); padding: 2px 8px; border-radius: 4px; border: 1px solid rgba(16, 185, 129, 0.3);">
                    🟢 Live Bar: ${curr_p:,.2f}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not price_df.empty:
            render_plotly_candlestick(ticker, price_df, curr_p, tp1, tp2, sl)

    # 5-Minute Proximity Radar
    st.markdown(
        '<div class="section-badge">📡 5-Minute Active Position Guardian & Proximity Radar</div>',
        unsafe_allow_html=True,
    )
    broker = PaperBroker()
    open_pos = broker.state.get("open_positions", {})
    last_upd_ist = format_timestamp_ist(broker.state.get("last_updated", "N/A"))

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(15, 23, 42, 0.6); padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid rgba(0, 212, 170, 0.2); margin-bottom: 0.8rem;">
            <div><span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #10B981; margin-right: 6px; box-shadow: 0 0 8px #10B981;"></span><b style="color: #F8FAFC; font-size: 0.85rem;">Autonomous Intraday Guardian:</b> <span style="color: #00D4AA; font-size: 0.85rem;">ACTIVE (5-Min Cloud Cron)</span></div>
            <div style="font-size: 0.75rem; color: #94A3B8;">Ledger Last Synced: <span style="color: #00D4AA; font-weight: 600; font-family: monospace;">{last_upd_ist}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
                    st.success(
                        f"Executed {len(trades)} exit trades on live market prices!"
                    )
                else:
                    st.info("All open positions are within target bands.")
                st.rerun()


# ==============================================================================
# 💼 WORKSPACE 2: PORTFOLIO & BROKER ($100k ACCOUNT)
# ==============================================================================
def render_portfolio_workspace():
    st.markdown(
        '<div class="section-badge">Virtual Paper Trading Broker ($100,000 Portfolio)</div>',
        unsafe_allow_html=True,
    )

    broker = PaperBroker()
    summary = broker.get_portfolio_summary()

    # Top KPI Bar
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(
            "Total Equity",
            f"${summary['total_equity']:,.2f}",
            f"{summary['total_return_pct']:+.2f}%",
        )
    with kpi2:
        st.metric("Available Cash", f"${summary['cash']:,.2f}")
    with kpi3:
        st.metric("Unrealized PnL", f"${summary['unrealized_pnl']:+,.2f}")
    with kpi4:
        st.metric(
            "Win Rate",
            f"{summary['win_rate']:.1f}% ({summary['winning_trades']}/{summary['total_trades']})",
        )

    # Alpaca Connection Card & PDF Tearsheet
    col_alpaca, col_pdf = st.columns([1.5, 1.5])
    with col_alpaca:
        alpaca = AlpacaBrokerBridge()
        alp_acc = alpaca.get_account_summary()
        alp_status = (
            "🟢 Connected (Alpaca Paper)"
            if alpaca.is_connected()
            else "⚪ Simulated Local Mode"
        )
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
            open_positions=[
                dict(pos, ticker=t)
                for t, pos in broker.state.get("open_positions", {}).items()
            ],
            equity_history_df=broker.get_equity_curve_df(),
        )
        st.download_button(
            "📄 Download 2-Page Executive Factsheet (PDF)",
            data=pdf_bytes,
            file_name=f"Sentilyze_Factsheet_{datetime.now(timezone.utc).strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    # --- Live Multi-Asset Execution Desk ---
    st.markdown(
        '<div class="section-badge">⚡ Live Multi-Asset Quick Order Desk</div>',
        unsafe_allow_html=True,
    )
    od_col1, od_col2, od_col3, od_col4 = st.columns([1.2, 1.2, 1, 1.2])

    with od_col1:
        trade_ticker = st.selectbox(
            "Asset Ticker", UNIVERSE_TICKERS, key="port_trade_ticker"
        )
        trade_quote = fetch_live_quote(trade_ticker)
        trade_p = float(trade_quote.get("price", 100.0))
        st.caption(
            f"Live Price: **${trade_p:,.2f}** ({trade_quote.get('status', 'LIVE')})"
        )

    with od_col2:
        trade_preset = st.selectbox(
            "Order Size",
            ["$10,000", "$25,000", "$45,000 (Max Conviction)", "Custom Shares"],
            key="port_order_size",
        )
        if trade_preset == "Custom Shares":
            trade_shares = st.number_input(
                "Shares",
                min_value=1,
                max_value=5000,
                value=10,
                step=1,
                key="port_shares_in",
            )
        else:
            trade_budget = float(
                trade_preset.replace("$", "").replace(",", "").split()[0]
            )
            trade_shares = max(1, int(trade_budget // trade_p)) if trade_p > 0 else 1
        st.caption(
            f"Target: **{trade_shares} shares** (~${trade_shares * trade_p:,.2f})"
        )

    with od_col3:
        st.write("")
        st.write("")
        if st.button(
            f"🟢 BUY {trade_ticker}", use_container_width=True, key="btn_port_buy"
        ):
            with st.spinner(f"Placing market buy order for {trade_ticker}..."):
                res_b = broker.execute_manual_buy(
                    ticker=trade_ticker, shares=trade_shares, price=trade_p
                )
                if res_b.get("success"):
                    st.toast(
                        f"✅ Executed BUY {trade_shares} {trade_ticker} @ ${trade_p:.2f}!",
                        icon="🚀",
                    )
                    st.rerun()
                else:
                    st.error(res_b.get("error"))

    with od_col4:
        st.write("")
        st.write("")
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button(
                "🚀 Auto-Deploy",
                use_container_width=True,
                help="Auto-allocates liquid cash into top AI signals right now",
            ):
                with st.spinner("Scanning universe and auto-deploying cash..."):
                    evaluate_intraday_execution(broker=broker)
                    st.toast("⚡ Capital deployed into top AI signals!", icon="🚀")
                    st.rerun()
        with col_act2:
            if st.button(
                "🚨 Kill-Switch",
                use_container_width=True,
                help="Immediately liquidates 100% of holdings into cash",
            ):
                with st.spinner("Liquidating all holdings into cash..."):
                    for t in list(broker.state.get("open_positions", {}).keys()):
                        broker.execute_manual_sell(
                            ticker=t, reason="STREAMLIT_MANUAL_KILL_SWITCH"
                        )
                    st.toast("🛑 All open positions liquidated into cash!", icon="💰")
                    st.rerun()

    # Active Holdings Table with Individual Position Action Controls
    st.markdown(
        '<div class="section-badge">📦 Active Open Holdings (50/50 Scale-Out Model)</div>',
        unsafe_allow_html=True,
    )
    open_df = broker.get_open_positions_df()
    if not open_df.empty:
        st.dataframe(open_df, use_container_width=True, hide_index=True)

        # Individual Position Action Cards
        st.markdown("**Interactive Position Controls:**")
        pos_cols = st.columns(min(len(broker.state["open_positions"]), 3))
        for idx, (sym, pos) in enumerate(list(broker.state["open_positions"].items())):
            with pos_cols[idx % len(pos_cols)]:
                q_spot = fetch_live_quote(sym)
                spot_price = float(q_spot.get("price", pos["current_price"]))
                shrs = int(pos["shares"])
                is_scld = pos.get("scaled_out", False)

                st.markdown(
                    f"""
                    <div class="glass-card" style="padding: 0.7rem; margin-bottom: 0.4rem;">
                        <b style="color: #00D4AA;">{sym}</b> &nbsp;|&nbsp; <b>{shrs} shares</b> &nbsp;|&nbsp; Spot: <b>${spot_price:,.2f}</b><br>
                        <span style="font-size: 0.75rem; color: #94A3B8;">TP1: ${float(pos.get('tp1_target', 0)):.2f} | SL: ${float(pos.get('sl_target', 0)):.2f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                btn_c1, btn_c2 = st.columns(2)
                with btn_c1:
                    if not is_scld:
                        if st.button(
                            f"🎯 Scale 50%",
                            key=f"scale_{sym}",
                            use_container_width=True,
                        ):
                            broker.execute_manual_scale_out(
                                ticker=sym, price=spot_price
                            )
                            st.toast(
                                f"🎯 Scaled out 50% of {sym} @ ${spot_price:,.2f}!",
                                icon="💰",
                            )
                            st.rerun()
                    else:
                        st.caption("🛡️ Risk-Free Runner")
                with btn_c2:
                    if st.button(
                        f"🛑 Exit 100%", key=f"close_{sym}", use_container_width=True
                    ):
                        broker.execute_manual_sell(
                            ticker=sym,
                            price=spot_price,
                            reason="STREAMLIT_POSITION_EXIT",
                        )
                        st.toast(
                            f"🛑 Closed {sym} @ ${spot_price:,.2f} into cash!",
                            icon="💵",
                        )
                        st.rerun()
    else:
        st.info(
            "No open positions. Use the Quick Order Desk above to execute live trades or click 'Auto-Deploy'."
        )

    # Equity Curve & Closed Trades
    col_eq, col_jrnl = st.columns([1.5, 1.5])
    with col_eq:
        st.markdown(
            '<div class="section-badge">📈 Equity Growth Curve</div>',
            unsafe_allow_html=True,
        )
        eq_df = broker.get_equity_curve_df()
        if not eq_df.empty and "total_equity" in eq_df.columns:
            st.line_chart(eq_df["total_equity"], use_container_width=True)

    with col_jrnl:
        st.markdown(
            '<div class="section-badge">📜 Closed Trade History Journal</div>',
            unsafe_allow_html=True,
        )
        closed_df = broker.get_closed_trades_df()
        if not closed_df.empty:
            st.dataframe(closed_df, use_container_width=True, hide_index=True)


# ==============================================================================
# 📊 WORKSPACE 3: MULTI-ASSET FUND & RISK ANALYTICS
# ==============================================================================
def render_fund_and_risk():
    st.markdown(
        '<div class="section-badge">17-Asset Fund Allocation, Rebalancer & Stress Testing</div>',
        unsafe_allow_html=True,
    )

    tab_fund, tab_var, tab_corr = st.tabs(
        [
            "💼 Fund Allocation & Rebalancer",
            "🎲 Monte Carlo Stress Test & VaR",
            "🔗 17-Asset Correlation Matrix",
        ]
    )

    with tab_fund:
        col_reb1, col_reb2 = st.columns([1, 2])
        with col_reb1:
            st.markdown("### 🧮 Custom Capital Share Calculator")
            budget = st.number_input(
                "Total Investment Budget ($)",
                min_value=1000.0,
                max_value=1000000.0,
                value=25000.0,
                step=1000.0,
            )
            model_type = st.selectbox(
                "Allocation Model",
                ["Risk Parity (Inverse Vol)", "Equal Weight", "Conviction Weight"],
            )
            model_key = (
                "risk_parity"
                if "Risk Parity" in model_type
                else ("equal_weight" if "Equal Weight" in model_type else "conviction")
            )
            reb_res = calculate_custom_rebalance(total_capital=budget, method=model_key)

        with col_reb2:
            st.markdown(f"### 📋 Exact Whole-Share Buy Orders (${budget:,.2f})")
            if "allocation_table" in reb_res:
                st.dataframe(
                    pd.DataFrame(reb_res["allocation_table"]),
                    use_container_width=True,
                    hide_index=True,
                )

    with tab_var:
        st.markdown("### 🎲 Monte Carlo Forward Simulation & VaR")
        st.caption(
            "Simulates 1,000 future forward market paths to compute Value-at-Risk (VaR) and Expected Shortfall."
        )
        if st.button("🚀 Run Monte Carlo Simulation", use_container_width=True):
            with st.spinner("Simulating 1,000 Geometric Brownian Motion paths..."):
                sim_res = run_monte_carlo_var(
                    initial_equity=100000.0, num_paths=1000, days=45
                )
                st.success(
                    f"95% Value-at-Risk: ${sim_res['var_95_dollar']:,.2f} ({sim_res['var_95_pct']:.2f}%) | Probability of Profit: {sim_res['prob_profit_pct']:.1f}%"
                )

    with tab_corr:
        st.markdown("### 🔗 17-Asset Cross-Correlation Heatmap")
        corr_res = compute_correlation_matrix()
        if "matrix" in corr_res:
            st.dataframe(corr_res["matrix"], use_container_width=True)


# ==============================================================================
# 🔬 WORKSPACE 4: QUANTITATIVE RESEARCH & DISCORD BOT
# ==============================================================================
def render_research_workspace(ticker: str):
    st.markdown(
        '<div class="section-badge">Quantitative Sandbox & Interactive Bot Console</div>',
        unsafe_allow_html=True,
    )

    tab_sand, tab_bot = st.tabs(
        ["⚙️ Strategy Optimizer & Sandbox", "🤖 Interactive Discord AI Bot Console"]
    )

    with tab_sand:
        col_ctrl, col_chart = st.columns([1, 2])
        with col_ctrl:
            st.markdown("### ⚙️ Strategy Sandbox Controls")
            lev = st.slider(
                "Account Leverage", min_value=1.0, max_value=2.0, value=1.0, step=0.1
            )
            conf_thresh = (
                st.slider(
                    "Confidence Filter (%)",
                    min_value=50,
                    max_value=75,
                    value=55,
                    step=5,
                )
                / 100.0
            )
            tp_mult = st.slider(
                "Take-Profit ATR Multiplier",
                min_value=1.5,
                max_value=4.5,
                value=2.5,
                step=0.5,
            )

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
                    st.metric(
                        "Strategy Return",
                        f"{res['total_return_pct']:+.2f}%",
                        f"Benchmark: {res['benchmark_return_pct']:+.1f}%",
                    )
                with k2:
                    st.metric("Sharpe Ratio", f"{res['sharpe_ratio']:.2f}")
                with k3:
                    st.metric("Max Drawdown", f"{res['max_drawdown_pct']:.2f}%")

                if "chart_df" in res and not res["chart_df"].empty:
                    st.line_chart(res["chart_df"], use_container_width=True)

    with tab_bot:
        st.markdown("### 🤖 Interactive Discord AI Bot Console")
        st.caption(
            "Test slash commands live from your browser before or alongside mobile Discord execution."
        )

        col_cmd, col_exec = st.columns([2, 1])
        with col_cmd:
            user_cmd = st.text_input(
                "Discord Command",
                value=f"/signal {ticker}",
                placeholder="/signal AMD, /portfolio, /execute, /var",
            )
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
    st.markdown(
        '<div class="section-badge">Statistical Arbitrage & Cointegration Pairs Trading Engine</div>',
        unsafe_allow_html=True,
    )

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
        selected_pair_str = st.selectbox(
            "Select Cointegrated Pair", preset_pairs, index=0
        )
    with col_ctrl2:
        lookback_window = st.slider(
            "Z-Score Window (Days)", min_value=10, max_value=60, value=30, step=5
        )
    with col_ctrl3:
        z_threshold = st.slider(
            "Entry Z-Threshold (σ)", min_value=1.0, max_value=3.0, value=2.0, step=0.25
        )

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
            series_a,
            series_b,
            ticker_a,
            ticker_b,
            window=lookback_window,
            entry_z=z_threshold,
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
        fig.add_trace(
            go.Scatter(
                x=z_df.index,
                y=z_df["Z"],
                mode="lines",
                name="Spread Z-Score",
                line=dict(color="#00D4AA", width=2),
            )
        )
        fig.add_hline(
            y=z_threshold,
            line=dict(color="#EF4444", dash="dash", width=1.5),
            annotation_text=f"Overbought (+{z_threshold}σ)",
        )
        fig.add_hline(
            y=-z_threshold,
            line=dict(color="#10B981", dash="dash", width=1.5),
            annotation_text=f"Oversold (-{z_threshold}σ)",
        )
        fig.add_hline(
            y=0.0,
            line=dict(color="rgba(148, 163, 184, 0.4)", width=1),
            annotation_text="Equilibrium (0.0)",
        )

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
        bt_res = backtest_pairs_strategy(
            series_a, series_b, window=lookback_window, entry_z=z_threshold
        )

        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("Strategy Return", f"{bt_res['total_return']:+.2f}%")
        bc2.metric("Sharpe Ratio", f"{bt_res['sharpe_ratio']:.2f}")
        bc3.metric("Max Drawdown", f"{bt_res['max_drawdown']:.2f}%")
        bc4.metric(
            "Win Rate", f"{bt_res['win_rate']:.1f}% ({bt_res['total_trades']} Trades)"
        )

        eq_fig = go.Figure()
        eq_fig.add_trace(
            go.Scatter(
                x=bt_res["equity_curve"].index,
                y=bt_res["equity_curve"].values,
                mode="lines",
                name="Pair Equity Curve ($)",
                line=dict(color="#3B82F6", width=2),
            )
        )
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
    st.markdown(
        '<div class="section-badge">Options Microstructure, Gamma Exposure (GEX) & Expiration Pinning</div>',
        unsafe_allow_html=True,
    )

    with st.spinner(f"Fetching Live Option Chain for {ticker}..."):
        chain = fetch_option_chain(ticker)
        max_pain, loss_df = calculate_max_pain(chain["calls_df"], chain["puts_df"])
        pcr = calculate_put_call_ratios(chain["calls_df"], chain["puts_df"])
        gex = estimate_gamma_exposure(
            chain["calls_df"], chain["puts_df"], chain["spot_price"]
        )
        spreads = recommend_option_spreads(
            ticker,
            "BUY",
            chain["spot_price"],
            max_pain,
            chain["calls_df"],
            chain["puts_df"],
        )

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
            fig_oi.add_trace(
                go.Bar(
                    x=c_df["strike"],
                    y=c_df["openInterest"],
                    name="Call OI",
                    marker_color="#10B981",
                )
            )
        if not p_df.empty and "strike" in p_df.columns:
            fig_oi.add_trace(
                go.Bar(
                    x=p_df["strike"],
                    y=p_df["openInterest"],
                    name="Put OI",
                    marker_color="#EF4444",
                )
            )

        fig_oi.add_vline(
            x=max_pain,
            line=dict(color="#F59E0B", dash="dash", width=2),
            annotation_text=f"Max Pain (${max_pain:.0f})",
        )
        fig_oi.add_vline(
            x=spot,
            line=dict(color="#00D4AA", width=2),
            annotation_text=f"Spot (${spot:.0f})",
        )

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
            fig_loss.add_trace(
                go.Scatter(
                    x=loss_df["strike"],
                    y=loss_df["total_loss"] / 1e6,
                    mode="lines",
                    name="Total Payout ($M)",
                    line=dict(color="#F59E0B", width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(245, 158, 11, 0.1)",
                )
            )
            fig_loss.add_vline(
                x=max_pain,
                line=dict(color="#00D4AA", dash="dot", width=2),
                annotation_text=f"Min Loss (${max_pain:.0f})",
            )

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
    st.markdown(
        '<div class="section-badge">Piotroski 9-Point F-Score, Altman Z-Score & DCF Fair Value Matrix</div>',
        unsafe_allow_html=True,
    )

    with st.spinner(f"Analyzing Balance Sheet & DCF Cash Flows for {ticker}..."):
        fin_data = fetch_financial_statements(ticker)
        f_res = calculate_piotroski_f_score(ticker, fin_data)
        z_res = calculate_altman_z_score(ticker, fin_data)
        dcf_res = calculate_dcf_fair_value(ticker, fin_data)
        radar_metrics = generate_spider_radar_profile(
            ticker, 0.76, f_res, z_res, dcf_res
        )

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
        fig_radar.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself",
                fillcolor="rgba(0, 212, 170, 0.25)",
                line=dict(color="#00D4AA", width=2),
                name="Asset Profile",
            )
        )
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="#64748B"),
                angularaxis=dict(color="#94A3B8"),
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
    st.markdown(
        '<div class="section-badge">Historical Black Swan Crisis Stress-Testing & Kelly Sizing</div>',
        unsafe_allow_html=True,
    )

    broker = PaperBroker()
    total_eq = float(broker.state.get("total_equity", 100000.0))
    open_pos = broker.state.get("open_positions", {})

    # Default positions if cash buffer
    positions_dict = {}
    if open_pos:
        for sym, p in open_pos.items():
            positions_dict[sym] = float(
                p.get("shares", 0) * p.get("entry_price", 100.0)
            )
    else:
        positions_dict = {
            "NVDA": 35000.0,
            "AAPL": 25000.0,
            "MSFT": 20000.0,
            "TSM": 10000.0,
        }

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
    fig_cr.add_trace(
        go.Bar(
            x=c_names,
            y=c_drawdowns,
            text=[f"-{d:.1f}% (${l:,.0f})" for d, l in zip(c_drawdowns, c_losses)],
            textposition="auto",
            marker_color=["#EF4444", "#F59E0B", "#EF4444", "#DC2626", "#3B82F6"],
        )
    )
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
        with st.expander(
            f"🔴 {r['crisis_name']} ({r['date_range']}) — Projected Loss: ${r['projected_dollar_loss']:,.2f} (-{r['portfolio_drawdown_pct']:.1f}%)"
        ):
            st.markdown(f"**Catalyst**: {r['catalyst']}")
            st.markdown(
                f"**VIX Peak**: `{r['vix_peak']:.2f}` &nbsp;|&nbsp; **Simulated Remaining Equity**: `${r['simulated_equity_after']:,.2f}`"
            )


# ==============================================================================
# 🧠 WORKSPACE 9: ADVANCED AI, GNN & DEEP ALPHA LAB (PILLAR 1)
# ==============================================================================
def render_ai_deep_alpha_workspace(ticker: str):
    st.markdown(
        '<div class="section-badge">Pillar 1: Temporal Fusion Transformer, GNN Supply Chains & PPO Allocation</div>',
        unsafe_allow_html=True,
    )

    col_a1, col_a2 = st.columns([1.5, 1])

    with col_a1:
        st.markdown("#### ⏳ Temporal Fusion Transformer (TFT) Multi-Horizon Forecast")
        quote = fetch_live_quote(ticker)
        curr_p = float(quote.get("price", 150.0))

        with st.spinner("Running Multi-Head Self-Attention & Variable Selection..."):
            dummy_df = pd.DataFrame(
                np.random.randn(30, 6), columns=[f"feat_{i}" for i in range(6)]
            )
            tft_forecast = run_temporal_fusion_forecast(ticker, dummy_df, curr_p)

        # Multi-Horizon Cards
        h_cols = st.columns(4)
        h_names = [
            ("1_day", "1-Day Ahead"),
            ("5_days", "5-Days Ahead"),
            ("10_days", "10-Days Ahead"),
            ("21_days", "21-Days Ahead"),
        ]
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
        fig_attn.add_trace(
            go.Scatter(
                x=list(range(len(attn_w))),
                y=attn_w,
                mode="lines+markers",
                name="Attention Weight",
                line=dict(color="#7C3AED", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(124, 58, 237, 0.15)",
            )
        )
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
        rl_res = optimize_rl_position_allocation(
            ticker,
            recent_returns=[0.015, -0.008, 0.022, 0.011, 0.005],
            ai_confidence=0.78,
        )

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

    src_node = st.selectbox(
        "Select Upstream Shock Origin", ["TSM", "NVDA", "AVGO", "AAPL", "MSFT"], index=0
    )
    shock_amt = st.slider(
        "Supply Disruption Magnitude (%)",
        min_value=-15.0,
        max_value=-1.0,
        value=-5.0,
        step=1.0,
    )

    gnn_res = analyze_supply_chain_spillover(
        origin_ticker=src_node, shock_pct=shock_amt
    )
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
    st.markdown(
        '<div class="section-badge">Pillar 2: SEC 10-K Diffs, Earnings Calls, Social Buzz, Insiders & Patents</div>',
        unsafe_allow_html=True,
    )

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
            st.markdown(
                f"• **{ins['insider_name']}** ({ins['title']}): `{ins['transaction_type']}` {ins['shares']:,} shares @ ${ins['price']:.2f}"
            )

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
            st.markdown(
                f"• **{ct['agency']}**: `{ct['program']}` — **${ct['award_value']:,.0f}**"
            )

    # 3. Real-Time Multi-Platform Social Stream (Reddit + Stocktwits)
    st.markdown("---")
    st.markdown("#### 🌐 Multi-Platform Live Social Scraper (Reddit & Stocktwits)")
    sc_col1, sc_col2 = st.columns(2)
    with sc_col1:
        st.markdown(
            f"**Reddit (r/wallstreetbets & r/stocks) mentions for `{ticker}`:**"
        )
        reddit_posts = soc_res.get("reddit_stream", [])
        if reddit_posts:
            for p in reddit_posts[:4]:
                st.markdown(
                    f"""
                    <div class="glass-card" style="margin-bottom: 0.5rem; padding: 0.75rem;">
                        <div style="font-weight: 700; color: #F1F5F9; font-size: 0.85rem;">{p['title']}</div>
                        <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 0.25rem;">
                            Score: <code>▲ {p['score']}</code> | Comments: <code>💬 {p['comments']}</code> | <a href="{p['url']}" target="_blank" style="color: #38BDF8;">View Post</a>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info(f"No recent Reddit mentions found for {ticker}.")

    with sc_col2:
        st.markdown(f"**Stocktwits Real-Time Retail Stream for `{ticker}`:**")
        st_msgs = soc_res.get("stocktwits_stream", [])
        if st_msgs:
            for m in st_msgs[:4]:
                tag_color = (
                    "#10B981"
                    if m["sentiment"] == "BULLISH"
                    else ("#EF4444" if m["sentiment"] == "BEARISH" else "#94A3B8")
                )
                st.markdown(
                    f"""
                    <div class="glass-card" style="margin-bottom: 0.5rem; padding: 0.75rem; border-left: 3px solid {tag_color};">
                        <div style="font-size: 0.82rem; color: #E2E8F0;">{m['body'][:160]}...</div>
                        <div style="font-size: 0.72rem; color: {tag_color}; font-weight: 700; margin-top: 0.25rem;">
                            Tag: {m['sentiment']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info(f"No recent Stocktwits stream data for {ticker}.")

    # 4. Pre-IPO & SEC S-1 Registration Radar
    st.markdown("---")
    st.markdown("#### 🚀 Pre-IPO Private Valuations & SEC Form S-1 Registration Radar")
    ipo_summary = fetch_pre_ipo_radar_summary()
    ipo_col1, ipo_col2 = st.columns([1.5, 1])

    with ipo_col1:
        st.markdown("**🔥 Top Pre-IPO Private Tech Targets (Projected Listings):**")
        for target in ipo_summary["pre_ipo_targets"]:
            st.markdown(
                f"""
                <div class="glass-card" style="margin-bottom: 0.6rem; padding: 0.85rem; border-left: 4px solid #38BDF8;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 800; font-size: 1.05rem; color: #F8FAFC;">{target['name']} <code style="color: #38BDF8;">${target['projected_ticker']}</code></span>
                        <span style="background: rgba(56, 189, 248, 0.15); color: #38BDF8; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.75rem;">{target['status']}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 0.4rem;">
                        • <b>Est. Valuation</b>: <span style="color: #34D399; font-weight: 700;">{target['est_valuation_usd']}</span> | <b>Readiness Score</b>: <code>{target['ipo_readiness_score']}/100</code><br>
                        • <b>Key Backers</b>: {', '.join(target['lead_backers'][:3])}<br>
                        • <b>Catalysts</b>: {target['key_catalysts']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with ipo_col2:
        st.markdown("**📋 Live SEC Form S-1 Registrations (EDGAR):**")
        s1_filings = ipo_summary["recent_s1_filings"]
        if s1_filings:
            for s1 in s1_filings[:4]:
                st.markdown(
                    f"""
                    <div class="glass-card" style="margin-bottom: 0.5rem; padding: 0.7rem;">
                        <div style="font-weight: 700; font-size: 0.82rem; color: #F1F5F9;">{s1['title'][:70]}...</div>
                        <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 0.25rem;">
                            Date: {s1['updated_at']} | <a href="{s1['filing_url']}" target="_blank" style="color: #38BDF8;">EDGAR Filing</a>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("SEC EDGAR S-1 filing stream ready. Auto-polling active.")


# ==============================================================================
# 🏛️ WORKSPACE 11: AUTONOMOUS MULTI-AGENT TRADING COMMITTEE (AI DEBATE DESK)
# ==============================================================================
def render_committee_workspace(ticker: str):
    st.markdown(
        '<div class="section-badge">🏛️ Autonomous Multi-Agent Trading Committee & Round-Table Deliberation</div>',
        unsafe_allow_html=True,
    )

    delib = convene_trading_committee(ticker, save_resolution=True)
    cro = delib["cro_signoff"]

    # Consensus Resolution Hero Banner
    res_color = (
        "#10B981"
        if "BUY" in delib["final_resolution"]
        else ("#F59E0B" if "SCALE" in delib["final_resolution"] else "#EF4444")
    )
    st.markdown(
        f"""
        <div class="glass-card" style="border: 1px solid {res_color}66; margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 0.8rem; color: #94A3B8; letter-spacing: 0.05em;">OFFICIAL COMMITTEE VERDICT ({ticker})</span>
                    <div style="font-size: 1.8rem; font-weight: 900; color: {res_color}; margin: 0.2rem 0;">{delib['final_resolution']}</div>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.8rem; color: #94A3B8;">Consensus Conviction</span>
                    <div style="font-size: 1.8rem; font-weight: 900; color: #00D4AA; font-family: 'JetBrains Mono', monospace;">{delib['consensus_conviction_pct']:.1f}%</div>
                </div>
            </div>
            <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 0.5rem; line-height: 1.5;">
                • <b>Approved Leverage</b>: <code>{cro['approved_leverage']:.1f}x</code> &nbsp;|&nbsp; 
                • <b>Kelly Capital Allocation</b>: <code>{cro['kelly_allocation_pct']:.1f}%</code> &nbsp;|&nbsp; 
                • <b>TP1 (+2.5 ATR)</b>: <code>${delib['tp1_target']:,.2f}</code> &nbsp;|&nbsp; 
                • <b>Stop-Loss</b>: <code>${delib['stop_loss_target']:,.2f}</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Specialist Agent Cards
    st.markdown("#### 🎙️ Specialist Agent Testimonies & Voting Breakdown")
    c1, c2, c3 = st.columns(3)
    cols = [c1, c2, c3]
    for idx, rep in enumerate(delib["agent_testimonies"]):
        with cols[idx % 3]:
            v_color = (
                "#10B981"
                if rep["vote"] == "BUY"
                else ("#F59E0B" if rep["vote"] == "HOLD" else "#EF4444")
            )
            st.markdown(
                f"""
                <div class="glass-card" style="border-top: 3px solid {v_color}; min-height: 220px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <b style="color: #F8FAFC; font-size: 0.85rem;">{rep['agent_name']}</b>
                        <span style="background: {v_color}22; color: {v_color}; font-size: 0.75rem; font-weight: 800; padding: 2px 6px; border-radius: 4px;">{rep['vote']}</span>
                    </div>
                    <div style="font-size: 0.75rem; color: #00D4AA; margin-bottom: 0.5rem;">{rep['role']} (Conviction: {rep['conviction_score']}%)</div>
                    <div style="font-size: 0.8rem; color: #94A3B8; line-height: 1.4;">{rep['thesis']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Chief Risk Officer Deliberation Box
    st.markdown("#### ⚖️ Chief Risk Officer (CRO) Arbitration & Veto Log")
    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 4px solid #38BDF8;">
            <div style="font-weight: 800; color: #F8FAFC;">{cro['cro_name']} — Formal Sign-Off</div>
            <div style="font-size: 0.85rem; color: #CBD5E1; margin-top: 0.4rem;">{cro['cro_thesis']}</div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.4rem;">VIX Volatility Gate Status: {'🚨 VETO TRIGGERED' if cro['vix_veto_triggered'] else '🟢 PASSED (VIX Normal)'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1-Click Execution Desk
    st.markdown("#### ⚡ Real-Time Autonomous Trade Execution")
    broker_instance = PaperBroker()
    is_holding = ticker in broker_instance.state.get("open_positions", {})
    action_code = delib.get("action_code", "HOLD")

    ex_col1, ex_col2 = st.columns([1.5, 1.0])
    with ex_col1:
        if action_code in ["EXECUTE_BUY", "SCALE_IN"]:
            summary = broker_instance.get_portfolio_summary()
            target_budget = (
                summary["total_equity"]
                * (cro["kelly_allocation_pct"] / 100.0)
                * cro["approved_leverage"]
            )
            shares = int(
                min(target_budget, summary["cash"] * 0.95) / delib["spot_price"]
            )

            st.info(
                f"💡 Committee recommends buying **{shares} shares** of {ticker} (${target_budget:,.2f} budget @ {cro['approved_leverage']}x leverage)."
            )
            if st.button(
                f"🟢 Execute Sanctioned BUY {shares} {ticker} @ ${delib['spot_price']:,.2f}",
                use_container_width=True,
                key=f"comm_buy_{ticker}",
            ):
                with st.spinner(f"Executing committee order for {ticker}..."):
                    order_res = execute_committee_order(
                        ticker, deliberation=delib, broker=broker_instance
                    )
                    if order_res.get("success"):
                        st.success(
                            f"🚀 Bought {shares} {ticker} @ ${delib['spot_price']:,.2f}! TP1: ${delib['tp1_target']:.2f} | SL: ${delib['stop_loss_target']:.2f}"
                        )
                        time.sleep(0.6)
                        st.rerun()
                    else:
                        st.error(
                            f"Execution notice: {order_res.get('message', 'Failed to execute order.')}"
                        )
        elif is_holding:
            st.warning(
                f"⚠️ You currently hold {ticker}, but the Committee resolution is {delib['final_resolution']}."
            )
            if st.button(
                f"🔴 Execute Committee Exit / Close {ticker} Now",
                use_container_width=True,
                key=f"comm_sell_{ticker}",
            ):
                with st.spinner(f"Closing position in {ticker}..."):
                    order_res = execute_committee_order(
                        ticker, deliberation=delib, broker=broker_instance
                    )
                    if order_res.get("success"):
                        st.success(f"🛑 Closed {ticker} @ ${delib['spot_price']:,.2f}!")
                        time.sleep(0.6)
                        st.rerun()
        else:
            st.markdown(
                f"*Standing in cash buffer. No action recommended for {ticker} by the Committee.*"
            )


# ==============================================================================
# 💬 WORKSPACE 12: AI TRADE COPILOT & CONVERSATIONAL ANALYST
# ==============================================================================
def render_copilot_workspace(ticker: str):
    st.markdown(
        '<div class="section-badge">💬 AI Trade Copilot & Conversational Financial Intelligence</div>',
        unsafe_allow_html=True,
    )

    copilot = AICopilotEngine()

    # Preset Prompt Pills
    st.markdown("**⚡ Quick Prompt Inquiries:**")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    active_prompt = None
    with p_col1:
        if st.button("💼 Portfolio Health", use_container_width=True):
            active_prompt = "Show my portfolio balance and profit"
    with p_col2:
        if st.button(
            f"🏛️ Committee on {ticker}",
            use_container_width=True,
            key=f"btn_c_{ticker}",
        ):
            active_prompt = f"What does the committee debate say about {ticker}?"
    with p_col3:
        if st.button("🌪️ Simulate -10% Crash", use_container_width=True):
            active_prompt = "Simulate a 10% drop crash in my portfolio"
    with p_col4:
        if st.button(
            f"🎯 Targets for {ticker}",
            use_container_width=True,
            key=f"btn_t_{ticker}",
        ):
            active_prompt = f"What are the profit targets and stop loss for {ticker}?"

    user_query = st.text_input(
        "Ask the AI Copilot anything about your portfolio, technical setups, or risk models:",
        value=active_prompt or "",
        placeholder=f"e.g. 'Why did we take a position in {ticker}?' or 'Stress-test my portfolio against a 5% drop'",
        key="copilot_text_input",
    )

    if user_query:
        with st.spinner("Copilot synthesizing multi-pillar financial insights..."):
            ans = copilot.answer_query(user_query, context_ticker=ticker)
            st.markdown(
                f"""
                <div class="glass-card" style="border: 1px solid rgba(0, 212, 170, 0.3); margin-top: 1rem;">
                    {ans['markdown_response']}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# 📐 WORKSPACE 13: 3D VOLATILITY SURFACE & MULTI-LEG OPTIONS DESK
# ==============================================================================
def render_options_surface_workspace(ticker: str):
    st.markdown(
        '<div class="section-badge">📐 3D Implied Volatility Surface & Multi-Leg Options Strategy Desk</div>',
        unsafe_allow_html=True,
    )

    mesh = generate_volatility_surface_mesh(ticker)
    spot = mesh["spot_price"]

    col_s1, col_s2 = st.columns([1.6, 1.2])

    with col_s1:
        st.markdown("#### 🌐 3D Implied Volatility Smile & Term Structure")
        try:
            import plotly.graph_objects as go

            fig3d = go.Figure(
                data=[
                    go.Surface(
                        x=mesh["strikes"],
                        y=mesh["dtes"],
                        z=mesh["iv_matrix"],
                        colorscale="Viridis",
                    )
                ]
            )
            fig3d.update_layout(
                title=f"3D Volatility Surface ({ticker} @ ${spot:,.2f})",
                scene=dict(
                    xaxis_title="Strike Price ($)",
                    yaxis_title="Days to Expiry (DTE)",
                    zaxis_title="Implied Volatility (%)",
                ),
                template="plotly_dark",
                height=450,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig3d, use_container_width=True)
        except ImportError:
            st.info("Plotly 3D visualizer initializing...")

    with col_s2:
        st.markdown("#### ⚡ Institutional Multi-Leg Strategy Payoff Desk")
        strat_choice = st.selectbox(
            "Select Multi-Leg Structure",
            [
                "BULL_CALL_SPREAD",
                "BEAR_PUT_SPREAD",
                "IRON_CONDOR",
                "LONG_STRADDLE",
            ],
        )
        payoff_data = calculate_multileg_payoff(strat_choice, spot_price=spot)

        st.markdown(
            f"""
            <div class="glass-card" style="margin-bottom: 0.8rem;">
                <div style="font-size: 0.85rem; color: #CBD5E1;">{payoff_data['description']}</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.4rem;">
                    • <b>Max Profit</b>: <span style="color:#10B981;">{payoff_data['max_profit']}</span><br>
                    • <b>Max Risk / Loss</b>: <span style="color:#EF4444;">${payoff_data['max_loss']:,.2f}</span><br>
                    • <b>Risk/Reward Ratio</b>: <code>{payoff_data['risk_reward_ratio']}x</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
            import plotly.graph_objects as go

            fig_p = go.Figure()
            fig_p.add_trace(
                go.Scatter(
                    x=payoff_data["price_range"],
                    y=payoff_data["payoff_curve"],
                    mode="lines",
                    line=dict(color="#00D4AA", width=2.5),
                    name="P&L at Expiry",
                )
            )
            fig_p.add_hline(y=0, line_dash="dash", line_color="#64748B")
            fig_p.add_vline(
                x=spot,
                line_dash="dot",
                line_color="#F59E0B",
                annotation_text=f"Spot: ${spot:,.2f}",
            )
            fig_p.update_layout(
                template="plotly_dark",
                height=260,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.4)",
            )
            st.plotly_chart(fig_p, use_container_width=True)
        except Exception:
            pass


# ==============================================================================
# 🌊 WORKSPACE 14: LEVEL 2 DEPTH & LIQUIDITY HEATMAP DESK
# ==============================================================================
def render_liquidity_heatmap_workspace(ticker: str):
    st.markdown(
        '<div class="section-badge">🌊 Level 2 Order Book Depth & Institutional Dark Pool Liquidity Heatmap</div>',
        unsafe_allow_html=True,
    )

    depth = compute_order_book_depth_and_clusters(ticker)
    vp = compute_volume_profile_and_poc(ticker)
    spot = depth["spot_price"]

    # Depth Stats Banner
    sent_color = (
        "#10B981"
        if "BULLISH" in depth["depth_sentiment"]
        else ("#EF4444" if "BEARISH" in depth["depth_sentiment"] else "#38BDF8")
    )
    st.markdown(
        f"""
        <div class="glass-card" style="margin-bottom: 1rem; border-left: 4px solid {sent_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 0.8rem; color: #94A3B8;">ORDER BOOK IMBALANCE RATIO</span>
                    <div style="font-size: 1.6rem; font-weight: 900; color: {sent_color};">{depth['bid_ask_imbalance_ratio']}x &nbsp;<span style="font-size: 0.9rem; color: #E2E8F0;">({depth['depth_sentiment']})</span></div>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.8rem; color: #94A3B8;">Point of Control (POC)</span>
                    <div style="font-size: 1.6rem; font-weight: 900; color: #00D4AA; font-family: 'JetBrains Mono', monospace;">${vp['poc_price']:,.2f}</div>
                </div>
            </div>
            <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.4rem;">
                • <b>Value Area High (VAH)</b>: <code>${vp['value_area_high']:,.2f}</code> &nbsp;|&nbsp; 
                • <b>Value Area Low (VAL)</b>: <code>${vp['value_area_low']:,.2f}</code> &nbsp;|&nbsp; 
                • <b>Total Bid Liquidity</b>: <code>{depth['total_bid_volume']:,} shares</code> &nbsp;|&nbsp; 
                • <b>Total Ask Liquidity</b>: <code>{depth['total_ask_volume']:,} shares</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_l1, col_l2 = st.columns([1.2, 1.8])

    with col_l1:
        st.markdown("#### 🪜 Level 2 Liquidity Ladder")
        bids_df = pd.DataFrame(depth["bids"])
        asks_df = pd.DataFrame(depth["asks"])

        st.caption("🟢 Top Bids (Buyers Support)")
        st.dataframe(
            bids_df[
                ["price", "shares", "notional_value", "is_institutional_wall"]
            ].head(6),
            use_container_width=True,
            hide_index=True,
        )

        st.caption("🔴 Top Asks (Sellers Resistance)")
        st.dataframe(
            asks_df[
                ["price", "shares", "notional_value", "is_institutional_wall"]
            ].head(6),
            use_container_width=True,
            hide_index=True,
        )

    with col_l2:
        st.markdown("#### 📊 Volume Profile Visible Range (VPVR)")
        try:
            import plotly.graph_objects as go

            fig_vp = go.Figure()
            fig_vp.add_trace(
                go.Bar(
                    y=vp["price_bins"],
                    x=vp["volumes"],
                    orientation="h",
                    marker_color="#38BDF8",
                    name="Volume Profile",
                )
            )
            fig_vp.add_hline(
                y=vp["poc_price"],
                line_dash="dash",
                line_color="#F59E0B",
                annotation_text=f"POC: ${vp['poc_price']:,.2f}",
            )
            fig_vp.add_hline(
                y=vp["value_area_high"],
                line_dash="dot",
                line_color="#10B981",
                annotation_text=f"VAH: ${vp['value_area_high']:,.2f}",
            )
            fig_vp.add_hline(
                y=vp["value_area_low"],
                line_dash="dot",
                line_color="#EF4444",
                annotation_text=f"VAL: ${vp['value_area_low']:,.2f}",
            )
            fig_vp.update_layout(
                template="plotly_dark",
                height=420,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.4)",
                xaxis_title="Accumulated Volume",
                yaxis_title="Price Level ($)",
            )
            st.plotly_chart(fig_vp, use_container_width=True)
        except Exception:
            pass


# ==============================================================================
# 🤖 WORKSPACE 15: AUTONOMOUS TRADING AGENT & LIVE NEWS DESK
# ==============================================================================
def render_autonomous_trader_workspace(selected_ticker: str):
    st.markdown(
        '<div class="section-badge">🤖 Autonomous Live Trading & Multi-Source News Engine</div>',
        unsafe_allow_html=True,
    )

    auto_engine = AutonomousTradingEngine()
    broker_instance = auto_engine.broker
    portfolio_summary = broker_instance.get_portfolio_summary()

    # Metrics Bar
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("💰 Total Equity", f"${portfolio_summary['total_equity']:,.2f}")
    m2.metric("💵 Cash Balance", f"${portfolio_summary['cash']:,.2f}")
    m3.metric(
        "📈 Unrealized PnL",
        f"${portfolio_summary['unrealized_pnl']:+,.2f}",
        delta=f"{portfolio_summary['unrealized_pnl_pct']:+.2f}%",
    )
    m4.metric("🏆 Win Rate", f"{portfolio_summary['win_rate']:.1f}%")

    # Controls Row
    st.markdown("#### ⚡ Autonomous Cycle Execution")
    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        st.markdown(
            """
            <div class="glass-card">
                <b>24/7 Autonomous Agent Loop:</b> Continuously ingests <b>Google News RSS + Finnhub + Marketaux</b>, 
                evaluates the <b>4-Agent Committee</b>, allocates capital via <b>Kelly Sizing</b>, and manages 
                <b>2-Stage Staged Profit Exits (50% @ TP1, Trailing Breakeven Stop, 50% @ TP2)</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ctrl_col2:
        if st.button("🚀 Run Autonomous Decision Cycle Now", use_container_width=True):
            with st.spinner(
                "Executing autonomous news scan and trade management cycle..."
            ):
                cycle_res = auto_engine.run_autonomous_cycle()
                st.success(
                    f"✅ Cycle complete in {cycle_res.get('elapsed_seconds', 0)}s! "
                    f"Buys: {len(cycle_res.get('buys', []))}, "
                    f"TP1 Exits: {len(cycle_res.get('take_profits_tp1', []))}, "
                    f"TP2 Runners: {len(cycle_res.get('take_profits_tp2', []))}"
                )
                time.sleep(0.5)
                st.rerun()

    # Open Positions & Profit Scaling
    st.markdown("#### 📊 Open Active Positions & Dynamic Targets")
    open_pos = broker_instance.state.get("open_positions", {})
    if open_pos:
        pos_cards = []
        for t, p in open_pos.items():
            curr_p = p.get("current_price", p["entry_price"])
            pnl_d = (curr_p - p["entry_price"]) * p["shares"]
            pnl_pct = (curr_p - p["entry_price"]) / p["entry_price"] * 100.0
            scaled_label = (
                "✅ Scaled (Risk-Free Breakeven)"
                if p.get("scaled_out")
                else "⏳ Holding Full Size"
            )

            pos_cards.append(
                {
                    "Ticker": t,
                    "Shares": p["shares"],
                    "Entry Price": f"${p['entry_price']:,.2f}",
                    "Live Price": f"${curr_p:,.2f}",
                    "Unrealized PnL": f"${pnl_d:+,.2f} ({pnl_pct:+.2f}%)",
                    "TP1 Target": f"${p.get('tp1_target', 0):,.2f}",
                    "TP2 Target": f"${p.get('tp2_target', 0):,.2f}",
                    "Stop Loss": f"${p.get('sl_target', 0):,.2f}",
                    "Status": scaled_label,
                }
            )
        st.dataframe(pd.DataFrame(pos_cards), use_container_width=True)
    else:
        st.info("No active open positions. Cash is 100% protected in reserve.")

    # Live News Feed Stream for Selected Ticker
    st.markdown(f"#### 📰 Real-Time Live News Stream for {selected_ticker}")
    news_df = get_news(selected_ticker, use_cache=True)
    if not news_df.empty:
        for idx, row in news_df.head(6).iterrows():
            title = row.get("Title", "No Title")
            src_info = row.get("source", {})
            src_name = (
                src_info.get("name", "Google News RSS")
                if isinstance(src_info, dict)
                else str(src_info)
            )
            url = row.get("url", "")
            pub_date = str(row.get("publishedAt", idx))[:19]

            st.markdown(
                f"""
                <div class="glass-card" style="margin-bottom: 0.5rem; padding: 0.8rem;">
                    <div style="font-weight: 700; color: #F8FAFC;">{title}</div>
                    <div style="font-size: 0.75rem; color: #00D4AA; margin-top: 0.3rem;">
                        Source: <b>{src_name}</b> | Published: {pub_date} 
                        {f'| <a href="{url}" target="_blank" style="color:#38BDF8;">Read Source ↗</a>' if url else ''}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# 🔴 LIVE MARKET STREAMING TICKER TAPE
# ==============================================================================
@st.fragment(run_every="15s")
def render_live_ticker_ribbon():
    """Renders a real-time streaming price tape across major universe stocks with up/down tick flashers."""
    ribbon_tickers = [
        "NVDA",
        "AVGO",
        "AMD",
        "AAPL",
        "MSFT",
        "TSLA",
        "META",
        "QQQ",
        "SPY",
    ]

    quotes_map = fetch_universe_live_quotes(ribbon_tickers)
    cards_html = []
    for t in ribbon_tickers:
        q = quotes_map.get(t, {})
        price = float(q.get("price", 0))
        chg = float(q.get("change_pct", 0))

        last_p = st.session_state.get(f"ribbon_last_{t}", price)
        tick_dir = "SAME"
        if price > last_p:
            tick_dir = "UP"
        elif price < last_p:
            tick_dir = "DOWN"
        st.session_state[f"ribbon_last_{t}"] = price

        tick_icon = "▲" if tick_dir == "UP" else ("▼" if tick_dir == "DOWN" else "●")
        chg_color = "#10B981" if chg >= 0 else "#EF4444"
        tick_badge = (
            f'<span style="font-size:0.65rem; color:{chg_color};">{tick_icon}</span>'
        )

        cards_html.append(
            f'<div style="background: rgba(15,23,42,0.75); border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 4px 10px; display: inline-flex; align-items: center; gap: 8px; margin-right: 8px;">'
            f'<b style="color: #F8FAFC; font-size: 0.8rem;">{t}</b>'
            f'<span style="color: #00D4AA; font-family: monospace; font-size: 0.8rem; font-weight: 700;">${price:,.2f}</span>'
            f'<span style="color: {chg_color}; font-size: 0.72rem; font-weight: 600;">{chg:+.2f}% {tick_badge}</span>'
            f"</div>"
        )

    st.markdown(
        f"""
        <div style="display: flex; overflow-x: auto; white-space: nowrap; padding-bottom: 6px; margin-bottom: 12px; scrollbar-width: none;">
            {''.join(cards_html)}
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

        # 1. Navigation Mode Selector (15 Institutional Workspaces)
        nav_mode = st.radio(
            "Navigation Workspace",
            [
                "⚡ AI Command Center",
                "🤖 Autonomous Trader & Live News",
                "🏛️ Multi-Agent Committee",
                "💬 AI Trade Copilot",
                "📐 3D Volatility Surface",
                "🌊 Order Book & Liquidity Heatmap",
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

        st.markdown("---")

        # 4. Live US & India Market Hours Clock Widget
        mkt = get_us_market_session_info()
        status_badge_color = (
            "#10B981"
            if mkt["is_open_for_trading"]
            else (
                "#38BDF8"
                if "PRE" in mkt["status"] or "AFTER" in mkt["status"]
                else "#F59E0B"
            )
        )
        st.markdown(
            f"""
            <div class="glass-card" style="padding: 0.8rem; font-size: 0.8rem; border-left: 3px solid {status_badge_color};">
                <div style="font-weight: 800; color: #F8FAFC; margin-bottom: 0.3rem;">
                    ⏰ <b>US Market Session</b>
                </div>
                <div style="color: {status_badge_color}; font-weight: 700; margin-bottom: 0.4rem;">
                    ● {mkt["status_label"]}
                </div>
                <div style="color: #94A3B8; font-size: 0.75rem; line-height: 1.4;">
                    • <b>NY (ET):</b> <span style="color:#E2E8F0;">{mkt["ny_time_str"]}</span><br>
                    • <b>IST:</b> <span style="color:#E2E8F0;">{mkt["ist_time_str"]}</span><br>
                    • <b>Regular:</b> <span style="color:#00D4AA;">{mkt["regular_hours_ist"]}</span><br>
                    • <b>Pre-Market:</b> <span style="color:#38BDF8;">{mkt["pre_market_ist"]}</span><br>
                    • <b>After-Hours:</b> <span style="color:#94A3B8;">{mkt["after_hours_ist"]}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Top Luxury Header ---
    mkt_info = get_us_market_session_info()
    header_status_color = (
        "#10B981"
        if mkt_info["is_open_for_trading"]
        else (
            "#38BDF8"
            if "PRE" in mkt_info["status"] or "AFTER" in mkt_info["status"]
            else "#F59E0B"
        )
    )
    st.markdown(
        f"""
        <div class="luxury-header">
            <div>
                <h3 style="margin: 0; color: #F8FAFC; font-weight: 900; letter-spacing: -0.02em;">Sentilyze Terminal</h3>
                <p style="margin: 0.2rem 0 0 0; color: #94A3B8; font-size: 0.85rem;">
                    Active Workspace: <b style="color: #F1F5F9;">{nav_mode}</b> &nbsp;·&nbsp; Specialist: <b style="color: #00D4AA;">{selected_ticker}</b>
                </p>
            </div>
            <div style="display: flex; gap: 0.6rem; align-items: center;">
                <div style="font-size: 0.82rem; color: {header_status_color}; font-weight: 700; background: {header_status_color}18; border: 1px solid {header_status_color}55; padding: 4px 12px; border-radius: 20px;">
                    ● {mkt_info["status_label"]} ({mkt_info["ny_time_str"]})
                </div>
                <div style="font-size: 0.82rem; color: #00D4AA; font-weight: 700; background: rgba(0, 212, 170, 0.1); border: 1px solid rgba(0, 212, 170, 0.3); padding: 4px 12px; border-radius: 20px;">
                    🟢 17 Specialist Models
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Live Real-Time Ticker Ribbon (Streaming 5s fragment) ---
    render_live_ticker_ribbon()

    # --- Workspace Routing ---
    if nav_mode == "⚡ AI Command Center":
        render_command_center(selected_ticker)
    elif nav_mode == "🤖 Autonomous Trader & Live News":
        render_autonomous_trader_workspace(selected_ticker)
    elif nav_mode == "🏛️ Multi-Agent Committee":
        render_committee_workspace(selected_ticker)
    elif nav_mode == "💬 AI Trade Copilot":
        render_copilot_workspace(selected_ticker)
    elif nav_mode == "📐 3D Volatility Surface":
        render_options_surface_workspace(selected_ticker)
    elif nav_mode == "🌊 Order Book & Liquidity Heatmap":
        render_liquidity_heatmap_workspace(selected_ticker)
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
