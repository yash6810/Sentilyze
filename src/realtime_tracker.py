import os
import time
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from src.paper_broker import PaperBroker
from src.sentiment_analysis import analyze_sentiment
from src.data_ingestion import get_news
from src.utils import get_logger

logger = get_logger(__name__)

UNIVERSE_TICKERS = [
    "NVDA", "AAPL", "MSFT", "GOOGL", "META", "TSLA", "AMZN",
    "AVGO", "AMD", "PLTR", "LLY", "QQQ", "SPY", "JPM", "COST", "NFLX", "TSM"
]


from zoneinfo import ZoneInfo


def get_us_market_session_info() -> Dict[str, Any]:
    """
    Computes current US stock market session status (Pre-Market, Regular Hours, After-Hours, Closed),
    taking into account US Daylight Saving Time (EDT vs EST) and India Standard Time (IST).
    """
    ny_tz = ZoneInfo("America/New_York")
    ist_tz = ZoneInfo("Asia/Kolkata")

    now_utc = datetime.now(timezone.utc)
    now_ny = now_utc.astimezone(ny_tz)
    now_ist = now_utc.astimezone(ist_tz)

    is_dst = bool(now_ny.dst())
    weekday = now_ny.weekday()  # 0 = Monday, 6 = Sunday
    is_weekend = weekday in (5, 6)

    # Minute of day in NY (0 to 1439)
    current_minute = now_ny.hour * 60 + now_ny.minute

    # Pre-market: 04:00 (240) to 09:30 (570)
    # Regular hours: 09:30 (570) to 16:00 (960)
    # After-hours: 16:00 (960) to 20:00 (1200)

    if is_weekend:
        status = "WEEKEND_CLOSED"
        status_label = "Weekend (Closed)"
    elif 570 <= current_minute < 960:
        status = "REGULAR_TRADING"
        status_label = "Regular Market Open (Live Trading)"
    elif 240 <= current_minute < 570:
        status = "PRE_MARKET"
        status_label = "Pre-Market Session"
    elif 960 <= current_minute < 1200:
        status = "AFTER_HOURS"
        status_label = "After-Hours Session"
    else:
        status = "OVERNIGHT_CLOSED"
        status_label = "Overnight (Closed)"

    return {
        "status": status,
        "status_label": status_label,
        "is_open_for_trading": status == "REGULAR_TRADING",
        "ny_time_str": now_ny.strftime("%I:%M %p %Z"),
        "ist_time_str": now_ist.strftime("%I:%M %p IST"),
        "is_dst": is_dst,
        "tz_label": "EDT (Daylight Saving Time)" if is_dst else "EST (Standard Time)",
        "regular_hours_ist": "7:00 PM – 1:30 AM IST" if is_dst else "8:00 PM – 2:30 AM IST",
        "pre_market_ist": "1:30 PM – 7:00 PM IST" if is_dst else "2:30 PM – 8:00 PM IST",
        "after_hours_ist": "1:30 AM – 5:30 AM IST" if is_dst else "2:30 AM – 6:30 AM IST",
    }


def _get_browser_session() -> requests.Session:
    """Creates a requests Session with modern desktop browser headers."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )
    return session


def fetch_live_quote(ticker: str) -> Dict[str, Any]:
    """
    Fetches sub-second real-time market quote using Yahoo Finance Direct Chart API / Finnhub / Alpaca.
    """
    # 1. Primary: Direct Yahoo Chart API
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"range": "1d", "interval": "1m"}
        session = _get_browser_session()
        res = session.get(url, params=params, timeout=6)
        if res.status_code == 200:
            meta = res.json()["chart"]["result"][0]["meta"]
            curr_price = float(meta.get("regularMarketPrice", 0))
            prev_close = float(meta.get("chartPreviousClose", curr_price))
            day_high = float(meta.get("regularMarketDayHigh", curr_price))
            day_low = float(meta.get("regularMarketDayLow", curr_price))
            change_pct = ((curr_price - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0

            if curr_price > 0:
                return {
                    "ticker": ticker,
                    "price": round(curr_price, 2),
                    "prev_close": round(prev_close, 2),
                    "day_high": round(day_high, 2),
                    "day_low": round(day_low, 2),
                    "change_pct": round(change_pct, 2),
                    "status": "LIVE",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
    except Exception as e:
        logger.debug(f"Direct Yahoo quote failed for {ticker}: {e}")

    # 2. Fallback: Finnhub API
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    if finnhub_key:
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={finnhub_key}"
            res = requests.get(url, timeout=6)
            if res.status_code == 200:
                data = res.json()
                curr_price = float(data.get("c", 0))
                if curr_price > 0:
                    prev_close = float(data.get("pc", curr_price))
                    change_pct = ((curr_price - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
                    return {
                        "ticker": ticker,
                        "price": round(curr_price, 2),
                        "prev_close": round(prev_close, 2),
                        "day_high": round(float(data.get("h", curr_price)), 2),
                        "day_low": round(float(data.get("l", curr_price)), 2),
                        "change_pct": round(change_pct, 2),
                        "status": "LIVE",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
        except Exception:
            pass

    return {
        "ticker": ticker,
        "price": 0.0,
        "prev_close": 0.0,
        "day_high": 0.0,
        "day_low": 0.0,
        "change_pct": 0.0,
        "status": "OFFLINE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def fetch_universe_live_quotes(tickers: List[str] = UNIVERSE_TICKERS) -> Dict[str, Dict[str, Any]]:
    """Fetches real-time quotes across the entire universe."""
    quotes = {}
    for ticker in tickers:
        quotes[ticker] = fetch_live_quote(ticker)
    return quotes


def check_live_news_sentiment_shock(ticker: str) -> bool:
    """
    Checks if breaking news in the last few hours has a severe negative sentiment shock (< -0.50).
    """
    try:
        news_df = get_news(ticker, cache_duration_hours=2)
        if not news_df.empty:
            scored = analyze_sentiment(news_df, ticker=ticker, use_cache=False)
            if not scored.empty and "sentiment_score" in scored.columns:
                recent_avg = scored["sentiment_score"].tail(3).mean()
                if recent_avg < -0.50:
                    logger.warning(f"🚨 Severe negative news shock detected for {ticker} (Score: {recent_avg:.2f})")
                    return True
    except Exception as e:
        logger.debug(f"News sentiment shock check skipped for {ticker}: {e}")
    return False


def evaluate_intraday_execution(
    broker: Optional[PaperBroker] = None,
    discord_url: Optional[str] = None,
    telegram_token: Optional[str] = None,
    telegram_chat: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Autonomous 5-Minute Intraday Market Execution Engine:
    1. Evaluates 50/50 Scale-Out (TP1 +2.5 ATR), Runner Exit (TP2 +4.5 ATR), Break-Even Ratchet, and Stop-Loss.
    2. Runs FinBERT news catalyst check for emergency loss prevention.
    3. Executes trades, updates paper portfolio, and dispatches instant flash alerts.
    """
    broker = broker or PaperBroker()
    open_positions = broker.state.get("open_positions", {})

    # If capital is liquid and open positions are below capacity (max 2), auto-deploy into top AI signals
    if len(open_positions) < 2 and broker.state.get("cash", 0) > 15000.0:
        signals_file = "results/daily_signals_latest.json"
        if os.path.exists(signals_file):
            try:
                with open(signals_file, "r") as f:
                    data = json.load(f)
                signals_list = data.get("signals", []) if isinstance(data, dict) else data
                if signals_list:
                    buy_actions = broker.execute_daily_signals(signals_list)
                    if buy_actions.get("buys"):
                        logger.info(f"🚀 [INTRADAY LIVE ENTRY] Auto-deployed cash into {len(buy_actions['buys'])} holdings: {[b['ticker'] for b in buy_actions['buys']]}")
                        for b_record in buy_actions["buys"]:
                            _send_intraday_buy_notification(b_record, discord_url, telegram_token, telegram_chat)
                        open_positions = broker.state.get("open_positions", {})
            except Exception as e:
                logger.warning(f"Failed to auto-deploy liquid cash during intraday check: {e}")

    if not open_positions:
        logger.info("No active open positions to track during intraday session.")
        return {"status": "NO_OPEN_POSITIONS", "actions": []}

    executed_trades = []
    open_tickers = list(open_positions.keys())
    logger.info(f"⚡ [5-MIN GUARDIAN] Checking live quotes for {len(open_tickers)} holdings: {open_tickers}")

    for ticker in open_tickers:
        pos = open_positions[ticker]
        quote = fetch_live_quote(ticker)
        curr_price = float(quote.get("price", 0))
        if curr_price <= 0:
            continue

        pos["current_price"] = curr_price
        shares = int(pos.get("shares", 0))
        entry_price = float(pos.get("entry_price", curr_price))
        tp1_target = float(pos.get("tp1_target", entry_price * 1.06))
        tp2_target = float(pos.get("tp2_target", entry_price * 1.12))
        sl_target = float(pos.get("sl_target", entry_price * 0.95))
        scaled_out = pos.get("scaled_out", False)

        # Check Emergency News Catalyst Shock
        news_shock = check_live_news_sentiment_shock(ticker)
        if news_shock:
            proceeds = float(shares * curr_price)
            cost_basis = float(shares * entry_price)
            pnl = float(proceeds - cost_basis)
            ret_pct = ((curr_price - entry_price) / entry_price) * 100.0

            broker.state["cash"] += proceeds
            broker.state["realized_pnl"] += pnl
            broker.state["total_trades"] += 1
            if pnl > 0:
                broker.state["winning_trades"] += 1
            else:
                broker.state["losing_trades"] += 1

            now_str = datetime.now(timezone.utc).isoformat()
            trade_record = {
                "ticker": ticker,
                "shares": shares,
                "entry_price": entry_price,
                "exit_price": curr_price,
                "entry_date": pos.get("entry_date", now_str[:10]),
                "exit_date": now_str[:10],
                "pnl": round(pnl, 2),
                "return_pct": round(ret_pct, 2),
                "reason": "🚨 EMERGENCY_NEWS_CATALYST_EXIT",
            }
            broker.state["closed_trades"].append(trade_record)
            del broker.state["open_positions"][ticker]
            executed_trades.append(trade_record)
            _send_flash_notifications(trade_record, discord_url, telegram_token, telegram_chat)
            continue

        # Check Stage 1 Scale-Out (+2.5 ATR)
        if not scaled_out and curr_price >= tp1_target:
            half_shares = max(1, shares // 2)
            proceeds = float(half_shares * curr_price)
            cost_basis = float(half_shares * entry_price)
            pnl = float(proceeds - cost_basis)
            ret_pct = ((curr_price - entry_price) / entry_price) * 100.0

            broker.state["cash"] += proceeds
            broker.state["realized_pnl"] += pnl
            pos["shares"] = shares - half_shares
            pos["scaled_out"] = True
            pos["sl_target"] = round(entry_price * 1.002, 2)  # Move SL to Break-Even

            now_str = datetime.now(timezone.utc).isoformat()
            trade_record = {
                "ticker": ticker,
                "shares": half_shares,
                "entry_price": entry_price,
                "exit_price": curr_price,
                "entry_date": pos.get("entry_date", now_str[:10]),
                "exit_date": now_str[:10],
                "pnl": round(pnl, 2),
                "return_pct": round(ret_pct, 2),
                "reason": "🎯 SCALE_OUT_TP1 (50% Banked)",
                "status": "RISK_FREE_RUNNER",
            }
            broker.state["closed_trades"].append(trade_record)
            executed_trades.append(trade_record)
            logger.info(f"🎯 [STAGE 1 SCALE-OUT] Banked 50% {ticker} @ ${curr_price:.2f} | PnL: ${pnl:+,.2f} ({ret_pct:+.2f}%)")
            _send_flash_notifications(trade_record, discord_url, telegram_token, telegram_chat)

        # Check Stage 2 Runner Exit (+4.5 ATR)
        elif scaled_out and curr_price >= tp2_target:
            proceeds = float(shares * curr_price)
            cost_basis = float(shares * entry_price)
            pnl = float(proceeds - cost_basis)
            ret_pct = ((curr_price - entry_price) / entry_price) * 100.0

            broker.state["cash"] += proceeds
            broker.state["realized_pnl"] += pnl
            broker.state["total_trades"] += 1
            broker.state["winning_trades"] += 1

            now_str = datetime.now(timezone.utc).isoformat()
            trade_record = {
                "ticker": ticker,
                "shares": shares,
                "entry_price": entry_price,
                "exit_price": curr_price,
                "entry_date": pos.get("entry_date", now_str[:10]),
                "exit_date": now_str[:10],
                "pnl": round(pnl, 2),
                "return_pct": round(ret_pct, 2),
                "reason": "🏆 FULL_TP2_RUNNER (+4.5 ATR Exit)",
            }
            broker.state["closed_trades"].append(trade_record)
            del broker.state["open_positions"][ticker]
            executed_trades.append(trade_record)
            logger.info(f"🏆 [STAGE 2 RUNNER EXIT] Closed {ticker} @ ${curr_price:.2f} | PnL: ${pnl:+,.2f} ({ret_pct:+.2f}%)")
            _send_flash_notifications(trade_record, discord_url, telegram_token, telegram_chat)

        # Check Stop-Loss / Break-Even Exit
        elif curr_price <= sl_target:
            proceeds = float(shares * curr_price)
            cost_basis = float(shares * entry_price)
            pnl = float(proceeds - cost_basis)
            ret_pct = ((curr_price - entry_price) / entry_price) * 100.0

            broker.state["cash"] += proceeds
            broker.state["realized_pnl"] += pnl
            broker.state["total_trades"] += 1
            if (pnl > 0) or scaled_out:
                broker.state["winning_trades"] += 1
            else:
                broker.state["losing_trades"] += 1

            reason = "🛡️ BREAK_EVEN_EXIT" if scaled_out else "🛑 STOP_LOSS"
            now_str = datetime.now(timezone.utc).isoformat()
            trade_record = {
                "ticker": ticker,
                "shares": shares,
                "entry_price": entry_price,
                "exit_price": curr_price,
                "entry_date": pos.get("entry_date", now_str[:10]),
                "exit_date": now_str[:10],
                "pnl": round(pnl, 2),
                "return_pct": round(ret_pct, 2),
                "reason": reason,
            }
            broker.state["closed_trades"].append(trade_record)
            del broker.state["open_positions"][ticker]
            executed_trades.append(trade_record)
            logger.info(f"[{reason}] Exited {ticker} @ ${curr_price:.2f} | PnL: ${pnl:+,.2f}")
            _send_flash_notifications(trade_record, discord_url, telegram_token, telegram_chat)

    now_str = datetime.now(timezone.utc).isoformat()
    broker._recalculate_metrics(now_str[:10], now_str)
    broker._save()

    return {"status": "SUCCESS", "executed_trades": executed_trades, "summary": broker.get_portfolio_summary()}


def _send_intraday_buy_notification(
    buy_record: Dict[str, Any],
    discord_url: Optional[str] = None,
    telegram_token: Optional[str] = None,
    telegram_chat: Optional[str] = None,
):
    """Sends immediate Discord & Telegram alert when a new position is opened."""
    d_url = discord_url or os.getenv("DISCORD_WEBHOOK_URL")
    if d_url:
        try:
            payload = {
                "embeds": [
                    {
                        "title": f"🚀 NEW INTRADAY POSITION OPENED • {buy_record['ticker']}",
                        "description": (
                            f"**Autonomous Intraday Market Entry**\n\n"
                            f"• **Ticker:** `{buy_record['ticker']}`\n"
                            f"• **Shares Bought:** `{buy_record['shares']}`\n"
                            f"• **Entry Spot Price:** `${buy_record['entry_price']:.2f}`\n"
                            f"• **Total Cost:** `${buy_record.get('cost', buy_record['shares']*buy_record['entry_price']):,.2f}`\n"
                            f"• **TP1 (50% Scale-Out):** `${buy_record.get('tp1_target', 0):.2f}`\n"
                            f"• **TP2 (Runner Target):** `${buy_record.get('tp2_target', 0):.2f}`\n"
                            f"• **Stop-Loss Protection:** `${buy_record.get('sl_target', 0):.2f}`\n"
                        ),
                        "color": 0x00D4AA,
                        "footer": {"text": "Sentilyze 5-Minute Trade Guardian • Autonomous Live Execution"},
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                ]
            }
            requests.post(d_url, json=payload, timeout=8)
        except Exception as e:
            logger.warning(f"Failed sending intraday BUY Discord alert: {e}")

    t_token = telegram_token or os.getenv("TELEGRAM_BOT_TOKEN")
    t_chat = telegram_chat or os.getenv("TELEGRAM_CHAT_ID")
    if t_token and t_chat:
        try:
            text = (
                f"🚀 *[NEW INTRADAY TRADE OPENED]*\n\n"
                f"• *Ticker:* `{buy_record['ticker']}`\n"
                f"• *Shares:* `{buy_record['shares']}`\n"
                f"• *Entry Price:* `${buy_record['entry_price']:.2f}`\n"
                f"• *TP1 Target (+2.5 ATR):* `${buy_record.get('tp1_target', 0):.2f}`\n"
                f"• *TP2 Runner (+4.5 ATR):* `${buy_record.get('tp2_target', 0):.2f}`\n"
                f"• *Stop-Loss Target:* `${buy_record.get('sl_target', 0):.2f}`"
            )
            url = f"https://api.telegram.org/bot{t_token}/sendMessage"
            requests.post(url, json={"chat_id": t_chat, "text": text, "parse_mode": "Markdown"}, timeout=8)
        except Exception as e:
            logger.warning(f"Failed sending intraday BUY Telegram alert: {e}")


def _send_flash_notifications(
    trade: Dict[str, Any],
    discord_url: Optional[str] = None,
    telegram_token: Optional[str] = None,
    telegram_chat: Optional[str] = None,
):
    """Sends instant flash notifications to Discord & Telegram."""
    d_url = discord_url or os.getenv("DISCORD_WEBHOOK_URL")
    if d_url:
        _send_intraday_discord_flash(trade, d_url)

    t_token = telegram_token or os.getenv("TELEGRAM_BOT_TOKEN")
    t_chat = telegram_chat or os.getenv("TELEGRAM_CHAT_ID")
    if t_token and t_chat:
        _send_intraday_telegram_flash(trade, t_token, t_chat)


def _send_intraday_discord_flash(trade: Dict[str, Any], webhook_url: str):
    """Sends immediate Discord flash notification."""
    try:
        is_win = trade["pnl"] >= 0
        color = 0x10B981 if is_win else 0xEF4444
        title = f"{trade['reason']} • {trade['ticker']}"
        desc = (
            f"**5-Minute Autonomous Intraday Execution**\n\n"
            f"• **Ticker:** `{trade['ticker']}`\n"
            f"• **Shares:** `{trade['shares']}`\n"
            f"• **Entry Price:** `${trade['entry_price']:.2f}`\n"
            f"• **Exit Price:** `${trade['exit_price']:.2f}`\n"
            f"• **Net Realized PnL:** **`${trade['pnl']:+,.2f}` ({trade['return_pct']:+.2f}%)**\n"
            f"• **Trigger:** `{trade['reason']}`\n"
        )
        payload = {
            "embeds": [
                {
                    "title": title,
                    "description": desc,
                    "color": color,
                    "footer": {"text": "Sentilyze 5-Minute Trade Guardian • Autonomous Live Execution"},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ]
        }
        requests.post(webhook_url, json=payload, timeout=8)
    except Exception as e:
        logger.warning(f"Failed sending intraday Discord alert: {e}")


def _send_intraday_telegram_flash(trade: Dict[str, Any], bot_token: str, chat_id: str):
    """Sends immediate Telegram flash notification."""
    try:
        icon = "💰" if trade["pnl"] >= 0 else "🛑"
        text = (
            f"{icon} *[5-MIN INTRADAY EXECUTION]*\n\n"
            f"• *Ticker:* `{trade['ticker']}`\n"
            f"• *Shares:* `{trade['shares']}`\n"
            f"• *Entry:* `${trade['entry_price']:.2f}` ➔ *Exit:* `${trade['exit_price']:.2f}`\n"
            f"• *Net PnL:* *`${trade['pnl']:+,.2f}` ({trade['return_pct']:+.2f}%)*\n"
            f"• *Trigger:* `{trade['reason']}`"
        )
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=8)
    except Exception as e:
        logger.warning(f"Failed sending intraday Telegram alert: {e}")


def run_5min_guardian_loop(duration_minutes: int = 360):
    """
    Continuous 5-Minute Intraday Guardian Loop during active market hours.
    """
    logger.info(f"Starting 5-Minute Intraday Trade Guardian Loop for {duration_minutes} minutes...")
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)

    while time.time() < end_time:
        res = evaluate_intraday_execution()
        logger.info(f"5-Minute Poll Complete. Executed: {len(res.get('executed_trades', []))} trades.")
        time.sleep(300)  # Sleep 5 minutes
