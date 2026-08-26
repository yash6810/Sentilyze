import os
import time
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from src.paper_broker import PaperBroker
from src.utils import get_logger

logger = get_logger(__name__)

UNIVERSE_TICKERS = [
    "NVDA", "AAPL", "MSFT", "GOOGL", "META", "TSLA", "AMZN",
    "AVGO", "AMD", "PLTR", "LLY", "QQQ", "SPY", "JPM", "COST", "NFLX", "TSM"
]


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


def evaluate_intraday_execution(
    broker: Optional[PaperBroker] = None,
    discord_url: Optional[str] = None,
    telegram_token: Optional[str] = None,
    telegram_chat: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Autonomous Intraday Live Market Execution Engine:
    1. Fetches live real-time prices for all open positions.
    2. Checks if any position hit its Take-Profit (+2.5 ATR) or Stop-Loss target.
    3. Executes the exit immediately, updates paper portfolio, and sends flash alerts.
    """
    broker = broker or PaperBroker()
    open_positions = broker.state.get("open_positions", {})
    if not open_positions:
        logger.info("No active open positions to track during intraday session.")
        return {"status": "NO_OPEN_POSITIONS", "actions": []}

    executed_trades = []
    open_tickers = list(open_positions.keys())
    logger.info(f"Checking intraday real-time quotes for {len(open_tickers)} open holdings: {open_tickers}")

    for ticker in open_tickers:
        pos = open_positions[ticker]
        quote = fetch_live_quote(ticker)
        curr_price = float(quote.get("price", 0))
        if curr_price <= 0:
            continue

        pos["current_price"] = curr_price
        shares = int(pos.get("shares", 0))
        entry_price = float(pos.get("entry_price", curr_price))
        tp_target = float(pos.get("tp_target", entry_price * 1.06))
        sl_target = float(pos.get("sl_target", entry_price * 0.95))

        exit_reason = None
        if curr_price >= tp_target:
            exit_reason = "TAKE_PROFIT"
        elif curr_price <= sl_target:
            exit_reason = "STOP_LOSS"

        if exit_reason:
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
                "reason": exit_reason,
            }
            broker.state["closed_trades"].append(trade_record)
            del broker.state["open_positions"][ticker]
            executed_trades.append(trade_record)

            logger.info(f"⚡ [INTRADAY EXECUTION] Sold {ticker} @ ${curr_price:.2f} ({exit_reason}) | PnL: ${pnl:+,.2f} ({ret_pct:+.2f}%)")

            # Dispatch Flash Alert to Discord
            d_url = discord_url or os.getenv("DISCORD_WEBHOOK_URL")
            if d_url:
                _send_intraday_discord_flash(trade_record, d_url)

            # Dispatch Flash Alert to Telegram
            t_token = telegram_token or os.getenv("TELEGRAM_BOT_TOKEN")
            t_chat = telegram_chat or os.getenv("TELEGRAM_CHAT_ID")
            if t_token and t_chat:
                _send_intraday_telegram_flash(trade_record, t_token, t_chat)

    if executed_trades:
        now_str = datetime.now(timezone.utc).isoformat()
        broker._recalculate_metrics(now_str[:10], now_str)
        broker._save()
        logger.info(f"Intraday execution complete. {len(executed_trades)} trades executed.")
    else:
        # Just update live unrealized mark-to-market prices
        now_str = datetime.now(timezone.utc).isoformat()
        broker._recalculate_metrics(now_str[:10], now_str)
        broker._save()

    return {"status": "SUCCESS", "executed_trades": executed_trades, "summary": broker.get_portfolio_summary()}


def _send_intraday_discord_flash(trade: Dict[str, Any], webhook_url: str):
    """Sends immediate Discord flash notification for Take-Profit / Stop-Loss hits."""
    try:
        is_tp = trade["reason"] == "TAKE_PROFIT"
        color = 0x10B981 if is_tp else 0xEF4444
        title = f"🎯 [TAKE-PROFIT EXECUTED] {trade['ticker']}" if is_tp else f"🛑 [STOP-LOSS EXECUTED] {trade['ticker']}"
        desc = (
            f"**Intraday Live Market Execution Triggered**\n\n"
            f"• **Ticker:** `{trade['ticker']}`\n"
            f"• **Shares:** `{trade['shares']}`\n"
            f"• **Entry Price:** `${trade['entry_price']:.2f}`\n"
            f"• **Exit Price:** `${trade['exit_price']:.2f}`\n"
            f"• **Net Realized PnL:** **`${trade['pnl']:+,.2f}` ({trade['return_pct']:+.2f}%)**\n"
            f"• **Execution Reason:** `{trade['reason']}`\n"
        )
        payload = {
            "embeds": [
                {
                    "title": title,
                    "description": desc,
                    "color": color,
                    "footer": {"text": "Sentilyze Autonomous Intraday Trader • Live Market Action"},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ]
        }
        requests.post(webhook_url, json=payload, timeout=8)
    except Exception as e:
        logger.warning(f"Failed sending intraday Discord alert: {e}")


def _send_intraday_telegram_flash(trade: Dict[str, Any], bot_token: str, chat_id: str):
    """Sends immediate Telegram flash notification for Take-Profit / Stop-Loss hits."""
    try:
        is_tp = trade["reason"] == "TAKE_PROFIT"
        icon = "🎯" if is_tp else "🛑"
        text = (
            f"{icon} *[INTRADAY TRADE EXECUTED]*\n\n"
            f"• *Ticker:* `{trade['ticker']}`\n"
            f"• *Shares:* `{trade['shares']}`\n"
            f"• *Entry:* `${trade['entry_price']:.2f}` ➔ *Exit:* `${trade['exit_price']:.2f}`\n"
            f"• *Net PnL:* *`${trade['pnl']:+,.2f}` ({trade['return_pct']:+.2f}%)*\n"
            f"• *Reason:* `{trade['reason']}`"
        )
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=8)
    except Exception as e:
        logger.warning(f"Failed sending intraday Telegram alert: {e}")
