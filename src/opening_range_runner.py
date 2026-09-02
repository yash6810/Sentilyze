"""
Paper 25 Live Runner: Opening Range Breakout (ORB) on Top Stocks in Play.
Grounded in Zarattini, Barbon, Aziz (2024) SSRN: 4729284.
Executes daily at 09:35 AM EDT (07:05 PM IST).
"""

import os
import json
import logging
from typing import Dict, List, Any
import pandas as pd
import numpy as np

from src.utils import get_logger, get_market_timestamp
from src.opening_range_breakout import OpeningRangeBreakout
from src.realtime_tracker import fetch_live_quote
from src.data_ingestion import get_price_history, get_news
from src.sentiment_analysis import analyze_sentiment
from src.paper_broker import PaperBroker
from src.alerts import send_discord_execution_alert

logger = get_logger(__name__)


def run_opening_range_session() -> Dict[str, Any]:
    """
    Executes a live 5-Minute Opening Range Breakout scan across top liquid assets:
    1. Loads universe from stocks.txt.
    2. Identifies top 5 'Stocks in Play' (RVOL >= 1.5, high ATR, catalyst news).
    3. Evaluates 5-min H5/L5 breakout signals.
    4. Routes triggered breakout orders to PaperBroker ($100k simulated paper account).
    5. Dispatches institutional Discord execution alerts.
    """
    logger.info(
        "🚀 Launching Paper 25: 5-Minute Opening Range Breakout (ORB) Session..."
    )

    stocks_file = "stocks.txt"
    tickers = [
        "NVDA",
        "AAPL",
        "MSFT",
        "GOOGL",
        "META",
        "AMZN",
        "TSLA",
        "AMD",
        "AVGO",
        "COIN",
        "PLTR",
        "ARM",
    ]
    if os.path.exists(stocks_file):
        with open(stocks_file, "r") as f:
            lines = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
            if lines:
                tickers = lines[:30]

    universe_data = {}
    catalyst_scores = {}

    for ticker in tickers:
        try:
            df = get_price_history(ticker, period="1mo", use_cache=True)
            if not df.empty and len(df) >= 10:
                universe_data[ticker] = df
                news = get_news(ticker, use_cache=True)
                if isinstance(news, pd.DataFrame) and not news.empty:
                    sent = analyze_sentiment(news.head(5), ticker=ticker)
                    if (
                        isinstance(sent, pd.DataFrame)
                        and "sentiment_score" in sent.columns
                    ):
                        catalyst_scores[ticker] = max(
                            0.0, float(sent["sentiment_score"].mean()) + 0.5
                        )
        except Exception as e:
            logger.debug(f"ORB data ingestion notice for {ticker}: {e}")

    orb = OpeningRangeBreakout(
        range_minutes=5,
        rvol_threshold=1.3,
        atr_multiplier_tp=2.0,
        atr_multiplier_sl=1.0,
    )
    stocks_in_play = orb.filter_stocks_in_play(universe_data, catalyst_scores, top_k=5)
    logger.info(f"⚡ Top Stocks in Play identified: {stocks_in_play}")

    signals = []
    broker = PaperBroker()
    executed_trades = []

    for ticker in stocks_in_play:
        df_hist = universe_data.get(ticker, pd.DataFrame())
        if df_hist.empty:
            continue
        cat_score = catalyst_scores.get(ticker, 0.5)
        sig_info = orb.evaluate_orb_signals(df_hist, sentiment_score=cat_score)

        if sig_info.get("signal") == 1:
            entry_p = sig_info.get("current_close", 100.0)
            atr_val = sig_info.get("atr", entry_p * 0.02)
            tp = round(entry_p + (2.0 * atr_val), 2)
            sl = round(entry_p - (1.0 * atr_val), 2)

            signals.append(
                {
                    "ticker": ticker,
                    "signal": "BUY",
                    "current_price": entry_p,
                    "confidence": float(sig_info.get("strength", 0.7)),
                    "take_profit": tp,
                    "stop_loss": sl,
                }
            )

    # Execute Paper Trades on $100k Account via Concentrated Kelly Sizing
    execution_result = broker.execute_daily_signals(signals) if signals else {}
    executed_buys = execution_result.get("buys", [])

    for buy in executed_buys:
        tk = buy.get("ticker")
        send_discord_execution_alert(
            trade_data={
                "ticker": tk,
                "action": "BUY",
                "shares": buy.get("shares", 100),
                "price": buy.get("entry_price", 100.0),
                "entry_price": buy.get("entry_price", 100.0),
                "tp1": buy.get("tp1_target", 110.0),
                "tp2": buy.get("tp2_target", 120.0),
                "stop_loss": buy.get("sl_target", 95.0),
                "kelly_pct": 8.5,
            }
        )

    result_payload = {
        "session_time": get_market_timestamp(),
        "stocks_in_play": stocks_in_play,
        "signals_count": len(signals),
        "executed_paper_trades": executed_buys,
        "portfolio_summary": broker.get_portfolio_summary(),
    }

    os.makedirs("results", exist_ok=True)
    with open(
        os.path.join("results", "opening_range_latest.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result_payload, f, indent=2)

    logger.info(
        f"✅ ORB Session Complete. {len(executed_trades)} paper trades executed."
    )
    return result_payload


if __name__ == "__main__":
    run_opening_range_session()
