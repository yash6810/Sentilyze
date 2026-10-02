"""
Benzinga Live News Catalyst Scanner & Autonomous Execution Engine.
===================================================================
1. Scrapes real-time Benzinga breaking financial news and analyst wire.
2. Extracts mentioned equity tickers and classifies catalyst events (earnings beats,
   multi-billion dollar contracts, analyst upgrades, regulatory/lawsuit penalties).
3. Evaluates FinBERT sentiment and convenes the Sentilyze Multi-Agent Trading Committee.
4. Executes autonomous paper buy/sell orders through PaperBroker with strict risk guards.
5. Persists live state to results/benzinga_live_scan_latest.json.
"""

import os
import re
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set
import requests
import pandas as pd
import defusedxml.ElementTree as ET

from src.utils import get_logger, sanitize_filename
from src.paper_broker import PaperBroker
from src.realtime_tracker import fetch_live_quote
from src.agent_committee import convene_trading_committee

logger = get_logger("benzinga_news_trader")

BENZINGA_SCAN_FILE = os.path.join("results", "benzinga_live_scan_latest.json")

# Stop-words that look like uppercase tickers but are common headline words
NON_TICKER_WORDS: Set[str] = {
    "AI",
    "CEO",
    "CFO",
    "CTO",
    "USA",
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "CAD",
    "ETF",
    "ETFS",
    "FED",
    "IRS",
    "SEC",
    "FDA",
    "IPO",
    "GDP",
    "CPI",
    "PPI",
    "NEW",
    "NEWS",
    "BUY",
    "SELL",
    "HOLD",
    "BEAT",
    "MISS",
    "GAIN",
    "FALL",
    "DROP",
    "SURGE",
    "RALLY",
    "DIP",
    "TOP",
    "BEST",
    "HIGH",
    "LOW",
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "FY24",
    "FY25",
    "FY26",
    "FY27",
    "TECH",
    "OIL",
    "GAS",
    "GOLD",
    "EV",
    "LNG",
    "ALL",
    "ONE",
    "TWO",
    "BIG",
    "FREE",
    "MORE",
    "GOOD",
    "SOAR",
    "PLUNGE",
    "ROBOT",
    "DEAL",
    "CHIP",
    "DEBT",
    "BANK",
    "FUND",
    "BULL",
    "BEAR",
    "SAFE",
    "WALL",
    "ST",
    "US",
    "UK",
    "EU",
    "NOW",
    "WEEK",
    "DAY",
    "YEAR",
}

# Bullish / Bearish catalyst keywords
BULLISH_KEYWORDS = [
    "beat",
    "beats",
    "record",
    "contract",
    "surge",
    "surges",
    "upgrade",
    "upgrades",
    "partnership",
    "deal",
    "secures",
    "acquisition",
    "soar",
    "soars",
    "approval",
    "double beat",
    "stronger",
    "boost",
    "boosts",
    "rally",
    "rallies",
    "dividend hike",
]
BEARISH_KEYWORDS = [
    "penalty",
    "penalty in",
    "fine",
    "lawsuit",
    "investigation",
    "miss",
    "misses",
    "downgrade",
    "downgrades",
    "probe",
    "fraud",
    "warns",
    "cut",
    "cuts",
    "slump",
    "drops",
    "falls",
    "plunge",
    "plunges",
    "52-week low",
    "antitrust",
]


def extract_tickers_from_headline(headline: str) -> List[str]:
    """
    Extracts high-confidence stock tickers from a Benzinga headline using
    exchange-qualified patterns (e.g., NASDAQ:NVDA, NYSE:NKE) and ticker syntax.
    """
    extracted = set()

    # Pattern 1: Exchange prefix e.g. (NASDAQ:AVGO), (NYSE:NKE)
    exchange_matches = re.findall(
        r"\b(?:NASDAQ|NYSE|AMEX):\s*([A-Z]{1,5})\b", headline, re.IGNORECASE
    )
    for m in exchange_matches:
        sym = m.upper().strip()
        if sym not in NON_TICKER_WORDS and 1 <= len(sym) <= 5:
            extracted.add(sym)

    # Pattern 2: Parenthesized ticker e.g. "Broadcom (AVGO)", "Tesla (TSLA)"
    paren_matches = re.findall(r"\(([A-Z]{1,5})\)", headline)
    for m in paren_matches:
        sym = m.upper().strip()
        if sym not in NON_TICKER_WORDS and 1 <= len(sym) <= 5:
            extracted.add(sym)

    # Pattern 3: Explicit "XYZ Stock" e.g. "Super Micro (SMCI) Stock", "NIO Stock"
    stock_matches = re.findall(r"\b([A-Z]{2,5})\s+Stock\b", headline, re.IGNORECASE)
    for m in stock_matches:
        sym = m.upper().strip()
        if sym not in NON_TICKER_WORDS and 1 <= len(sym) <= 5:
            extracted.add(sym)

    return sorted(list(extracted))


def fetch_top_benzinga_news_wire(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Scrapes the live Benzinga breaking news stream via Google News RSS wire.
    Returns parsed articles with title, pub_date, link, and extracted tickers.
    """
    url = "https://news.google.com/rss/search?q=site:benzinga.com&hl=en-US&gl=US&ceid=US:en"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
    }

    articles: List[Dict[str, Any]] = []
    try:
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            logger.warning(f"Benzinga news stream HTTP {resp.status_code}")
            return articles

        root = ET.fromstring(resp.content)
        items = root.findall(".//item")

        for item in items[:limit]:
            title_elem = item.find("title")
            pub_elem = item.find("pubDate")
            link_elem = item.find("link")

            title = (
                title_elem.text.strip()
                if title_elem is not None and title_elem.text
                else ""
            )
            pub_date = (
                pub_elem.text.strip() if pub_elem is not None and pub_elem.text else ""
            )
            link = (
                link_elem.text.strip()
                if link_elem is not None and link_elem.text
                else ""
            )

            if not title:
                continue

            clean_title = re.sub(
                r"\s*-\s*Benzinga\s*$", "", title, flags=re.IGNORECASE
            ).strip()
            tickers = extract_tickers_from_headline(clean_title)

            # Analyze headline sentiment keywords
            lower_title = clean_title.lower()
            bull_count = sum(1 for k in BULLISH_KEYWORDS if k in lower_title)
            bear_count = sum(1 for k in BEARISH_KEYWORDS if k in lower_title)

            if bull_count > bear_count:
                sentiment = "BULLISH"
                score = min(1.0, 0.40 + 0.20 * bull_count)
            elif bear_count > bull_count:
                sentiment = "BEARISH"
                score = max(-1.0, -0.40 - 0.20 * bear_count)
            else:
                sentiment = "NEUTRAL"
                score = 0.0

            articles.append(
                {
                    "title": clean_title,
                    "published_at": pub_date,
                    "url": link,
                    "extracted_tickers": tickers,
                    "catalyst_sentiment": sentiment,
                    "sentiment_score": round(score, 2),
                }
            )

        logger.info(f"Successfully scraped {len(articles)} live Benzinga wire items.")
    except Exception as e:
        logger.error(f"Error fetching Benzinga news wire: {e}")

    return articles


def scan_and_trade_benzinga_catalysts(
    execute_paper: bool = False,
    max_stocks_to_evaluate: int = 8,
    broker: Optional[PaperBroker] = None,
) -> Dict[str, Any]:
    """
    Main Orchestrator:
    1. Ingests the latest Benzinga wire.
    2. Identifies all stocks with breaking news.
    3. Runs multi-agent committee deliberation on the top active stocks.
    4. Executes paper buying (on Bullish news + Committee Buy) or selling (on Bearish news).
    5. Saves full scan report to results/benzinga_live_scan_latest.json.
    """
    scan_timestamp = datetime.now(timezone.utc).isoformat()
    articles = fetch_top_benzinga_news_wire(limit=100)

    # Aggregate news by ticker
    ticker_catalysts: Dict[str, List[Dict[str, Any]]] = {}
    for a in articles:
        for t in a["extracted_tickers"]:
            ticker_catalysts.setdefault(t, []).append(a)

    logger.info(
        f"Identified {len(ticker_catalysts)} distinct stocks with breaking Benzinga news."
    )

    active_broker = broker or PaperBroker()
    current_equity = active_broker.state.get("total_equity", 144000.0)
    current_cash = active_broker.state.get("cash", 134000.0)
    open_positions = active_broker.state.get("open_positions", {})

    ranked_candidates = []
    for ticker, news_items in ticker_catalysts.items():
        # Score aggregated catalyst flow
        net_score = sum(n["sentiment_score"] for n in news_items)
        headline_count = len(news_items)
        latest_headline = news_items[0]["title"]
        latest_sentiment = news_items[0]["catalyst_sentiment"]

        ranked_candidates.append(
            {
                "ticker": ticker,
                "headline_count": headline_count,
                "net_catalyst_score": round(net_score, 2),
                "latest_sentiment": latest_sentiment,
                "latest_headline": latest_headline,
                "news_items": news_items,
            }
        )

    # Sort candidates by absolute catalyst score and count (highest impact first)
    ranked_candidates.sort(
        key=lambda x: (abs(x["net_catalyst_score"]), x["headline_count"]),
        reverse=True,
    )

    evaluated_stocks: List[Dict[str, Any]] = []
    trade_actions: List[Dict[str, Any]] = []

    for item in ranked_candidates[:max_stocks_to_evaluate]:
        ticker = item["ticker"]
        latest_headline = item["latest_headline"]
        cat_sentiment = item["latest_sentiment"]
        net_score = item["net_catalyst_score"]

        # Fetch spot quote
        quote = fetch_live_quote(ticker)
        price = float(quote.get("price", 0.0))
        chg_pct = float(quote.get("change_pct", 0.0))

        if price <= 0.0:
            continue

        # Deliberate with Multi-Agent Committee
        cat_titles = [n["title"] for n in item.get("news_items", [])]
        try:
            committee_res = convene_trading_committee(
                ticker=ticker, spot_price=price, catalyst_headlines=cat_titles
            )
            verdict = (
                committee_res.get("action_code")
                or committee_res.get("verdict")
                or "HOLD"
            )
            conviction = float(
                committee_res.get("consensus_conviction_pct")
                or committee_res.get("conviction")
                or 50.0
            )
            votes = committee_res.get("agent_testimonies", [])
        except Exception as ce:
            logger.debug(f"Committee deliberation note for {ticker}: {ce}")
            verdict = (
                "BUY"
                if net_score >= 0.50
                else ("SELL" if net_score <= -0.50 else "HOLD")
            )
            conviction = 65.0 if abs(net_score) >= 0.50 else 50.0
            votes = []

        # Trade Decision Logic:
        action = "HOLD"
        trade_reason = ""

        # Case 1: Bearish News on an existing position -> EXIT / SELL
        if ticker in open_positions and (
            cat_sentiment == "BEARISH" or verdict in ("SELL", "AVOID")
        ):
            action = "SELL"
            trade_reason = f"BENZINGA_BEARISH_CATALYST_EXIT: {latest_headline}"
            if execute_paper:
                exit_res = active_broker.execute_daily_signals(
                    signals_list=[
                        {
                            "ticker": ticker,
                            "signal": "SELL",
                            "confidence": 0.85,
                            "current_price": price,
                        }
                    ]
                )
                trade_actions.append(
                    {
                        "ticker": ticker,
                        "action": "SELL",
                        "price": price,
                        "result": exit_res,
                    }
                )
                logger.info(
                    f"🚨 [BENZINGA SELL EXECUTED] {ticker} @ ${price:.2f} due to bearish Benzinga catalyst."
                )

        # Case 2: Bullish News + Strong Catalyst + Committee Buy -> BUY
        elif (
            cat_sentiment == "BULLISH"
            and net_score >= 0.30
            and (
                verdict in ("BUY", "STRONG_BUY", "EXECUTE_BUY", "SCALE_IN")
                or (net_score >= 0.60 and conviction >= 60.0)
            )
        ):
            action = "BUY"
            trade_reason = f"BENZINGA_BREAKING_BULLISH_CATALYST: {latest_headline}"
            if execute_paper and current_cash > 5000.0:
                buy_res = active_broker.execute_daily_signals(
                    signals_list=[
                        {
                            "ticker": ticker,
                            "signal": "BUY",
                            "confidence": conviction / 100.0,
                            "current_price": price,
                            "take_profit": round(price * 1.05, 2),
                            "stop_loss": round(price * 0.975, 2),
                        }
                    ]
                )
                trade_actions.append(
                    {
                        "ticker": ticker,
                        "action": "BUY",
                        "price": price,
                        "result": buy_res,
                    }
                )
                logger.info(
                    f"⚡ [BENZINGA BUY EXECUTED] {ticker} @ ${price:.2f} based on breaking Benzinga catalyst."
                )

        evaluated_stocks.append(
            {
                "ticker": ticker,
                "current_price": price,
                "change_pct": chg_pct,
                "latest_headline": latest_headline,
                "benzinga_sentiment": cat_sentiment,
                "net_catalyst_score": net_score,
                "headline_count": item["headline_count"],
                "committee_verdict": verdict,
                "committee_conviction_pct": conviction,
                "trade_action": action,
                "trade_reason": trade_reason,
                "is_currently_held": ticker in open_positions,
            }
        )

    report = {
        "scan_timestamp": scan_timestamp,
        "total_articles_scraped": len(articles),
        "total_tickers_identified": len(ticker_catalysts),
        "top_evaluated_stocks": evaluated_stocks,
        "trade_actions_executed": trade_actions,
        "portfolio_equity": round(current_equity, 2),
        "portfolio_cash": round(current_cash, 2),
    }

    # Save to results/
    try:
        os.makedirs(os.path.dirname(BENZINGA_SCAN_FILE), exist_ok=True)
        with open(BENZINGA_SCAN_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved Benzinga live news scan report to {BENZINGA_SCAN_FILE}")
    except Exception as se:
        logger.warning(f"Could not save Benzinga scan report: {se}")

    return report


def get_latest_benzinga_scan_report() -> Dict[str, Any]:
    """Retrieves cached Benzinga news scan report."""
    if os.path.exists(BENZINGA_SCAN_FILE):
        try:
            with open(BENZINGA_SCAN_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}
