"""
Autonomous Benzinga Breaking News Watchdog & Discord Dispatcher.
=================================================================
Runs continuously in the background (or on schedule):
1. Polls the live Benzinga breaking news stream every N seconds.
2. Deduplicates incoming stories to prevent spam / duplicate alerts.
3. Classifies Tier-1 breaking catalysts (deals >= $1B, double beats, FDA, penalties).
4. Deliberates with the Multi-Agent Council.
5. Dispatches formatted Rich Discord Embeds with target prices and catalyst tags.
6. Automatically triggers PaperBroker orders if conviction exceeds threshold.
"""

import os
import json
import time
import hashlib
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set

import requests
from src.utils import get_logger
from src.benzinga_news_trader import (
    fetch_top_benzinga_news_wire,
    scan_and_trade_benzinga_catalysts,
)
from src.paper_broker import PaperBroker

logger = get_logger("benzinga_watchdog")

WATCHDOG_STATE_FILE = os.path.join("results", "benzinga_watchdog_state.json")


class BenzingaNewsWatchdog:
    """
    Autonomous 5-minute background watchdog monitoring Benzinga catalyst wire.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        state_file: Optional[str] = None,
    ):
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        self.state_file = state_file or WATCHDOG_STATE_FILE
        self.seen_headline_hashes: Set[str] = self._load_seen_hashes()
        self._daemon_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.last_poll_time: Optional[str] = None
        self.total_alerts_dispatched: int = 0

    def _load_seen_hashes(self) -> Set[str]:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return set(data.get("seen_hashes", []))
            except Exception:
                pass
        return set()

    def _save_seen_hashes(self):
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, "w", encoding="utf-8") as f:
                # Keep last 1,000 hashes to prevent unbounded growth
                recent_hashes = list(self.seen_headline_hashes)[-1000:]
                json.dump(
                    {
                        "last_poll_time": self.last_poll_time,
                        "total_alerts_dispatched": self.total_alerts_dispatched,
                        "seen_hashes": recent_hashes,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.debug(f"Watchdog state save note: {e}")

    def hash_headline(self, title: str) -> str:
        return hashlib.sha256(title.strip().lower().encode("utf-8")).hexdigest()[:16]

    def poll_and_dispatch(
        self,
        auto_trade: bool = False,
        min_catalyst_score: float = 0.40,
        broker: Optional[PaperBroker] = None,
    ) -> Dict[str, Any]:
        """
        Executes single watchdog polling cycle:
        - Detects newly published breaking headlines.
        - Filters for material catalysts.
        - Dispatches Discord notification and optional paper trade.
        """
        self.last_poll_time = datetime.now(timezone.utc).isoformat()
        articles = fetch_top_benzinga_news_wire(limit=60)
        new_catalysts = []

        for a in articles:
            h_hash = self.hash_headline(a["title"])
            if h_hash in self.seen_headline_hashes:
                continue

            self.seen_headline_hashes.add(h_hash)

            # Check if article has identified tickers and material catalyst score
            if (
                a["extracted_tickers"]
                and abs(a["sentiment_score"]) >= min_catalyst_score
            ):
                new_catalysts.append(a)

        self._save_seen_hashes()

        dispatched_alerts = []
        executed_trades = []

        if new_catalysts:
            logger.info(
                f"🚨 [BENZINGA WATCHDOG] Discovered {len(new_catalysts)} new material breaking catalysts!"
            )

            # Evaluate through Council and PaperBroker
            active_broker = broker or PaperBroker()
            scan_rep = scan_and_trade_benzinga_catalysts(
                execute_paper=auto_trade,
                max_stocks_to_evaluate=len(new_catalysts),
                broker=active_broker,
            )
            executed_trades = scan_rep.get("trade_actions_executed", [])

            for item in scan_rep.get("top_evaluated_stocks", []):
                # Send Discord Rich Embed
                if self.webhook_url:
                    embed_success = self.send_discord_catalyst_alert(item)
                    if embed_success:
                        self.total_alerts_dispatched += 1
                        dispatched_alerts.append(item["ticker"])

        return {
            "status": "COMPLETED",
            "poll_time": self.last_poll_time,
            "articles_scanned": len(articles),
            "new_catalysts_detected": len(new_catalysts),
            "alerts_dispatched": dispatched_alerts,
            "trades_executed": executed_trades,
        }

    def send_discord_catalyst_alert(self, evaluated_stock: Dict[str, Any]) -> bool:
        """Constructs and posts rich embed card to Discord."""
        if not self.webhook_url:
            return False

        ticker = evaluated_stock["ticker"]
        price = evaluated_stock["current_price"]
        sentiment = evaluated_stock["benzinga_sentiment"]
        score = evaluated_stock["net_catalyst_score"]
        verdict = evaluated_stock["committee_verdict"]
        conviction = evaluated_stock["committee_conviction_pct"]
        action = evaluated_stock["trade_action"]
        headline = evaluated_stock["latest_headline"]

        color = (
            0x10B981
            if sentiment == "BULLISH"
            else (0xEF4444 if sentiment == "BEARISH" else 0xF59E0B)
        )
        icon = (
            "🟢"
            if sentiment == "BULLISH"
            else ("🔴" if sentiment == "BEARISH" else "🟡")
        )

        embed = {
            "title": f"⚡ Benzinga Wire Alert: {ticker} (${price:,.2f}) {icon}",
            "description": (
                f"**Headline:** {headline}\n\n"
                f"• **Catalyst Sentiment:** `{sentiment}` (Score: `{score:+.2f}`)\n"
                f"• **Council Verdict:** `{verdict}` ({conviction:.1f}% Conviction)\n"
                f"• **Trading Action:** `{action}`\n"
            ),
            "color": color,
            "footer": {
                "text": "Sentilyze Benzinga Autonomous Watchdog • Zero-GPU Stream"
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            res = requests.post(self.webhook_url, json={"embeds": [embed]}, timeout=6)
            return res.status_code in (200, 204)
        except Exception as e:
            logger.debug(f"Discord webhook dispatch notice: {e}")
            return False

    def start_background_watchdog(
        self, interval_seconds: int = 300, auto_trade: bool = False
    ):
        """Launches continuous background watchdog thread."""
        if self._daemon_thread and self._daemon_thread.is_alive():
            logger.info("Benzinga background watchdog is already running.")
            return

        self._stop_event.clear()

        def _run_loop():
            logger.info(
                f"🚀 [BENZINGA WATCHDOG] Daemon thread started (Interval: {interval_seconds}s)..."
            )
            while not self._stop_event.is_set():
                try:
                    self.poll_and_dispatch(auto_trade=auto_trade)
                except Exception as e:
                    logger.error(f"Error in watchdog poll: {e}")
                self._stop_event.wait(interval_seconds)

        self._daemon_thread = threading.Thread(
            target=_run_loop, daemon=True, name="BenzingaWatchdogDaemon"
        )
        self._daemon_thread.start()

    def stop_background_watchdog(self):
        """Stops the continuous watchdog thread."""
        self._stop_event.set()
        if self._daemon_thread:
            self._daemon_thread.join(timeout=3)
        logger.info("🛑 [BENZINGA WATCHDOG] Daemon stopped.")


_WATCHDOG_INSTANCE: Optional[BenzingaNewsWatchdog] = None


def get_benzinga_watchdog() -> BenzingaNewsWatchdog:
    global _WATCHDOG_INSTANCE
    if _WATCHDOG_INSTANCE is None:
        _WATCHDOG_INSTANCE = BenzingaNewsWatchdog()
    return _WATCHDOG_INSTANCE
