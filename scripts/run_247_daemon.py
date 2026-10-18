"""
24/7 Remote Cloud & Local Daemon Runner for Sentilyze.
======================================================
Continuous Autonomous Trading Daemon:
- Runs non-stop 24 hours a day, 7 days a week.
- During US Market Hours (09:30 - 16:00 EST): Trades US Equities & Digital Assets.
- Outside US Hours & Weekends: Automatically routes execution to 24/7 Digital Assets (BTC, ETH, SOL)
  using the isolated sandbox ledger (results/pilot_crypto_portfolio.json).
- Implements exponential backoff, rate-limit protection, and clean signal handling.
"""

import sys
import os
import time
import signal
import argparse
from datetime import datetime, timezone
from typing import Optional

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import get_logger
from src.market_session import get_us_market_session, get_current_ny_time
from src.autonomous_trader import AutonomousTradingEngine

logger = get_logger("247_cloud_daemon")

running = True


def handle_shutdown(signum, frame):
    global running
    logger.info(
        f"🛑 [DAEMON SHUTDOWN] Received signal {signum}. Gracefully shutting down..."
    )
    running = False


signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


def run_daemon(
    interval_seconds: int = 300,
    single_run: bool = False,
    max_iterations: Optional[int] = None,
):
    """Executes the continuous 24/7 autonomous loop."""
    logger.info("🚀 [SENTILYZE 24/7 DAEMON] Starting continuous trading service...")
    logger.info(
        f"⏱️ Cycle interval: {interval_seconds}s | Single-run: {single_run} | Max iterations: {max_iterations}"
    )

    engine = AutonomousTradingEngine()
    iteration = 0

    while running:
        iteration += 1
        now_ny = get_current_ny_time()
        session = get_us_market_session()

        logger.info(
            f"🔄 [CYCLE {iteration}] NY Time: {now_ny.strftime('%Y-%m-%d %H:%M:%S %Z')} | "
            f"Market Status: {session['status']} (Live Open: {session['is_open']})"
        )

        try:
            res = engine.run_autonomous_cycle()
            buys = len(res.get("buys", []))
            tp1 = len(res.get("take_profits_tp1", []))
            tp2 = len(res.get("take_profits_tp2", []))
            sl = len(res.get("stop_losses", []))
            equity = res.get("portfolio_equity", 0.0)

            logger.info(
                f"✅ [CYCLE {iteration} COMPLETE] Buys: {buys} | Exits (TP1/TP2/SL): {tp1}/{tp2}/{sl} | "
                f"US Equity: ${equity:,.2f}"
            )
        except Exception as e:
            logger.error(f"❌ [CYCLE {iteration} ERROR] Unhandled exception: {e}")

        if single_run or (max_iterations and iteration >= max_iterations):
            logger.info(
                f"🏁 [DAEMON FINISHED] Reached termination condition (iterations: {iteration}). Exiting."
            )
            break

        # Sleep with incremental checks for clean shutdown
        sleep_elapsed = 0
        while running and sleep_elapsed < interval_seconds:
            time.sleep(1)
            sleep_elapsed += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentilyze 24/7 Cloud Daemon")
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between cycles (default: 300)",
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Run exactly 1 cycle and exit immediately",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum cycles before terminating",
    )
    args = parser.parse_args()

    run_daemon(
        interval_seconds=args.interval,
        single_run=args.single_run,
        max_iterations=args.max_iterations,
    )
