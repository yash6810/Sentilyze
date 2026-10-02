"""
Autonomous Daily Execution Bridge for Sentilyze.
================================================
Bridges machine learning daily signals (results/daily_signals_latest.json)
into the institutional PaperBroker ledger with:
1. Anti-churn hysteresis and 3-day post-exit cooldown.
2. GICS sector diversification (max 1 position per sector).
3. Dynamic CPPI cushion protection and Quarter-Kelly position sizing.
4. Multi-tier progressive profit locks and bracket stops.

Python ONLY. Zero-GPU / CPU-only execution.
STRICT PORTFOLIO PRESERVATION: Live ledger is modified only during live runs.
Tests must provide custom portfolio_path or use tmp_path.
"""

import os
import json
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import get_logger
from src.paper_broker import PaperBroker, PORTFOLIO_FILE

logger = get_logger(__name__)

DEFAULT_SIGNALS_FILE = os.path.join("results", "daily_signals_latest.json")
DEFAULT_TRADES_FILE = os.path.join("results", "executed_trades.csv")


class DailyExecutionBridge:
    """Orchestrates autonomous transition from daily signals to active execution."""

    def __init__(
        self,
        signals_file: str = DEFAULT_SIGNALS_FILE,
        portfolio_file: str = PORTFOLIO_FILE,
        trades_file: str = DEFAULT_TRADES_FILE,
        dry_run: bool = False,
    ):
        self.signals_file = signals_file
        self.portfolio_file = portfolio_file
        self.trades_file = trades_file
        self.dry_run = dry_run
        self.broker = PaperBroker(
            portfolio_file=portfolio_file, trades_file=trades_file
        )

    def load_signals(self) -> List[Dict[str, Any]]:
        """Loads and parses the latest daily signals scan."""
        if not os.path.exists(self.signals_file):
            logger.warning(f"Signals file not found at {self.signals_file}")
            return []
        try:
            with open(self.signals_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            signals = data.get("signals", [])
            logger.info(
                f"Loaded {len(signals)} signals scanned at {data.get('scan_time', 'N/A')}"
            )
            return signals
        except Exception as e:
            logger.error(f"Failed to load daily signals: {e}")
            return []

    def get_actionable_signals(
        self, min_confidence: float = 0.55
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Categorizes actionable signals into BUYs, SELLs, and HOLDs."""
        signals = self.load_signals()
        buys = []
        sells = []
        holds = []

        for s in signals:
            sig = s.get("signal", "HOLD")
            conf = float(s.get("confidence", 0.50))
            if sig == "BUY" and conf >= min_confidence:
                buys.append(s)
            elif sig == "SELL":
                sells.append(s)
            else:
                holds.append(s)

        buys = sorted(buys, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        return {"buys": buys, "sells": sells, "holds": holds}

    def execute_daily_cycle(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes the daily cycle:
        1. Evaluates open positions for TP1, TP2, stop loss, or model sell.
        2. Opens new diversified positions from top buy candidates.
        """
        actionable = self.get_actionable_signals()
        all_signals = actionable["buys"] + actionable["sells"] + actionable["holds"]

        if not all_signals:
            logger.warning("No signals available to execute.")
            return {
                "success": False,
                "reason": "NO_SIGNALS",
                "executed_actions": {},
            }

        if self.dry_run:
            logger.info("Executing in DRY-RUN mode. Portfolio will NOT be modified.")
            os.environ["SENTILYZE_DRY_RUN"] = "1"
            import tempfile

            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_p = os.path.join(tmp_dir, "mock_portfolio.json")
                    tmp_t = os.path.join(tmp_dir, "mock_trades.csv")
                    with open(tmp_p, "w", encoding="utf-8") as f:
                        json.dump(self.broker.state, f, indent=2)
                    sim_broker = PaperBroker(portfolio_file=tmp_p, trades_file=tmp_t)
                    sim_res = sim_broker.execute_daily_signals(
                        signals_list=all_signals, date_str=target_date
                    )
                    return {
                        "success": True,
                        "dry_run": True,
                        "simulated_actions": sim_res,
                        "portfolio_state": sim_broker.get_portfolio_summary(),
                    }
            finally:
                os.environ.pop("SENTILYZE_DRY_RUN", None)

        # Live Execution
        res = self.broker.execute_daily_signals(
            signals_list=all_signals, date_str=target_date
        )
        summary = self.broker.get_portfolio_summary()

        logger.info(
            f"Daily cycle executed successfully. Total Equity: ${summary.get('total_equity', 0):,.2f} | "
            f"Cash: ${summary.get('cash', 0):,.2f} | Open Positions: {summary.get('open_positions_count', 0)}"
        )

        return {
            "success": True,
            "dry_run": False,
            "executed_actions": res,
            "portfolio_state": summary,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Sentilyze Autonomous Daily Execution Bridge"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate execution without modifying ledger",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Execution date in YYYY-MM-DD format (defaults to UTC today)",
    )
    args = parser.parse_args()

    bridge = DailyExecutionBridge(dry_run=args.dry_run)
    result = bridge.execute_daily_cycle(target_date=args.date)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
