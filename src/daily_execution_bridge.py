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
        enable_sector_quota: Optional[bool] = None,
    ):
        self.signals_file = signals_file
        self.portfolio_file = portfolio_file
        self.trades_file = trades_file
        self.dry_run = dry_run
        self.enable_sector_quota = (
            enable_sector_quota
            if enable_sector_quota is not None
            else (signals_file == DEFAULT_SIGNALS_FILE)
        )
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
        """
        Categorizes actionable signals into BUYs, SELLs, and HOLDs with Multi-Sector Quota.
        1. Loads model daily signals (BUY, SELL, HOLD).
        2. Incorporates high-conviction breaking catalyst BUYs from Benzinga wire (if enable_sector_quota is True).
        3. For unoccupied sectors, selects the highest-conviction bullish candidate to prevent cash starvation.
        """
        from src.cross_asset_pooling import get_sector_for_ticker

        signals = self.load_signals()
        buys = []
        sells = []
        holds = []

        if not signals:
            return {"buys": [], "sells": [], "holds": []}

        open_positions = self.broker.state.get("open_positions", {})
        occupied_sectors = {get_sector_for_ticker(t) for t in open_positions.keys()}

        # 1. Existing BUY/SELL signals
        for s in signals:
            sig = s.get("signal", "HOLD")
            conf = float(s.get("confidence", 0.50))
            if sig == "BUY" and conf >= min_confidence:
                buys.append(s)
            elif sig == "SELL":
                sells.append(s)
            else:
                holds.append(s)

        if not self.enable_sector_quota:
            buys = sorted(
                buys, key=lambda x: float(x.get("confidence", 0.0)), reverse=True
            )
            return {"buys": buys, "sells": sells, "holds": holds}

        # 2. Add approved Benzinga Catalyst BUYs if sector is unoccupied
        bz_file = os.path.join("results", "benzinga_live_scan_latest.json")
        if os.path.exists(bz_file):
            try:
                with open(bz_file, "r", encoding="utf-8") as bf:
                    bz_data = json.load(bf)
                for item in bz_data.get("top_evaluated_stocks", []):
                    tk = item.get("ticker")
                    v = item.get("committee_verdict", "HOLD")
                    c = float(item.get("committee_conviction_pct", 50.0)) / 100.0
                    p = float(item.get("current_price", 0.0))
                    sec = get_sector_for_ticker(tk)
                    if (
                        v in ("BUY", "EXECUTE_BUY", "STRONG_BUY")
                        and c >= min_confidence
                        and p > 0
                        and tk not in open_positions
                        and sec not in occupied_sectors
                        and not any(b["ticker"] == tk for b in buys)
                    ):
                        logger.info(
                            f"📰 [Catalyst Quota] Added {tk} ({sec}, Conf: {c:.1%}) to candidate pool"
                        )
                        buys.append(
                            {
                                "ticker": tk,
                                "signal": "BUY",
                                "confidence": c,
                                "current_price": p,
                                "take_profit": round(p * 1.06, 2),
                                "stop_loss": round(p * 0.975, 2),
                                "regime": "▲ BULLISH (Benzinga Catalyst + Council)",
                                "source": "BENZINGA_CATALYST_COUNCIL",
                            }
                        )
            except Exception as bze:
                logger.debug(f"Benzinga catalyst bridge notice: {bze}")

        # 3. Multi-Sector Quota: For unoccupied sectors, scan holds for high-conviction bullish leaders
        sector_candidates: Dict[str, List[Dict[str, Any]]] = {}
        for s in holds:
            tk = s.get("ticker", "")
            conf = float(s.get("confidence", 0.0))
            regime = str(s.get("regime", ""))
            price = float(s.get("current_price", 0.0))
            sec = get_sector_for_ticker(tk)

            if (
                sec not in occupied_sectors
                and sec not in ("General", "Unknown", "General S&P 100")
                and conf >= 0.58
                and "BULLISH" in regime.upper()
                and price > 0
                and tk not in open_positions
                and not any(b["ticker"] == tk for b in buys)
            ):
                sector_candidates.setdefault(sec, []).append(s)

        # For each unoccupied sector, add the top-1 ranked candidate if buys don't already have one
        for sec, cands in sector_candidates.items():
            if not any(get_sector_for_ticker(b["ticker"]) == sec for b in buys):
                cands.sort(key=lambda x: float(x.get("confidence", 0)), reverse=True)
                top_cand = cands[0].copy()
                top_cand["signal"] = "BUY"
                top_cand["source"] = "MULTI_SECTOR_QUOTA_SCANNER"
                logger.info(
                    f"🌐 [Sector Quota] Selected {top_cand['ticker']} for unoccupied sector '{sec}' (Conf: {top_cand.get('confidence', 0):.1%})"
                )
                buys.append(top_cand)

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
