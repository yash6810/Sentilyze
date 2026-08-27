"""
Autonomous Live Trading & News Intelligence Engine for Sentilyze.
Institutional 24/7 Autonomous Execution:
1. Multi-Source Live News Ingestion (Google RSS + Finnhub + Marketaux + Yahoo)
2. 4-Agent Trading Committee Deliberation (Technicals, FinBERT, Valuation, Risk Officer)
3. Kelly Criterion Capital Allocation & Dynamic Leverage
4. 2-Stage Staged Profit Scale-Out (50% @ TP1, Trailing Breakeven Stop, 50% @ TP2)
5. Capital Preservation & Emergency Macro Volatility / Forensic Vetoes
"""

import os
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import requests
import pandas as pd

from src.utils import get_logger
from src.paper_broker import PaperBroker
from src.agent_committee import convene_trading_committee, execute_committee_order
from src.realtime_tracker import fetch_universe_live_quotes, fetch_live_quote
from src.data_ingestion import get_news
from src.alerts import (
    send_discord_execution_alert,
    send_discord_committee_alert,
    send_discord_social_spike_alert,
)

logger = get_logger(__name__)

STOCKS_FILE = "stocks.txt"
AUTONOMOUS_LOG_FILE = os.path.join("results", "autonomous_execution_log.json")


def load_universe_tickers() -> List[str]:
    """Loads universe of tickers from stocks.txt."""
    if os.path.exists(STOCKS_FILE):
        with open(STOCKS_FILE, "r") as f:
            tickers = [
                line.strip().upper()
                for line in f
                if line.strip() and not line.startswith("#")
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


class AutonomousTradingEngine:
    """
    Autonomous Execution Engine that integrates Live News Ingestion,
    4-Agent Committee Deliberation, and 2-Stage Profit Scaling.
    """

    def __init__(self, broker: Optional[PaperBroker] = None):
        self.broker = broker or PaperBroker()
        self.tickers = load_universe_tickers()

    def dispatch_discord_alert(
        self, title: str, description: str, color: int = 0x00D4AA
    ):
        """Dispatches an institutional execution alert to Discord Webhook if configured."""
        webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        if not webhook_url or not webhook_url.startswith("http"):
            return

        payload = {
            "embeds": [
                {
                    "title": title,
                    "description": description,
                    "color": color,
                    "footer": {"text": "Sentilyze Autonomous Trading Agent Desk"},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ]
        }
        try:
            requests.post(webhook_url, json=payload, timeout=5)
        except Exception as e:
            logger.debug(f"Discord webhook dispatch notice: {e}")

    def run_autonomous_cycle(
        self,
        candidate_tickers: Optional[List[str]] = None,
        max_concurrent_positions: int = 4,
    ) -> Dict[str, Any]:
        """
        Executes one full autonomous decision and execution cycle.
        """
        start_time = time.time()
        tickers_to_scan = candidate_tickers or self.tickers
        now_str = datetime.now(timezone.utc).isoformat()
        date_str = now_str[:10]

        logger.info(
            f"🤖 [AUTONOMOUS TRADER] Starting execution cycle across {len(tickers_to_scan)} universe assets..."
        )

        # 1. Fetch Parallel Live Quotes
        quotes_map = fetch_universe_live_quotes(tickers_to_scan)
        portfolio_summary = self.broker.get_portfolio_summary()

        executed_actions = {
            "timestamp": now_str,
            "buys": [],
            "take_profits_tp1": [],
            "take_profits_tp2": [],
            "stop_losses": [],
            "veto_exits": [],
            "committee_resolutions": {},
            "portfolio_equity": portfolio_summary["total_equity"],
            "cash_balance": portfolio_summary["cash"],
            "unrealized_pnl": portfolio_summary["unrealized_pnl"],
        }

        # 2. Phase A: Manage & Audit Open Positions (Profit Taking / Trailing Stops)
        open_positions = list(self.broker.state.get("open_positions", {}).keys())
        for ticker in open_positions:
            pos = self.broker.state["open_positions"].get(ticker)
            if not pos:
                continue

            q = quotes_map.get(ticker) or fetch_live_quote(ticker)
            spot_price = float(q.get("price", 0))
            if spot_price <= 0:
                continue

            pos["current_price"] = spot_price
            shares = pos["shares"]
            entry_price = pos["entry_price"]
            tp1_target = pos.get("tp1_target", entry_price * 1.05)
            tp2_target = pos.get("tp2_target", entry_price * 1.10)
            sl_target = pos.get("sl_target", entry_price * 0.96)
            scaled_out = pos.get("scaled_out", False)

            # Check Stage 1 Scale-Out (+2.5 ATR)
            if not scaled_out and spot_price >= tp1_target:
                half_shares = max(1, shares // 2)
                proceeds = float(half_shares * spot_price)
                cost_basis = float(half_shares * entry_price)
                pnl = float(proceeds - cost_basis)
                ret_pct = float((spot_price - entry_price) / entry_price * 100.0)

                self.broker.state["cash"] += proceeds
                self.broker.state["realized_pnl"] += pnl
                pos["shares"] = shares - half_shares
                pos["scaled_out"] = True
                pos["sl_target"] = round(entry_price * 1.002, 2)  # Breakeven + 0.2%

                tp1_record = {
                    "ticker": ticker,
                    "shares_sold": half_shares,
                    "remaining_shares": pos["shares"],
                    "exit_price": spot_price,
                    "pnl": round(pnl, 2),
                    "return_pct": round(ret_pct, 2),
                    "action": "TP1_PROFIT_LOCK_50PCT",
                }
                executed_actions["take_profits_tp1"].append(tp1_record)
                logger.info(
                    f"💰 [TP1 PROFIT LOCK] Sold {half_shares} shares of {ticker} @ ${spot_price:.2f} | Realized PnL: ${pnl:+,.2f} ({ret_pct:+.2f}%) | Stop moved to ${pos['sl_target']:.2f}"
                )
                send_discord_execution_alert(
                    {
                        "action": "SELL",
                        "stage": "TP1_PROFIT_LOCK",
                        "ticker": ticker,
                        "price": spot_price,
                        "entry_price": entry_price,
                        "shares": pos["shares"],
                        "realized_pnl": pnl,
                        "tp2": pos.get("tp2_target", 0.0),
                    }
                )

            # Check Stage 2 Runner Exit (+4.5 ATR)
            elif spot_price >= tp2_target:
                proceeds = float(pos["shares"] * spot_price)
                cost_basis = float(pos["shares"] * entry_price)
                pnl = float(proceeds - cost_basis)
                ret_pct = float((spot_price - entry_price) / entry_price * 100.0)

                self.broker.state["cash"] += proceeds
                self.broker.state["realized_pnl"] += pnl
                self.broker.state["total_trades"] += 1
                self.broker.state["winning_trades"] += 1

                trade_record = {
                    "ticker": ticker,
                    "shares": pos["shares"],
                    "entry_price": entry_price,
                    "exit_price": spot_price,
                    "entry_date": pos["entry_date"],
                    "exit_date": date_str,
                    "pnl": round(pnl, 2),
                    "return_pct": round(ret_pct, 2),
                    "reason": "TP2_RUNNER_EXIT",
                }
                self.broker.state["closed_trades"].append(trade_record)
                del self.broker.state["open_positions"][ticker]
                executed_actions["take_profits_tp2"].append(trade_record)
                logger.info(
                    f"🎯 [TP2 RUNNER EXIT] Closed runner for {ticker} @ ${spot_price:.2f} | Realized PnL: ${pnl:+,.2f} ({ret_pct:+.2f}%)"
                )
                send_discord_execution_alert(
                    {
                        "action": "SELL",
                        "stage": "TP2_RUNNER_EXIT",
                        "ticker": ticker,
                        "price": spot_price,
                        "shares": trade_record["shares"],
                        "realized_pnl": pnl,
                        "return_pct": ret_pct,
                    }
                )

            # Check Stop-Loss / Breakeven Exit
            elif spot_price <= sl_target:
                proceeds = float(pos["shares"] * spot_price)
                cost_basis = float(pos["shares"] * entry_price)
                pnl = float(proceeds - cost_basis)
                ret_pct = float((spot_price - entry_price) / entry_price * 100.0)

                self.broker.state["cash"] += proceeds
                self.broker.state["realized_pnl"] += pnl
                self.broker.state["total_trades"] += 1
                if pnl > 0 or scaled_out:
                    self.broker.state["winning_trades"] += 1
                else:
                    self.broker.state["losing_trades"] += 1

                reason = "BREAK_EVEN_TRAIL" if scaled_out else "STOP_LOSS"
                trade_record = {
                    "ticker": ticker,
                    "shares": pos["shares"],
                    "entry_price": entry_price,
                    "exit_price": spot_price,
                    "entry_date": pos["entry_date"],
                    "exit_date": date_str,
                    "pnl": round(pnl, 2),
                    "return_pct": round(ret_pct, 2),
                    "reason": reason,
                }
                self.broker.state["closed_trades"].append(trade_record)
                del self.broker.state["open_positions"][ticker]
                executed_actions["stop_losses"].append(trade_record)
                logger.info(
                    f"🛡️ [{reason}] Closed position for {ticker} @ ${spot_price:.2f} | Realized PnL: ${pnl:+,.2f} ({ret_pct:+.2f}%)"
                )
                send_discord_execution_alert(
                    {
                        "action": "SELL",
                        "stage": reason,
                        "ticker": ticker,
                        "price": spot_price,
                        "shares": trade_record["shares"],
                        "realized_pnl": pnl,
                    }
                )

        # 3. Phase B: Scan Universe for Committee Buy Opportunities
        curr_open_count = len(self.broker.state.get("open_positions", {}))
        available_slots = max(0, max_concurrent_positions - curr_open_count)
        cash_available = self.broker.state.get("cash", 0.0)

        if available_slots > 0 and cash_available > 5000.0:
            unheld_tickers = [
                t
                for t in tickers_to_scan
                if t not in self.broker.state.get("open_positions", {})
            ]

            deliberations = []
            for t in unheld_tickers:
                try:
                    # Ingest fresh live news silently
                    get_news(t, use_cache=True)
                    delib = convene_trading_committee(t, save_resolution=True)
                    deliberations.append((t, delib))
                    executed_actions["committee_resolutions"][t] = delib[
                        "final_resolution"
                    ]
                except Exception as e:
                    logger.debug(f"Committee scan error for {t}: {e}")

            # Sort candidate opportunities by CRO consensus conviction
            buy_candidates = [
                (t, d)
                for t, d in deliberations
                if d.get("action_code") in ["EXECUTE_BUY", "SCALE_IN"]
            ]
            buy_candidates.sort(
                key=lambda x: x[1].get("consensus_conviction_pct", 0.0),
                reverse=True,
            )

            # Execute entries into Top candidate setups up to available slots
            for t, delib in buy_candidates[:available_slots]:
                spot_price = float(delib.get("spot_price", 0))
                if spot_price <= 0:
                    continue

                order_res = execute_committee_order(
                    ticker=t, deliberation=delib, broker=self.broker
                )
                if order_res.get("success"):
                    executed_actions["buys"].append(order_res)
                    logger.info(
                        f"🚀 [AUTONOMOUS BUY] Executed {order_res.get('shares')} shares of {t} @ ${spot_price:.2f} (Verdict: {delib['final_resolution']})"
                    )
                    send_discord_execution_alert(
                        {
                            "action": "BUY",
                            "stage": "ENTRY",
                            "ticker": t,
                            "price": spot_price,
                            "shares": order_res.get("shares"),
                            "kelly_pct": delib.get("cro_signoff", {}).get(
                                "approved_kelly_pct", 8.0
                            ),
                            "tp1": delib.get("tp1_target", spot_price * 1.06),
                            "tp2": delib.get("tp2_target", spot_price * 1.12),
                            "stop_loss": delib.get(
                                "stop_loss_target", spot_price * 0.965
                            ),
                        }
                    )
                    send_discord_committee_alert(delib)

        # 4. Phase C: Update Portfolio Summary Ledger
        self.broker._recalculate_metrics(date_str, now_str)
        self.broker._save()

        elapsed = round(time.time() - start_time, 2)
        executed_actions["elapsed_seconds"] = elapsed
        logger.info(
            f"✅ [AUTONOMOUS TRADER] Completed cycle in {elapsed}s. Buys: {len(executed_actions['buys'])}, Exits: {len(executed_actions['take_profits_tp1']) + len(executed_actions['take_profits_tp2']) + len(executed_actions['stop_losses'])}"
        )

        # Save cycle log
        os.makedirs(os.path.dirname(AUTONOMOUS_LOG_FILE), exist_ok=True)
        with open(AUTONOMOUS_LOG_FILE, "w") as f:
            json.dump(executed_actions, f, indent=2)

        return executed_actions


def run_autonomous_daemon(interval_seconds: int = 300):
    """Runs the Autonomous Trading Engine continuously on an interval."""
    engine = AutonomousTradingEngine()
    logger.info(
        f"🤖 [AUTONOMOUS DAEMON] Launched. Polling market every {interval_seconds}s..."
    )
    while True:
        try:
            engine.run_autonomous_cycle()
        except Exception as e:
            logger.error(f"Error in autonomous daemon cycle: {e}")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Autonomous Live Trading & News Intelligence Agent Desk"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run single autonomous decision & execution cycle",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run 24/7 continuous autonomous trading daemon",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Daemon cycle interval in seconds (default: 300s / 5 min)",
    )

    args = parser.parse_args()
    engine = AutonomousTradingEngine()

    if args.daemon:
        run_autonomous_daemon(interval_seconds=args.interval)
    else:
        summary = engine.run_autonomous_cycle()
        print(json.dumps(summary, indent=2))
