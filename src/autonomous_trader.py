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
    send_discord_premarket_briefing,
)

logger = get_logger(__name__)

STOCKS_FILE = "stocks.txt"
AUTONOMOUS_LOG_FILE = os.path.join("results", "autonomous_execution_log.json")
LOCK_FILE = os.path.join("results", ".autonomous_trader.lock")
KILL_SWITCH_FILE = os.path.join("results", "KILL_SWITCH.flag")


def is_kill_switch_active() -> bool:
    """
    Task 7: Master Kill Switch Check.
    Returns True if SENTILYZE_KILL_SWITCH environment variable is enabled or results/KILL_SWITCH.flag exists.
    """
    env_kill = os.getenv("SENTILYZE_KILL_SWITCH", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    file_kill = os.path.exists(KILL_SWITCH_FILE)
    return env_kill or file_kill


def check_daily_loss_circuit_breaker(
    portfolio_summary: Dict[str, Any], max_daily_loss_pct: float = 3.0
) -> bool:
    """
    Task 8: Independent Max-Daily-Loss Circuit Breaker.
    Compares current total equity against start-of-day equity baseline.
    Returns True if intraday drawdown / loss exceeds max_daily_loss_pct (default: 3.0%).
    """
    daily_return_pct = float(portfolio_summary.get("daily_return_pct", 0.0))
    if daily_return_pct <= -abs(max_daily_loss_pct):
        return True

    # Robust fallback based on start-of-day equity vs total equity:
    start_equity = float(
        portfolio_summary.get(
            "start_of_day_equity",
            portfolio_summary.get("total_equity", 100000.0),
        )
    )
    current_equity = float(portfolio_summary.get("total_equity", start_equity))
    if start_equity > 0:
        intraday_return_pct = ((current_equity - start_equity) / start_equity) * 100.0
        return intraday_return_pct <= -abs(max_daily_loss_pct)

    return False


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
        Executes one full autonomous decision and execution cycle with:
        - Task 6: Idempotency lock guard
        - Task 7: Master Kill Switch check
        - Task 8: Daily loss & position size circuit breakers
        - Task 9: Unhandled exception alerting
        """
        # 1. Idempotency Lock Check (Task 6)
        if os.path.exists(LOCK_FILE):
            try:
                with open(LOCK_FILE, "r") as f:
                    lock_data = json.load(f)
                lock_timestamp = float(lock_data.get("timestamp", 0))
                age_seconds = time.time() - lock_timestamp
                if age_seconds < 600:  # Lock is active (< 10 minutes)
                    logger.warning(
                        f"🔒 [IDEMPOTENCY LOCK] Active trading cycle in progress (PID {lock_data.get('pid')}, age: {int(age_seconds)}s). Skipping overlapping cycle."
                    )
                    return {
                        "status": "SKIPPED_LOCKED",
                        "message": "Autonomous cycle already in progress",
                        "lock_pid": lock_data.get("pid"),
                    }
                else:
                    logger.warning(
                        f"⚠️ [STALE LOCK DETECTED] Lock is {int(age_seconds)}s old (>600s). Overriding stale lock."
                    )
            except Exception as e:
                logger.debug(f"Error reading lock file: {e}")

        # Acquire lock
        os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
        try:
            with open(LOCK_FILE, "w") as f:
                json.dump(
                    {
                        "pid": os.getpid(),
                        "timestamp": time.time(),
                        "iso": datetime.now(timezone.utc).isoformat(),
                    },
                    f,
                )
        except Exception as e:
            logger.debug(f"Could not persist lock file: {e}")

        try:
            return self._execute_cycle_body(
                candidate_tickers=candidate_tickers,
                max_concurrent_positions=max_concurrent_positions,
            )
        except Exception as e:
            # 2. Unhandled Exception Alerting (Task 9)
            logger.critical(f"💥 [CRITICAL TRADING LOOP EXCEPTION] {e}", exc_info=True)
            self.dispatch_discord_alert(
                title="🚨 CRITICAL UNHANDLED EXCEPTION IN TRADING LOOP",
                description=f"Autonomous trading loop encountered a fatal exception:\n```{str(e)[:500]}```",
                color=0xFF0000,
            )
            return {
                "status": "ERROR",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        finally:
            # Release lock in all circumstances
            if os.path.exists(LOCK_FILE):
                try:
                    os.remove(LOCK_FILE)
                except Exception:
                    pass

    def _execute_cycle_body(
        self,
        candidate_tickers: Optional[List[str]] = None,
        max_concurrent_positions: int = 4,
    ) -> Dict[str, Any]:
        """Core cycle execution body."""
        start_time = time.time()
        tickers_to_scan = candidate_tickers or self.tickers
        now_str = datetime.now(timezone.utc).isoformat()
        date_str = now_str[:10]

        from src.market_session import get_us_market_session

        market_session = get_us_market_session()
        logger.info(
            f"🏛️ [US MARKET STATUS] {market_session['status']} | EDT: {market_session['time_edt']} | "
            f"Live Open: {market_session['is_open']} | UTC: {market_session['utc_time']}"
        )

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
            "portfolio_equity": portfolio_summary.get("total_equity", 100000.0),
            "cash_balance": portfolio_summary.get("cash", 100000.0),
            "unrealized_pnl": portfolio_summary.get("unrealized_pnl", 0.0),
            "kill_switch_active": is_kill_switch_active(),
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
            shares = int(pos.get("shares", 0))
            entry_price = float(pos.get("entry_price") or spot_price or 1.0)
            denom = entry_price if entry_price > 0 else 1.0
            tp1_target = float(pos.get("tp1_target", entry_price * 1.05))
            tp2_target = float(pos.get("tp2_target", entry_price * 1.10))
            sl_target = float(pos.get("sl_target", entry_price * 0.96))
            stage0_taken = bool(pos.get("stage0_taken", False))
            scaled_out = bool(pos.get("scaled_out", False))

            # 🛡️ High-Water Mark & Dynamic Trailing Profit Shield
            high_water = max(pos.get("high_water_mark", entry_price), spot_price)
            pos["high_water_mark"] = high_water
            unrealized_gain_pct = (spot_price - entry_price) / denom * 100.0
            max_gain_pct = (high_water - entry_price) / denom * 100.0

            # Level 1: Breakeven Shield (Once in profit >= +1.2%, never allow trade to go red)
            if (
                not scaled_out
                and max_gain_pct >= 1.2
                and sl_target < entry_price * 1.002
            ):
                new_sl = round(entry_price * 1.002, 2)
                pos["sl_target"] = new_sl
                sl_target = new_sl
                logger.info(
                    f"🛡️ [BREAKEVEN SHIELD] {ticker} gained +{max_gain_pct:.2f}%. Stop trailed up to Breakeven (${new_sl:.2f})"
                )

            # Level 2: Trailing Profit Lock (If gained >= +2.5%, lock in >= +1.0% profit)
            if max_gain_pct >= 2.5 and sl_target < entry_price * 1.01:
                new_sl = round(entry_price * 1.01, 2)
                pos["sl_target"] = new_sl
                sl_target = new_sl
                logger.info(
                    f"🔒 [PROFIT LOCK TIER 1] {ticker} peaked at +{max_gain_pct:.2f}%. Stop trailed to lock +1.0% profit (${new_sl:.2f})"
                )

            # Level 3: Trailing Profit Lock (If gained >= +4.0%, lock in >= +2.0% profit)
            if max_gain_pct >= 4.0 and sl_target < entry_price * 1.02:
                new_sl = round(entry_price * 1.02, 2)
                pos["sl_target"] = new_sl
                sl_target = new_sl
                logger.info(
                    f"🔒 [PROFIT LOCK TIER 2] {ticker} peaked at +{max_gain_pct:.2f}%. Stop trailed to lock +2.0% profit (${new_sl:.2f})"
                )

            # ⚡ Check Stage 0 Quick Profit Micro-Harvest (+1.5% Gain)
            if not stage0_taken and spot_price >= entry_price * 1.015 and shares >= 2:
                third_shares = max(1, shares // 3)
                proceeds = float(third_shares * spot_price)
                cost_basis = float(third_shares * entry_price)
                pnl = float(proceeds - cost_basis)
                ret_pct = float((spot_price - entry_price) / denom * 100.0)

                self.broker.state["cash"] += proceeds
                self.broker.state["realized_pnl"] += pnl
                pos["shares"] = shares - third_shares
                pos["stage0_taken"] = True
                pos["sl_target"] = max(
                    pos.get("sl_target", 0.0), round(entry_price * 1.002, 2)
                )

                tp0_record = {
                    "ticker": ticker,
                    "shares_sold": third_shares,
                    "remaining_shares": pos["shares"],
                    "exit_price": spot_price,
                    "pnl": round(pnl, 2),
                    "return_pct": round(ret_pct, 2),
                    "action": "STAGE0_MICRO_HARVEST_33PCT",
                }
                executed_actions.setdefault("take_profits_tp0", []).append(tp0_record)
                logger.info(
                    f"💰 [STAGE 0 QUICK HARVEST] Sold {third_shares} shares of {ticker} @ ${spot_price:.2f} (+{ret_pct:.2f}%) | Realized PnL: ${pnl:+,.2f} | Stop locked at Breakeven (${pos['sl_target']:.2f})"
                )
                send_discord_execution_alert(
                    {
                        "action": "SELL",
                        "stage": "STAGE0_QUICK_HARVEST_1.5PCT",
                        "ticker": ticker,
                        "price": spot_price,
                        "entry_price": entry_price,
                        "shares": pos["shares"],
                        "realized_pnl": pnl,
                        "return_pct": ret_pct,
                    }
                )
                shares = pos["shares"]

            # Check Stage 1 Scale-Out (+2.5 ATR)
            if not scaled_out and spot_price >= tp1_target:

                half_shares = max(1, shares // 2)
                proceeds = float(half_shares * spot_price)
                cost_basis = float(half_shares * entry_price)
                pnl = float(proceeds - cost_basis)
                ret_pct = float((spot_price - entry_price) / denom * 100.0)

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
                ret_pct = float((spot_price - entry_price) / denom * 100.0)

                self.broker.state["cash"] += proceeds
                self.broker.state["realized_pnl"] += pnl
                self.broker.state["total_trades"] += 1
                self.broker.state["winning_trades"] += 1

                trade_record = {
                    "ticker": ticker,
                    "shares": pos["shares"],
                    "entry_price": entry_price,
                    "exit_price": spot_price,
                    "entry_date": pos.get("entry_date", date_str),
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
                ret_pct = float((spot_price - entry_price) / denom * 100.0)

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
                    "entry_date": pos.get("entry_date", date_str),
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
        from src.opening_range_engine import (
            is_opening_15min_whipsaw_period,
            find_low_of_day_pullback_entry,
        )

        curr_open_count = len(self.broker.state.get("open_positions", {}))
        available_slots = max(0, max_concurrent_positions - curr_open_count)

        # Task 7 Master Kill Switch Check
        if is_kill_switch_active():
            logger.warning(
                "🛑 [MASTER KILL SWITCH ACTIVE] New order submissions and candidate entries are strictly HALTED."
            )
            available_slots = 0

        # Task 8 Max Daily Loss Circuit Breaker
        max_daily_loss = float(os.getenv("MAX_DAILY_LOSS_PCT", "3.0"))
        if check_daily_loss_circuit_breaker(
            portfolio_summary, max_daily_loss_pct=max_daily_loss
        ):
            logger.warning(
                f"🚨 [CIRCUIT BREAKER: MAX DAILY LOSS] Daily drawdown exceeded {max_daily_loss}%. Halting new buy orders for session."
            )
            available_slots = 0

        # Opening 15-Minute Shield
        if is_opening_15min_whipsaw_period():
            logger.info(
                "🛡️ [OPENING 15-MIN SHIELD] Pausing aggressive buy orders (09:30 - 09:45 EDT) to let morning whiplash & retail gap traps settle."
            )
            available_slots = 0

        cash_available = self.broker.state.get("cash", 0.0)

        if available_slots > 0 and cash_available > 5000.0:
            unheld_tickers = [
                t
                for t in tickers_to_scan
                if t not in self.broker.state.get("open_positions", {})
            ]

            # Stage 1: High-Speed Full Universe Scanning (Scans all 500+ S&P assets)
            logger.info(
                f"🌐 [FULL UNIVERSE SCAN] Initiating parallel multi-agent evaluation across all {len(unheld_tickers)} candidate stocks..."
            )

            # Pre-warm FinBERT singleton once before spawning threads
            try:
                from src.preprocessing import _load_sentiment_analyzer

                _load_sentiment_analyzer()
            except Exception as se:
                logger.debug(f"FinBERT pre-warm notice: {se}")

            import concurrent.futures

            def _deliberate_single(ticker_sym: str):
                try:
                    q_data = quotes_map.get(ticker_sym, {})
                    cached_price = float(q_data.get("price", 0.0))
                    get_news(ticker_sym, use_cache=True)
                    delib_res = convene_trading_committee(
                        ticker_sym,
                        save_resolution=False,
                        spot_price=cached_price,
                    )
                    return ticker_sym, delib_res
                except Exception as e:
                    logger.debug(f"Committee scan notice for {ticker_sym}: {e}")
                    return ticker_sym, None

            deliberations = []
            # High-throughput thread pool (12 workers) to evaluate entire 500-stock universe rapidly
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                results = executor.map(_deliberate_single, unheld_tickers)
                for t, delib in results:
                    if delib and isinstance(delib, dict):
                        deliberations.append((t, delib))
                        executed_actions["committee_resolutions"][t] = delib.get(
                            "final_resolution", "NEUTRAL"
                        )

            # Sort candidate opportunities by CRO consensus conviction safely
            buy_candidates = [
                (t, d)
                for t, d in deliberations
                if isinstance(d, dict)
                and d.get("action_code") in ["EXECUTE_BUY", "SCALE_IN"]
            ]
            buy_candidates.sort(
                key=lambda x: (
                    float(x[1].get("consensus_conviction_pct") or 0.0)
                    if isinstance(x[1], dict)
                    else 0.0
                ),
                reverse=True,
            )

            # Task 8 Max Position Size Hard Constraint: Max 20% of total portfolio equity
            total_eq = max(
                float(portfolio_summary.get("total_equity") or 100000.0), 1.0
            )
            max_position_dollars = total_eq * 0.20

            # Execute entries into Top candidate setups up to available slots
            for t, delib in buy_candidates[:available_slots]:
                spot_price = float(delib.get("spot_price", 0))
                if spot_price <= 0:
                    continue

                # Portfolio Correlation Matrix Shield Check (Markowitz Diversification)
                from src.correlation_shield import check_correlation_shield

                corr_check = check_correlation_shield(
                    candidate_ticker=t,
                    open_positions=self.broker.state.get("open_positions", {}),
                    max_corr_threshold=0.70,
                )
                if not corr_check.get("allowed"):
                    logger.warning(
                        f"🛡️ [CORRELATION SHIELD VETO] Candidate {t} rejected: {corr_check.get('reason')}"
                    )
                    continue

                # Ensure Kelly sizing allocation does not breach max position size
                cro_info = delib.get("cro_signoff") or {}
                approved_kelly = float(cro_info.get("approved_kelly_pct", 8.0))
                max_allowed_kelly = round(
                    (max_position_dollars / total_eq) * 100.0,
                    1,
                )
                if approved_kelly > max_allowed_kelly:
                    logger.info(
                        f"🔒 [CIRCUIT BREAKER: POSITION SIZE] Sizing for {t} capped from {approved_kelly}% to {max_allowed_kelly}% max position limit."
                    )
                    if isinstance(delib.get("cro_signoff"), dict):
                        delib["cro_signoff"]["approved_kelly_pct"] = max_allowed_kelly

                order_res = execute_committee_order(
                    ticker=t, deliberation=delib, broker=self.broker
                )
                if order_res.get("success"):
                    executed_actions["buys"].append(order_res)
                    resolution_text = delib.get("final_resolution", "APPROVED")
                    logger.info(
                        f"🚀 [AUTONOMOUS BUY] Executed {order_res.get('shares')} shares of {t} @ ${spot_price:.2f} (Verdict: {resolution_text})"
                    )
                    send_discord_execution_alert(
                        {
                            "action": "BUY",
                            "stage": "ENTRY",
                            "ticker": t,
                            "price": spot_price,
                            "shares": order_res.get("shares"),
                            "kelly_pct": (delib.get("cro_signoff") or {}).get(
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

        # 5. Phase D: Self-Improving Feedback Loop & Trade Autopsy
        self_improvement_summary = self._run_self_improvement_feedback_loop(
            executed_actions
        )
        executed_actions["self_improvement"] = self_improvement_summary

        elapsed = round(time.time() - start_time, 2)
        executed_actions["elapsed_seconds"] = elapsed
        logger.info(
            f"✅ [AUTONOMOUS TRADER] Completed cycle in {elapsed}s. Buys: {len(executed_actions['buys'])}, Exits: {len(executed_actions['take_profits_tp1']) + len(executed_actions['take_profits_tp2']) + len(executed_actions['stop_losses'])}"
        )

        # Save cycle log safely with default=str serialization
        os.makedirs(os.path.dirname(AUTONOMOUS_LOG_FILE), exist_ok=True)
        try:
            with open(AUTONOMOUS_LOG_FILE, "w") as f:
                json.dump(executed_actions, f, indent=2, default=str)
        except Exception as log_err:
            logger.warning(f"Could not persist autonomous log file: {log_err}")

        return executed_actions

    def _run_self_improvement_feedback_loop(
        self, cycle_actions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes the Self-Improving Feedback Loop:
        1. Analyzes trade autopsies on closed trades (Wins vs Losses).
        2. Dynamically calibrates Committee voting weights based on trailing precision.
        3. Scales adaptive Kelly multiplier via reinforcement feedback.
        4. Triggers background continuous model retraining for decayed tickers.
        """
        memory_file = os.path.join("results", "agent_learning_memory.json")
        learning_state = {
            "total_learning_cycles": 0,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "agent_voting_weights": {
                "technicals_weight": 0.30,
                "sentiment_weight": 0.35,
                "valuation_weight": 0.15,
                "cro_weight": 0.20,
            },
            "adaptive_kelly_multiplier": 1.0,
            "recent_trade_autopsies": [],
            "retrained_models": [],
        }

        if os.path.exists(memory_file):
            try:
                with open(memory_file, "r") as f:
                    learning_state.update(json.load(f))
            except Exception as e:
                logger.warning(
                    f"Notice loading agent learning memory from {memory_file} ({e}). Initializing default learning state."
                )

        if not isinstance(learning_state.get("agent_voting_weights"), dict):
            learning_state["agent_voting_weights"] = {
                "technicals_weight": 0.30,
                "sentiment_weight": 0.35,
                "valuation_weight": 0.15,
                "cro_weight": 0.20,
            }
        for k, default_val in [
            ("technicals_weight", 0.30),
            ("sentiment_weight", 0.35),
            ("valuation_weight", 0.15),
            ("cro_weight", 0.20),
        ]:
            if k not in learning_state["agent_voting_weights"]:
                learning_state["agent_voting_weights"][k] = default_val

        learning_state["total_learning_cycles"] += 1
        learning_state["last_updated"] = datetime.now(timezone.utc).isoformat()

        # Collect closed trades in this cycle
        new_exits = (
            cycle_actions.get("take_profits_tp1", [])
            + cycle_actions.get("take_profits_tp2", [])
            + cycle_actions.get("stop_losses", [])
        )

        autopsies = []
        for trade in new_exits:
            ticker = trade.get("ticker", "UNKNOWN")
            pnl = float(trade.get("pnl", 0.0))
            ret_pct = float(trade.get("return_pct", 0.0))
            reason = trade.get("reason", "EXIT")

            is_win = pnl > 0 or "TAKE_PROFIT" in reason or "BREAK_EVEN" in reason
            if is_win:
                verdict = "🏆 POSITIVE ALPHA HARVEST"
                lesson = f"Committee conviction on {ticker} succeeded with +{ret_pct:.2f}% gain."
                # Reward sentiment & technical weights
                curr_sent = float(
                    learning_state["agent_voting_weights"].get("sentiment_weight", 0.35)
                )
                learning_state["agent_voting_weights"]["sentiment_weight"] = min(
                    0.45, round(curr_sent + 0.01, 3)
                )
                curr_kelly = float(learning_state.get("adaptive_kelly_multiplier", 1.0))
                learning_state["adaptive_kelly_multiplier"] = min(
                    1.25, round(curr_kelly + 0.02, 2)
                )
            else:
                verdict = "🛑 CONTROLLED RISK SHUTDOWN"
                lesson = f"Stop loss on {ticker} triggered at {ret_pct:.2f}%. Protecting capital."
                # Increase CRO risk weight and penalize volatile sentiment
                curr_cro = float(
                    learning_state["agent_voting_weights"].get("cro_weight", 0.20)
                )
                learning_state["agent_voting_weights"]["cro_weight"] = min(
                    0.35, round(curr_cro + 0.01, 3)
                )
                curr_kelly = float(learning_state.get("adaptive_kelly_multiplier", 1.0))
                learning_state["adaptive_kelly_multiplier"] = max(
                    0.75, round(curr_kelly - 0.03, 2)
                )

                # Queue model for continuous retraining
                try:
                    from src.continuous_learner import (
                        execute_continuous_retrain_cycle,
                    )

                    retrain_res = execute_continuous_retrain_cycle(
                        ticker, tune_hyperparameters=False
                    )
                    learning_state["retrained_models"].append(
                        {
                            "ticker": ticker,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "verdict": retrain_res.get("deployment_verdict"),
                        }
                    )
                except Exception as re:
                    logger.debug(f"Continuous retraining notice for {ticker}: {re}")

            autopsies.append(
                {
                    "ticker": ticker,
                    "pnl": pnl,
                    "return_pct": ret_pct,
                    "verdict": verdict,
                    "lesson": lesson,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        if autopsies:
            learning_state["recent_trade_autopsies"].extend(autopsies)
            learning_state["recent_trade_autopsies"] = learning_state[
                "recent_trade_autopsies"
            ][-25:]

        # Normalize weights to sum to 1.0
        w_dict = learning_state["agent_voting_weights"]
        tot_w = sum(w_dict.values())
        if tot_w > 0:
            learning_state["agent_voting_weights"] = {
                k: round(v / tot_w, 3) for k, v in w_dict.items()
            }

        # Persist learning memory
        os.makedirs(os.path.dirname(memory_file), exist_ok=True)
        with open(memory_file, "w") as f:
            json.dump(learning_state, f, indent=2)

        return {
            "total_cycles": learning_state["total_learning_cycles"],
            "agent_weights": learning_state["agent_voting_weights"],
            "kelly_multiplier": learning_state["adaptive_kelly_multiplier"],
            "new_autopsies_count": len(autopsies),
        }

    def run_premarket_briefing(self) -> Dict[str, Any]:
        """
        Gathers overnight macro VIX volatility regime, paper portfolio balance,
        and top watchlist committee resolutions, then dispatches a rich morning briefing to Discord.
        """
        logger.info("🌅 [PRE-MARKET BRIEFING] Convening morning intelligence desk...")
        summary = self.broker.get_portfolio_summary()

        # Fetch Macro VIX
        try:
            vix_quote = fetch_live_quote("^VIX")
            macro_vix = float(vix_quote.get("price", 16.5))
        except Exception:
            macro_vix = 16.5

        # Scan top 5 benchmark tickers for morning setup
        scan_tickers = ["NVDA", "AAPL", "MSFT", "TSLA", "AMZN"]
        top_watchlist = []

        for tk in scan_tickers:
            try:
                delib = convene_trading_committee(tk, save_resolution=False)
                res = delib.get("final_resolution", "HOLD")
                conv = delib.get("consensus_conviction_pct", 50.0)
                # Find FinBERT sentiment score from testimonies
                sent_score = 0.0
                for t in delib.get("agent_testimonies", []):
                    if "Sentiment" in t.get("agent_name", ""):
                        sent_score = t.get("conviction_score", 50.0) / 100.0
                top_watchlist.append(
                    {
                        "ticker": tk,
                        "resolution": res,
                        "conviction": conv,
                        "sentiment_score": sent_score,
                    }
                )
            except Exception as e:
                logger.warning(f"Error evaluating {tk} for morning briefing: {e}")

        # Send Discord Briefing
        sent = send_discord_premarket_briefing(
            portfolio_summary=summary,
            macro_vix=macro_vix,
            top_watchlist=top_watchlist,
        )

        return {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "discord_dispatched": sent,
            "macro_vix": macro_vix,
            "total_equity": summary.get("total_equity", 100000.0),
            "watchlist_evaluated": len(top_watchlist),
        }


import threading

_DAEMON_THREAD = None
_DAEMON_LOCK = threading.Lock()
_LAST_DAEMON_PULSE = {"timestamp": None, "status": "IDLE", "actions": None}


def get_daemon_status() -> Dict[str, Any]:
    """Returns current live daemon running state and last pulse timestamp."""
    global _DAEMON_THREAD, _LAST_DAEMON_PULSE
    is_alive = _DAEMON_THREAD is not None and _DAEMON_THREAD.is_alive()
    return {
        "is_active": is_alive,
        "last_pulse": _LAST_DAEMON_PULSE.get("timestamp"),
        "last_status": _LAST_DAEMON_PULSE.get("status", "IDLE"),
        "last_actions": _LAST_DAEMON_PULSE.get("actions"),
    }


def ensure_background_daemon_thread_running(interval_seconds: int = 60):
    """
    Ensures a single background autonomous trading daemon thread is permanently active.
    Polls the market continuously and executes cycles during open market hours.
    """
    global _DAEMON_THREAD, _DAEMON_LOCK
    with _DAEMON_LOCK:
        if _DAEMON_THREAD is not None and _DAEMON_THREAD.is_alive():
            return _DAEMON_THREAD

        def _daemon_loop():
            engine = AutonomousTradingEngine()
            logger.info(
                f"🤖 [PERMANENT 24/7 DAEMON THREAD] Initialized. Interval: {interval_seconds}s"
            )
            while threading.main_thread().is_alive():
                try:
                    from src.market_session import get_us_market_session

                    sess = get_us_market_session()
                    if sess.get("is_open", False):
                        logger.info(
                            f"🟢 [DAEMON CYCLE] Market is LIVE ({sess.get('time_edt')}). Running autonomous scan..."
                        )
                        cycle_res = engine.run_autonomous_cycle()
                        _LAST_DAEMON_PULSE["timestamp"] = datetime.now(
                            timezone.utc
                        ).isoformat()
                        _LAST_DAEMON_PULSE["status"] = "EXECUTED_CYCLE"
                        _LAST_DAEMON_PULSE["actions"] = {
                            "buys": len(cycle_res.get("buys", [])),
                            "tp1": len(cycle_res.get("take_profits_tp1", [])),
                            "tp2": len(cycle_res.get("take_profits_tp2", [])),
                            "stops": len(cycle_res.get("stop_losses", [])),
                        }
                    else:
                        _LAST_DAEMON_PULSE["timestamp"] = datetime.now(
                            timezone.utc
                        ).isoformat()
                        _LAST_DAEMON_PULSE["status"] = (
                            f"STANDBY_{sess.get('status', 'MARKET_CLOSED')}"
                        )
                except (RuntimeError, ValueError):
                    break
                except Exception as e:
                    logger.error(f"Daemon background loop exception: {e}")
                    _LAST_DAEMON_PULSE["status"] = f"ERROR: {str(e)[:100]}"

                # Sleep in short increments to allow rapid shutdown
                for _ in range(max(1, interval_seconds // 2)):
                    if not threading.main_thread().is_alive():
                        break
                    time.sleep(2)

        _DAEMON_THREAD = threading.Thread(
            target=_daemon_loop, daemon=True, name="SentilyzeAutonomousDaemon"
        )
        _DAEMON_THREAD.start()
        logger.info(
            "🚀 [PERMANENT 24/7 DAEMON THREAD] Background trading worker started successfully."
        )
        return _DAEMON_THREAD


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
        "--premarket",
        action="store_true",
        help="Run morning pre-market briefing and dispatch Discord report",
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

    if args.premarket:
        res = engine.run_premarket_briefing()
        print(json.dumps(res, indent=2))
    elif args.daemon:
        run_autonomous_daemon(interval_seconds=args.interval)
    else:
        summary = engine.run_autonomous_cycle()
        print(json.dumps(summary, indent=2))
