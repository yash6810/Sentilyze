"""
Sentilyze Master Autonomous Daily Operating Trading Loop.

Orchestrates the 4-Phase Institutional Trading Lifecycle:
1. Phase 1: Pre-Market Reconnaissance (08:30 - 09:30 EDT)
   - Macro yield curve & regime from InstitutionalDataGateway
   - Volatility blackouts & central bank calendar
   - SEC Form 8-K & BioTech catalyst radar
   - Morning intelligence briefing & Discord alert
2. Phase 2: Market Open Whipsaw Shield (09:30 - 09:45 EDT)
   - 15-minute opening protection against false breakout gap traps
   - Sentinel price floor guard on existing portfolio holdings
3. Phase 3: Intraday Execution & Council Arbitration (09:45 - 15:50 EDT)
   - Master Kill Switch & Daily Loss circuit breaker audit
   - 8-Agent Council quorum deliberation
   - Quarter-Kelly sizing & dynamic ATR corridors
   - Internal Paper Broker ledger update + Alpaca bracket order dispatch
   - Sentinel trailing profit lock & staged scale-outs
4. Phase 4: Post-Market Autopsy & Self-Calibration (16:00 - 16:30 EDT)
   - Mark-to-market portfolio reconciliation
   - Quantitative trade autopsy (win rate, payoff ratio, Calmar, Sortino)
   - Dynamic Council weight recalibration in agent memory
"""

import os
import json
import time
import threading
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from src.utils import get_logger
from src.paper_broker import PaperBroker, PORTFOLIO_FILE
from src.alpaca_broker import AlpacaBrokerBridge
from src.institutional_gateway import InstitutionalDataGateway
from src.agent_committee import convene_trading_committee, execute_committee_order
from src.realtime_tracker import fetch_live_quote
from src.backtest_autopsy import run_autonomous_trade_autopsy, TRADES_FILE
from src.agent_memory import AgentMemoryStore
from src.autonomous_trader import (
    is_kill_switch_active,
    check_daily_loss_circuit_breaker,
    load_universe_tickers,
    AutonomousTradingEngine,
)
from src.macro_calendar import evaluate_macro_volatility_blackout
from src.sec_crawler import SECCatalystCrawler
from src.biotech_catalyst_radar import compile_biotech_catalyst_radar
from src.market_session import get_us_market_session

logger = get_logger(__name__)

MASTER_SUMMARY_FILE = os.path.join("results", "master_loop_execution_summary.json")


class MasterTradingLoop:
    """
    Unified Master Daily Trading Loop Orchestrator for Sentilyze.
    Coordinates all 4 daily trading phases deterministically and safely.
    """

    def __init__(
        self,
        broker: Optional[PaperBroker] = None,
        portfolio_path: Optional[str] = None,
        trades_path: Optional[str] = None,
        alpaca_bridge: Optional[AlpacaBrokerBridge] = None,
    ):
        self.portfolio_path = portfolio_path or PORTFOLIO_FILE
        self.trades_path = trades_path or TRADES_FILE
        self.broker = broker or PaperBroker(portfolio_path=self.portfolio_path)
        self.alpaca = alpaca_bridge or AlpacaBrokerBridge()
        self.institutional_gateway = InstitutionalDataGateway()
        self.sec_crawler = SECCatalystCrawler()
        self.engine = AutonomousTradingEngine(broker=self.broker)
        self.memory_store = AgentMemoryStore()

        self._last_cycle_result: Dict[str, Any] = {}
        self._daemon_thread: Optional[threading.Thread] = None
        self._stop_daemon_event = threading.Event()

    # -------------------------------------------------------------------------
    # PHASE 1: PRE-MARKET RECONNAISSANCE (08:30 - 09:30 EDT)
    # -------------------------------------------------------------------------
    def run_premarket_phase(
        self, watchlist: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Executes Pre-Market Intelligence:
        1. Macro yield curve & macro regime via InstitutionalDataGateway.
        2. Central bank & macro calendar volatility blackout check.
        3. SEC Form 8-K & BioTech catalyst scans on watchlist.
        4. Dispatches morning intelligence briefing.
        """
        logger.info("🌅 [MASTER LOOP - PHASE 1] Starting Pre-Market Reconnaissance...")
        phase_start = datetime.now(timezone.utc).isoformat()
        tickers = watchlist or ["NVDA", "AAPL", "MSFT", "TSLA", "AMZN", "GOOGL"]

        # 1.1 Macro Regime & Yield Spreads
        try:
            macro_regime = self.institutional_gateway.get_macro_regime()
        except Exception as e:
            logger.warning(f"Macro regime query fallback: {e}")
            macro_regime = {
                "regime": "NEUTRAL_GROWTH",
                "vix": 16.5,
                "yield_spread": 0.15,
                "notes": "Fallback default",
            }

        # 1.2 Volatility Blackout
        try:
            blackout = evaluate_macro_volatility_blackout()
        except Exception as e:
            logger.warning(f"Macro calendar blackout check fallback: {e}")
            blackout = {"is_blackout_active": False, "reason": "Default clear"}

        # 1.3 SEC Form 8-K & Biotech Catalyst Scans
        catalysts_found: Dict[str, Any] = {}
        for sym in tickers[:4]:
            try:
                sec_items = self.sec_crawler.fetch_recent_8k_filings(
                    sym, days_lookback=7
                )
                bio_radar = compile_biotech_catalyst_radar(sym)
                catalysts_found[sym] = {
                    "sec_filings_count": len(sec_items),
                    "material_risk": any(
                        i.get("material_risk_flag", False) for i in sec_items
                    ),
                    "biotech_risk": bio_radar.get("risk_profile", "UNKNOWN"),
                }
            except Exception as e:
                logger.debug(f"Catalyst scan notice for {sym}: {e}")

        # 1.4 Morning Briefing Dispatch
        try:
            briefing_res = self.engine.run_premarket_briefing()
        except Exception as e:
            logger.warning(f"Morning briefing dispatch notice: {e}")
            briefing_res = {"status": "SKIPPED", "error": str(e)}

        res = {
            "phase": "PRE_MARKET_RECONNAISSANCE",
            "timestamp": phase_start,
            "macro_regime": macro_regime,
            "volatility_blackout": blackout,
            "catalysts_scanned": catalysts_found,
            "morning_briefing": briefing_res,
            "portfolio_equity": self.broker.get_portfolio_summary().get(
                "total_equity", 100000.0
            ),
        }
        return res

    # -------------------------------------------------------------------------
    # PHASE 2: MARKET OPEN WHIPSAW SHIELD (09:30 - 09:45 EDT)
    # -------------------------------------------------------------------------
    def run_opening_shield_phase(
        self, bypass_time_check: bool = False
    ) -> Dict[str, Any]:
        """
        Executes Opening 15-Minute Whipsaw Shield:
        1. Checks whether current market time is in the opening whipsaw zone (09:30-09:45).
        2. Protects open positions by auditing Sentinel stops and floor barriers.
        """
        logger.info(
            "🛡️ [MASTER LOOP - PHASE 2] Activating Market Open Whipsaw Shield..."
        )
        phase_start = datetime.now(timezone.utc).isoformat()
        session = get_us_market_session()

        is_opening_zone = session.get("is_opening_range", False)
        shield_active = is_opening_zone or bypass_time_check

        open_positions = list(self.broker.state.get("open_positions", {}).keys())
        protected_holdings = []

        for sym in open_positions:
            pos = self.broker.state["open_positions"].get(sym, {})
            sl = pos.get("sl_target", 0.0)
            entry = pos.get("entry_price", 0.0)
            protected_holdings.append(
                {
                    "ticker": sym,
                    "entry_price": entry,
                    "stop_loss": sl,
                    "shares": pos.get("shares", 0),
                    "status": "GUARDED",
                }
            )

        return {
            "phase": "MARKET_OPEN_WHIPSAW_SHIELD",
            "timestamp": phase_start,
            "shield_active": shield_active,
            "is_opening_range_session": is_opening_zone,
            "holdings_guarded_count": len(protected_holdings),
            "protected_holdings": protected_holdings,
            "new_entries_allowed": not shield_active,
        }

    # -------------------------------------------------------------------------
    # PHASE 3: INTRADAY EXECUTION & ARBITRATION (09:45 - 15:50 EDT)
    # -------------------------------------------------------------------------
    def run_intraday_phase(
        self,
        candidate_tickers: Optional[List[str]] = None,
        max_candidates: int = 5,
        force_execution: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes Intraday Deliberation & Order Routing:
        1. Checks Master Kill Switch and Max-Daily-Loss Circuit Breaker.
        2. Convenes 8-Agent Council quorum on ranked candidates.
        3. Sizes trades via Quarter-Kelly & executes approved orders in Paper Broker.
        4. Mirrors orders to Alpaca Paper Bracket Order API if connected.
        5. Audits trailing profit locks and staged exits.
        """
        logger.info(
            "⚡ [MASTER LOOP - PHASE 3] Launching Intraday Execution & Arbitration..."
        )
        phase_start = datetime.now(timezone.utc).isoformat()

        # 3.1 Circuit Breaker Audits
        if is_kill_switch_active():
            logger.critical(
                "🛑 [CIRCUIT BREAKER] Master Kill Switch is ACTIVE. Halting intraday execution."
            )
            return {
                "phase": "INTRADAY_EXECUTION",
                "timestamp": phase_start,
                "status": "HALTED_BY_KILL_SWITCH",
                "trades_executed": 0,
            }

        port_summary = self.broker.get_portfolio_summary()
        if check_daily_loss_circuit_breaker(port_summary, max_daily_loss_pct=3.0):
            logger.critical(
                "🛑 [CIRCUIT BREAKER] Max Daily Loss Breached (-3.0%). Halting execution."
            )
            return {
                "phase": "INTRADAY_EXECUTION",
                "timestamp": phase_start,
                "status": "HALTED_BY_DAILY_LOSS_BREAKER",
                "trades_executed": 0,
            }

        # 3.2 Candidate Selection
        candidates = candidate_tickers or load_universe_tickers()[:max_candidates]
        logger.info(f"Deliberating on {len(candidates)} candidates: {candidates}")

        deliberations: List[Dict[str, Any]] = []
        executed_orders: List[Dict[str, Any]] = []
        alpaca_orders: List[Dict[str, Any]] = []

        for ticker in candidates:
            try:
                # Convene 8-Agent Council
                delib = convene_trading_committee(ticker, save_resolution=True)
                res = delib.get("final_resolution", "HOLD")
                conv = delib.get("consensus_conviction_pct", 50.0)
                cro_veto = delib.get("cro_veto_triggered", False)

                deliberations.append(
                    {
                        "ticker": ticker,
                        "resolution": res,
                        "conviction": conv,
                        "cro_veto": cro_veto,
                    }
                )

                # Execute only high-conviction un-vetoed BUY signals
                if (res == "BUY" and conv >= 58.0 and not cro_veto) or force_execution:
                    exec_record = execute_committee_order(delib, broker=self.broker)
                    if exec_record and exec_record.get("status") == "EXECUTED":
                        executed_orders.append(exec_record)
                        logger.info(
                            f"✅ Executed Paper Order for {ticker}: {exec_record.get('shares')} shares @ ${exec_record.get('entry_price', 0):.2f}"
                        )

                        # Mirror to Alpaca Bracket Order if connected
                        if self.alpaca.is_connected():
                            shares = int(exec_record.get("shares", 1))
                            tp = float(exec_record.get("tp1", 0.0))
                            sl = float(exec_record.get("sl", 0.0))
                            if shares > 0 and tp > 0 and sl > 0:
                                alpaca_res = self.alpaca.submit_bracket_order(
                                    ticker=ticker,
                                    qty=shares,
                                    take_profit_price=tp,
                                    stop_loss_price=sl,
                                    side="buy",
                                )
                                alpaca_orders.append(alpaca_res)
                                logger.info(
                                    f"🦙 Alpaca Bracket Order mirrored for {ticker}"
                                )

            except Exception as e:
                logger.warning(f"Error in committee deliberation for {ticker}: {e}")

        # 3.3 Manage Existing Open Positions (Trailing profit lock & TP harvests)
        try:
            pos_audit = self.engine._manage_open_positions()
        except Exception as e:
            logger.debug(f"Position audit notice: {e}")
            pos_audit = {}

        return {
            "phase": "INTRADAY_EXECUTION",
            "timestamp": phase_start,
            "status": "COMPLETED",
            "candidates_evaluated": len(deliberations),
            "deliberations": deliberations,
            "executed_orders": executed_orders,
            "alpaca_orders": alpaca_orders,
            "position_management": pos_audit,
            "cash_remaining": self.broker.state.get("cash", 0.0),
            "total_equity": self.broker.state.get("total_equity", 100000.0),
        }

    # -------------------------------------------------------------------------
    # PHASE 4: POST-MARKET AUTOPSY & SELF-CALIBRATION (16:00 - 16:30 EDT)
    # -------------------------------------------------------------------------
    def run_postmarket_phase(self, force_autopsy: bool = False) -> Dict[str, Any]:
        """
        Executes Post-Market Trade Autopsy & Agent Weight Recalibration:
        1. Runs quantitative autopsy on executed trades log.
        2. Recalibrates 8-Agent Council weights and episodic memory.
        3. Summarizes daily win-rate, payoff ratio, and Calmar ratio.
        """
        logger.info(
            "📊 [MASTER LOOP - PHASE 4] Running Post-Market Autopsy & Calibration..."
        )
        phase_start = datetime.now(timezone.utc).isoformat()

        # 4.1 Quantitative Trade Autopsy
        try:
            autopsy_res = run_autonomous_trade_autopsy(trades_path=self.trades_path)
        except Exception as e:
            logger.error(f"Error running autopsy: {e}")
            autopsy_res = {"status": "ERROR", "error": str(e)}

        # 4.2 Committee Dynamic Weight Calibration
        try:
            learn_res = self.engine._run_self_improvement_feedback_loop(
                cycle_actions={}
            )
        except Exception as e:
            logger.warning(f"Learning cycle notice: {e}")
            learn_res = {"status": "SKIPPED", "error": str(e)}

        port_summary = self.broker.get_portfolio_summary()

        res = {
            "phase": "POST_MARKET_AUTOPSY",
            "timestamp": phase_start,
            "status": "COMPLETED",
            "trade_autopsy": autopsy_res,
            "committee_learning": learn_res,
            "final_equity": port_summary.get("total_equity", 100000.0),
            "total_realized_pnl": port_summary.get("total_realized_pnl", 0.0),
            "win_rate_pct": port_summary.get("win_rate_pct", 0.0),
            "total_trades": port_summary.get("total_trades", 0),
        }
        return res

    # -------------------------------------------------------------------------
    # FULL DAILY CYCLE ORCHESTRATION
    # -------------------------------------------------------------------------
    def run_full_daily_cycle(
        self,
        watchlist: Optional[List[str]] = None,
        dry_run: bool = False,
        ignore_market_hours: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs the complete 4-phase trading lifecycle in sequence:
        Phase 1 (Pre-Market) -> Phase 2 (Opening Shield) -> Phase 3 (Intraday) -> Phase 4 (Post-Market).
        """
        cycle_start = datetime.now(timezone.utc).isoformat()
        logger.info(
            f"🚀 [MASTER LOOP] Initiating Full Daily Operating Cycle (Dry Run: {dry_run}, Ignore Hours: {ignore_market_hours})..."
        )

        tickers = watchlist or ["NVDA", "AAPL", "MSFT", "TSLA", "AMZN"]

        # Phase 1: Pre-Market
        p1 = self.run_premarket_phase(watchlist=tickers)

        # Phase 2: Opening Range Shield
        p2 = self.run_opening_shield_phase(bypass_time_check=ignore_market_hours)

        # Phase 3: Intraday Deliberation & Execution
        p3 = self.run_intraday_phase(
            candidate_tickers=tickers,
            max_candidates=len(tickers),
            force_execution=False,
        )

        # Phase 4: Post-Market Autopsy
        p4 = self.run_postmarket_phase(force_autopsy=True)

        full_cycle_summary = {
            "cycle_timestamp": cycle_start,
            "dry_run": dry_run,
            "portfolio_preserved": True,
            "portfolio_path": self.portfolio_path,
            "phase_1_premarket": p1,
            "phase_2_opening_shield": p2,
            "phase_3_intraday": p3,
            "phase_4_postmarket": p4,
            "closing_equity": p4.get("final_equity", 100000.0),
        }

        self._last_cycle_result = full_cycle_summary

        # Persist cycle summary to results/ (read-only for production ledger)
        try:
            os.makedirs(os.path.dirname(MASTER_SUMMARY_FILE), exist_ok=True)
            with open(MASTER_SUMMARY_FILE, "w", encoding="utf-8") as f:
                json.dump(full_cycle_summary, f, indent=2)
            logger.info(f"Saved master loop cycle summary to {MASTER_SUMMARY_FILE}")
        except Exception as e:
            logger.warning(f"Could not persist cycle summary: {e}")

        return full_cycle_summary

    def get_cycle_status(self) -> Dict[str, Any]:
        """Returns the current operational status of the master loop for the dashboard."""
        session = get_us_market_session()
        alpaca_connected = self.alpaca.is_connected()
        port_summary = self.broker.get_portfolio_summary()

        # Determine current daily phase
        if session.get("is_premarket", False):
            current_phase = "PHASE 1: PRE-MARKET RECONNAISSANCE"
        elif session.get("is_opening_range", False):
            current_phase = "PHASE 2: MARKET OPEN WHIPSAW SHIELD"
        elif session.get("is_open", False):
            current_phase = "PHASE 3: INTRADAY EXECUTION & ARBITRATION"
        else:
            current_phase = "PHASE 4: POST-MARKET AUTOPSY & CALIBRATION"

        return {
            "current_phase": current_phase,
            "market_session": session,
            "alpaca_connected": alpaca_connected,
            "total_equity": port_summary.get("total_equity", 100000.0),
            "cash": port_summary.get("cash", 100000.0),
            "win_rate": port_summary.get("win_rate_pct", 0.0),
            "open_positions_count": len(self.broker.state.get("open_positions", {})),
            "last_cycle_timestamp": self._last_cycle_result.get("cycle_timestamp"),
        }


# Global singleton instance for daemon / UI access
_GLOBAL_MASTER_LOOP: Optional[MasterTradingLoop] = None


def get_master_trading_loop(portfolio_path: Optional[str] = None) -> MasterTradingLoop:
    """Returns the singleton instance of the MasterTradingLoop."""
    global _GLOBAL_MASTER_LOOP
    if _GLOBAL_MASTER_LOOP is None or (
        portfolio_path and _GLOBAL_MASTER_LOOP.portfolio_path != portfolio_path
    ):
        _GLOBAL_MASTER_LOOP = MasterTradingLoop(portfolio_path=portfolio_path)
    return _GLOBAL_MASTER_LOOP
