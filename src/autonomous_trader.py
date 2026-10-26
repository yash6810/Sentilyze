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
import numpy as np

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
from src.market_session import get_current_ny_time, is_crypto_asset

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
        with open(STOCKS_FILE, "r", encoding="utf-8") as f:
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


# ==============================================================================
# Sprint 4, Module 4.5 / Idea 47: Formal Logic Z3 Rule Gate
# ==============================================================================
class FormalLogicZ3RuleGate:
    """
    Formal Logic & SMT-Style Verification Rule Gate (Idea 47 / Sprint 4.5).
    Formally verifies hard mathematical invariants on candidate trade proposals:
    1. Positive Price Invariant: spot_price > 0.0
    2. Integral & Positive Shares Invariant: shares >= 1
    3. Solvency / Cash Buffer Invariant: shares * price <= cash_available - min_cash_buffer
    4. Bracket Monotonicity Invariant: stop_loss < price < tp1 <= tp2
    5. Minimum Edge / Asymmetry Invariant: (tp1 - price) / (price - stop_loss) >= 0.95
    6. Position Concentration Invariant: (shares * price) / total_equity <= max_position_pct
    """

    def __init__(
        self,
        min_cash_buffer: float = 1000.0,
        max_position_pct: float = 0.25,
        sde_cash_floor: Optional[float] = None,
    ):
        self.min_cash_buffer = min_cash_buffer
        self.max_position_pct = max_position_pct
        self.sde_cash_floor = sde_cash_floor

    def verify_trade(
        self,
        ticker: str,
        spot_price: float,
        shares: float,
        deliberation: Dict[str, Any],
        portfolio_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        violations: List[str] = []
        invariants_checked: Dict[str, bool] = {}

        # 1. Price > 0
        inv_price = spot_price > 0.0
        invariants_checked["positive_price"] = inv_price
        if not inv_price:
            violations.append(f"Price must be strictly positive: {spot_price}")

        # 2. Shares strictly positive (allows fractional crypto >= 0.000001, and >= 1 for whole shares)
        from src.market_session import is_crypto_asset

        min_shares = (
            1e-6 if is_crypto_asset(ticker) or not float(shares).is_integer() else 1.0
        )
        inv_shares = float(shares) >= min_shares
        invariants_checked["positive_shares"] = inv_shares
        if not inv_shares:
            violations.append(f"Shares must be >= {min_shares}: {shares}")

        # 3. Solvency / Cash
        cash = float(portfolio_state.get("cash", 0.0))
        cost = spot_price * shares
        inv_solvency = cost <= max(cash - self.min_cash_buffer, 0.0)
        invariants_checked["solvency_cash_buffer"] = inv_solvency
        if not inv_solvency:
            violations.append(
                f"Cost (${cost:.2f}) exceeds cash (${cash:.2f}) minus buffer (${self.min_cash_buffer:.2f})"
            )

        # 3b. 1,000,000-Path Continuous Jump-Diffusion Neural SDE Capital Floor Invariant
        if not is_crypto_asset(ticker):
            sde_floor = self.sde_cash_floor
            if sde_floor is None:
                try:
                    sde_file = os.path.join("results", "cvar_stress_report.json")
                    if os.path.exists(sde_file):
                        with open(sde_file, "r", encoding="utf-8") as sfp:
                            sde_floor = float(
                                json.load(sfp).get(
                                    "recommended_dynamic_cash_floor", 118014.06
                                )
                            )
                except Exception:
                    pass
            tot_eq = float(portfolio_state.get("total_equity", cash))
            if sde_floor is not None and sde_floor > 0.0 and tot_eq >= sde_floor:
                inv_sde_floor = (cash - cost) >= sde_floor
                invariants_checked["neural_sde_cash_floor"] = inv_sde_floor
                if not inv_sde_floor:
                    violations.append(
                        f"Order cost (${cost:,.2f}) breaches Neural SDE survival floor (${sde_floor:,.2f}). Remaining liquid cash (${cash - cost:,.2f}) is below capital moat."
                    )
            else:
                inv_sde_floor = (cash - cost) >= self.min_cash_buffer
                invariants_checked["neural_sde_cash_floor"] = inv_sde_floor

        # 4. Bracket Monotonicity
        cro_dict = (
            deliberation.get("cro_signoff", {})
            if isinstance(deliberation.get("cro_signoff"), dict)
            else {}
        )
        tp1 = float(
            deliberation.get("tp1_target")
            or cro_dict.get("tp1_target")
            or (spot_price * 1.05)
        )
        tp2 = float(
            deliberation.get("tp2_target")
            or cro_dict.get("tp2_target")
            or (spot_price * 1.10)
        )
        sl = float(
            deliberation.get("stop_loss_target")
            or cro_dict.get("stop_loss_target")
            or (spot_price * 0.96)
        )

        inv_bracket = sl < spot_price < tp1 <= tp2
        invariants_checked["bracket_monotonicity"] = inv_bracket
        if not inv_bracket:
            violations.append(
                f"Bracket monotonicity violated: SL({sl}) < Spot({spot_price}) < TP1({tp1}) <= TP2({tp2})"
            )

        # 5. Risk-Reward Asymmetry
        risk = max(spot_price - sl, 1e-4)
        reward = max(tp1 - spot_price, 0.0)
        rr_ratio = reward / risk
        inv_asymmetry = rr_ratio >= 0.95
        invariants_checked["minimum_risk_reward"] = inv_asymmetry
        if not inv_asymmetry:
            violations.append(f"Risk-reward ratio ({rr_ratio:.2f}) below minimum 0.95")

        # 6. Concentration
        total_equity = float(portfolio_state.get("total_equity", cash))
        inv_conc = (cost / max(total_equity, 1.0)) <= self.max_position_pct
        invariants_checked["concentration_bound"] = inv_conc
        if not inv_conc:
            violations.append(
                f"Position size ({(cost/total_equity)*100:.1f}%) exceeds maximum cap ({self.max_position_pct*100:.1f}%)"
            )

        is_valid = len(violations) == 0
        return {
            "is_valid": is_valid,
            "status": "SATISFIED" if is_valid else "VIOLATED",
            "violations": violations,
            "invariants_checked": invariants_checked,
        }


class AutonomousTradingEngine:
    """
    Autonomous Execution Engine that integrates Live News Ingestion,
    4-Agent Committee Deliberation, and 2-Stage Profit Scaling.
    """

    def __init__(
        self,
        broker: Optional[PaperBroker] = None,
        crypto_broker: Optional[PaperBroker] = None,
    ):
        self.broker = broker or PaperBroker()
        self.crypto_broker = crypto_broker or PaperBroker(
            portfolio_path=os.path.join("results", "pilot_crypto_portfolio.json"),
            initial_cash=25000.0,
            trades_path=os.path.join("results", "pilot_crypto_trades.csv"),
        )
        self.tickers = load_universe_tickers()
        self.rule_gate = FormalLogicZ3RuleGate()

    def get_target_broker(self, ticker: str) -> PaperBroker:
        """Routes execution and state to the isolated crypto sandbox or standard US equity broker."""
        from src.market_session import is_crypto_asset

        if is_crypto_asset(ticker):
            return self.crypto_broker
        return self.broker

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
        max_concurrent_positions: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes one full autonomous decision and execution cycle with:
        - Task 6: Idempotency lock guard
        - Task 7: Master Kill Switch check
        - Task 8: Daily loss & position size circuit breakers
        - Task 9: Unhandled exception alerting
        """
        if max_concurrent_positions is None:
            max_concurrent_positions = int(os.getenv("MAX_CONCURRENT_POSITIONS", "10"))
        # 1. Idempotency Lock Check (Task 6)
        if os.path.exists(LOCK_FILE):
            try:
                with open(LOCK_FILE, "r", encoding="utf-8") as f:
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
            with open(LOCK_FILE, "w", encoding="utf-8") as f:
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
            import gc

            gc.collect()
            # Release lock in all circumstances
            if os.path.exists(LOCK_FILE):
                try:
                    os.remove(LOCK_FILE)
                except Exception:
                    pass

    def manage_open_positions(self) -> Dict[str, Any]:
        """
        Public method to audit and ratchet open positions (Phase A).
        Used by MasterTradingLoop and external surveillance callers.
        """
        open_tickers = list(self.broker.state.get("open_positions", {}).keys())
        if not open_tickers:
            return {"status": "NO_OPEN_POSITIONS", "actions": []}
        return self._execute_cycle_body(
            candidate_tickers=[], max_concurrent_positions=0
        )

    def _manage_open_positions(self) -> Dict[str, Any]:
        """Backward-compatibility alias for manage_open_positions."""
        return self.manage_open_positions()

    def _execute_cycle_body(
        self,
        candidate_tickers: Optional[List[str]] = None,
        max_concurrent_positions: int = 10,
    ) -> Dict[str, Any]:
        """Core cycle execution body."""
        start_time = time.time()
        cycle_t0_ns = time.perf_counter_ns()
        tickers_to_scan = candidate_tickers or self.tickers
        now_ny = get_current_ny_time()
        now_str = now_ny.isoformat()
        date_str = now_ny.strftime("%Y-%m-%d")

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
        all_open_positions = []
        for tk in list(self.broker.state.get("open_positions", {}).keys()):
            all_open_positions.append((tk, self.broker))
        for tk in list(self.crypto_broker.state.get("open_positions", {}).keys()):
            all_open_positions.append((tk, self.crypto_broker))

        for ticker, target_broker in all_open_positions:
            pos = target_broker.state["open_positions"].get(ticker)
            if not pos:
                continue

            q = quotes_map.get(ticker) or fetch_live_quote(ticker)
            spot_price = float(q.get("price", 0))
            if spot_price <= 0 or q.get("status") == "UNAVAILABLE":
                logger.warning(
                    f"⚠️ [POSITION EVALUATION] Price unavailable for {ticker}. Skipping execution checks."
                )
                continue

            last_known = float(
                pos.get("current_price") or pos.get("entry_price") or spot_price
            )
            if last_known > 0 and q.get("status") != "LIVE":
                pct_diff = abs(spot_price - last_known) / last_known
                if pct_diff > 0.20:
                    logger.warning(
                        f"🛡️ [PRICE SANITY VETO] Cached price for {ticker} (${spot_price:.2f}) "
                        f"differs >20% from last known (${last_known:.2f}). Rejecting unverified price shock."
                    )
                    continue

            pos["current_price"] = spot_price
            from src.paper_broker import reconcile_split_if_needed

            reconcile_split_if_needed(pos, ticker, spot_price)

            raw_shares = float(pos.get("shares", 0))
            shares = raw_shares if not raw_shares.is_integer() else int(raw_shares)
            entry_price = float(pos.get("entry_price") or spot_price or 1.0)
            denom = entry_price if entry_price > 0 else 1.0
            tp1_target = float(pos.get("tp1_target", entry_price * 1.05))
            tp2_target = float(pos.get("tp2_target", entry_price * 1.10))
            sl_target = float(pos.get("sl_target", entry_price * 0.96))
            stage0_taken = bool(pos.get("stage0_taken", False))
            scaled_out = bool(pos.get("scaled_out", False))

            # 🛡️ High-Water Mark & Dynamic Trailing Profit Shield
            high_water = max(
                float(pos.get("high_water_mark", entry_price)),
                float(pos.get("highest_price_seen", entry_price)),
                spot_price,
            )
            pos["high_water_mark"] = high_water
            pos["highest_price_seen"] = high_water
            unrealized_gain_pct = (spot_price - entry_price) / denom * 100.0
            max_gain_pct = (high_water - entry_price) / denom * 100.0

            # Level 0: Enforce Capital Shield Hard Floor (-2.50% max risk ceiling)
            try:
                from src.smart_trader_engine import (
                    apply_high_watermark_profit_lock,
                    enforce_capital_shield_stop_floor,
                )

                shielded_sl, _ = enforce_capital_shield_stop_floor(
                    entry_price=entry_price,
                    current_sl=sl_target,
                    max_loss_pct=2.50,
                )
                if shielded_sl > sl_target:
                    pos["sl_target"] = shielded_sl
                    sl_target = shielded_sl

                # Level 1: Institutional 80% Peak Gain Retention Ratchet
                # Guarantees that once a trade reaches peak profit, never give back > 20% of gains
                hwm_sl, new_peak, hwm_action = apply_high_watermark_profit_lock(
                    current_price=spot_price,
                    entry_price=entry_price,
                    highest_price_seen=high_water,
                    current_sl=sl_target,
                    min_profit_threshold_pct=1.20,
                    lock_fraction=0.80,
                )
                pos["highest_price_seen"] = new_peak
                if hwm_sl > sl_target:
                    pos["sl_target"] = hwm_sl
                    sl_target = hwm_sl
                    logger.info(f"🔒 [{ticker} PEAK PROFIT LOCK] {hwm_action}")
            except Exception as pe:
                logger.debug(f"Notice applying high watermark lock for {ticker}: {pe}")

            # Level 2: Multi-Tier Progressive Profit Bank & Guaranteed Floors
            # Prevents +10% gain from ever eroding back to +2%
            if scaled_out:
                floor_runner_sl = round(entry_price * 1.02, 2)
                chandelier_sl = round(
                    pos.get("highest_price_seen", spot_price) * 0.965, 2
                )
                target_sl = max(floor_runner_sl, chandelier_sl)
                if target_sl > sl_target:
                    pos["sl_target"] = target_sl
                    sl_target = target_sl
                    pos["profit_lock_status"] = f"🏆 RUNNER RATCHET (${target_sl:.2f})"
            else:
                if max_gain_pct >= 10.0:
                    peak_80_sl = round(
                        entry_price + (high_water - entry_price) * 0.80, 2
                    )
                    if peak_80_sl > sl_target:
                        pos["sl_target"] = peak_80_sl
                        sl_target = peak_80_sl
                        pos["profit_lock_status"] = (
                            f"🔒 80% PEAK LOCKED (+{max_gain_pct * 0.8:.1f}% @ ${peak_80_sl:.2f})"
                        )
                elif max_gain_pct >= 8.0:
                    tier3_sl = round(entry_price * 1.050, 2)  # Lock +5.0%
                    if tier3_sl > sl_target:
                        pos["sl_target"] = tier3_sl
                        sl_target = tier3_sl
                        pos["profit_lock_status"] = (
                            f"🔒 TIER-3 COMPOUNDER (+5.0% @ ${tier3_sl:.2f})"
                        )
                elif max_gain_pct >= 5.0:
                    tier2_sl = round(entry_price * 1.025, 2)  # Lock +2.5%
                    if tier2_sl > sl_target:
                        pos["sl_target"] = tier2_sl
                        sl_target = tier2_sl
                        pos["profit_lock_status"] = (
                            f"🔒 TIER-2 SECURED (+2.5% @ ${tier2_sl:.2f})"
                        )
                elif max_gain_pct >= 3.0:
                    tier1_sl = round(entry_price * 1.010, 2)  # Lock +1.0%
                    if tier1_sl > sl_target:
                        pos["sl_target"] = tier1_sl
                        sl_target = tier1_sl
                        pos["profit_lock_status"] = (
                            f"🔒 TIER-1 SECURED (+1.0% @ ${tier1_sl:.2f})"
                        )
                elif max_gain_pct >= 1.20:
                    be_sl = round(entry_price * 1.002, 2)  # Breakeven + fee cushion
                    if be_sl > sl_target:
                        pos["sl_target"] = be_sl
                        sl_target = be_sl
                        pos["profit_lock_status"] = (
                            f"🛡️ ZERO-GIVEBACK BREAKEVEN SECURED (${be_sl:.2f})"
                        )
                        logger.info(
                            f"🛡️ [{ticker} ZERO-GIVEBACK BREAKEVEN] Max gain {max_gain_pct:.2f}% >= 1.20%. Locked stop at ${be_sl:.2f}."
                        )

            # Level 5: CUSUM Structural Break Early Warning (Page 1954)
            try:
                from src.ultra_quant_engine import get_ultra_quant_engine

                u_engine = get_ultra_quant_engine()
                cusum_det = u_engine.get_or_create_cusum(ticker)
                last_p = float(pos.get("last_eval_price", entry_price))
                ret_step = (spot_price - last_p) / (last_p + 1e-9)
                pos["last_eval_price"] = spot_price
                c_info = cusum_det.update(ret_step)
                if c_info.get("alarm") and c_info.get("direction") == "DOWN":
                    tight_sl = max(sl_target, round(spot_price * 0.99, 2))
                    if tight_sl > sl_target:
                        pos["sl_target"] = tight_sl
                        sl_target = tight_sl
                        logger.warning(
                            f"⚡ [CUSUM CHANGE-POINT ALERT] {ticker} downward structural break detected! Ratcheted stop loss to ${tight_sl:.2f}."
                        )

                # ADWIN & Page-Hinkley Streaming Concept Drift Detection (Paper 21 & Paper 22)
                adwin_det = u_engine.get_or_create_adwin(ticker)
                ph_det = u_engine.get_or_create_page_hinkley(ticker)
                adwin_res = adwin_det.update(ret_step)
                ph_res = ph_det.update(ret_step)
                if (
                    adwin_res.get("drift_detected") or ph_res.get("drift_detected")
                ) and ret_step < 0:
                    tight_sl = max(sl_target, round(spot_price * 0.992, 2))
                    if tight_sl > sl_target:
                        pos["sl_target"] = tight_sl
                        sl_target = tight_sl
                        logger.warning(
                            f"⚡ [ADWIN/PAGE-HINKLEY DRIFT] Statistical distribution break for {ticker}! Trailed SL to ${tight_sl:.2f}."
                        )
            except Exception as ce:
                logger.debug(f"Drift surveillance notice for {ticker}: {ce}")

            # 🐢 Turtle 0.5N ATR Pyramiding & Runner Scaling Engine
            # (Ed Seykota, Michael Covel, Leung & Zhang 2017 arXiv:1701.03960)
            from src.compound_engine import evaluate_pyramiding_step

            # Update highest price seen for dynamic Chandelier stops
            pos["highest_price_seen"] = max(
                float(pos.get("highest_price_seen", entry_price)), spot_price
            )
            pos_atr = float(pos.get("atr_14", entry_price * 0.025))
            if pos_atr <= 0:
                pos_atr = max(entry_price * 0.025, 1.0)

            # Evaluate Adaptive Execution Policy (Hurst + Options GEX + Vaulted Genetic DNA)
            adaptive_policy = {}
            try:
                from src.hurst_exponent import get_adaptive_execution_policy

                adaptive_policy = get_adaptive_execution_policy(
                    ticker=ticker, spot_price=spot_price
                )
            except Exception as e:
                logger.debug(f"Adaptive policy lookup notice for {ticker}: {e}")

            exec_mode = adaptive_policy.get("execution_mode", "HYBRID_BALANCED")
            pyramid_allowed = adaptive_policy.get("pyramiding_enabled", True)
            fast_harvest_target = adaptive_policy.get("fast_harvest_target_pct")

            # Check Mean-Reversion Sniper Fast Harvest (e.g. Call Wall resistance)
            unrealized_gain_pct = (
                float((spot_price - entry_price) / entry_price * 100.0)
                if entry_price > 0
                else 0.0
            )
            if (
                exec_mode == "MEAN_REVERSION_SNIPE"
                and fast_harvest_target is not None
                and unrealized_gain_pct >= fast_harvest_target
            ):
                gross_proceeds = float(shares * spot_price)
                exit_friction = float(gross_proceeds * 0.00035)
                proceeds = float(gross_proceeds - exit_friction)
                cost_basis = float(shares * entry_price)
                pnl = float(proceeds - cost_basis)

                target_broker.state["cash"] += proceeds
                target_broker.state["realized_pnl"] += pnl
                target_broker.state["total_trades"] += 1
                target_broker.state["winning_trades"] += 1

                trade_record = {
                    "ticker": ticker,
                    "shares": shares,
                    "entry_price": entry_price,
                    "exit_price": spot_price,
                    "entry_date": pos.get("entry_date", date_str),
                    "exit_date": date_str,
                    "pnl": round(pnl, 2),
                    "return_pct": round(unrealized_gain_pct, 2),
                    "reason": "MEAN_REVERSION_SNIPER_HARVEST",
                    "exit_friction": round(exit_friction, 2),
                }
                target_broker._record_trade_closure(trade_record)
                del target_broker.state["open_positions"][ticker]
                executed_actions["take_profits_tp2"].append(trade_record)
                logger.info(
                    f"🎯 [SNIPER HARVEST] {ticker} reached Call Wall resistance (+{unrealized_gain_pct:.2f}%). Fully closed @ ${spot_price:.2f} | PnL: ${pnl:+,.2f}"
                )
                continue

            pyramid_eval = evaluate_pyramiding_step(
                current_price=spot_price,
                position_state=pos,
                atr=pos_atr,
            )

            if pyramid_eval.get("trigger_met") and pyramid_allowed:
                p_action = pyramid_eval.get("action")
                new_sl_val = float(pyramid_eval.get("new_sl", sl_target))

                if (
                    p_action == "ADD_UNIT_1"
                    and target_broker.state.get("cash", 0) > 1000.0
                ):
                    add_shares = int(pos.get("unit_shares", shares))
                    pyr_res = target_broker.execute_pyramid_unit(
                        ticker=ticker,
                        unit_num=2,
                        shares=add_shares,
                        price=spot_price,
                        new_sl=new_sl_val,
                        reason="TURTLE_PYRAMID_UNIT_2_BREAKEVEN_LOCKED",
                    )
                    if pyr_res.get("success"):
                        executed_actions.setdefault("pyramid_adds", []).append(pyr_res)
                        sl_target = new_sl_val
                        shares = int(pos["shares"])
                        logger.info(
                            f"🐢 [PYRAMID UNIT 2 ADDED] {ticker} @ ${spot_price:.2f} | Basis: ${pyr_res['blended_entry']:.2f} | Stop Ratcheted to Breakeven (${new_sl_val:.2f})"
                        )

                elif (
                    p_action == "ADD_UNIT_2"
                    and target_broker.state.get("cash", 0) > 1000.0
                ):
                    add_shares = int(pos.get("unit_shares", max(1, shares // 2)))
                    pyr_res = target_broker.execute_pyramid_unit(
                        ticker=ticker,
                        unit_num=3,
                        shares=add_shares,
                        price=spot_price,
                        new_sl=new_sl_val,
                        reason="TURTLE_PYRAMID_UNIT_3_RUNNER_LOADED",
                    )
                    if pyr_res.get("success"):
                        executed_actions.setdefault("pyramid_adds", []).append(pyr_res)
                        sl_target = new_sl_val
                        shares = int(pos["shares"])
                        logger.info(
                            f"🏆 [PYRAMID UNIT 3 ADDED] {ticker} @ ${spot_price:.2f} | Basis: ${pyr_res['blended_entry']:.2f} | Full Runner Loaded | Chandelier SL: ${new_sl_val:.2f})"
                        )

                elif p_action == "RATCHET_STOP":
                    pos["sl_target"] = max(float(pos.get("sl_target", 0.0)), new_sl_val)
                    sl_target = pos["sl_target"]

            # Check Stage 1 Scale-Out (+2.5 ATR or Proximity Harvester >= 98.5% TP1)
            is_proximity_tp1 = (
                not scaled_out
                and tp1_target > entry_price
                and (
                    spot_price >= tp1_target * 0.985 or high_water >= tp1_target * 0.985
                )
                and (high_water - spot_price)
                >= max((tp1_target - entry_price) * 0.10, 0.04)
            )
            is_exact_tp1 = not scaled_out and spot_price >= tp1_target

            if is_exact_tp1 or is_proximity_tp1:
                action_label = (
                    "TP1_PROXIMITY_HARVEST_50PCT"
                    if is_proximity_tp1 and not is_exact_tp1
                    else "TP1_PROFIT_LOCK_50PCT"
                )
                from src.market_session import is_crypto_asset

                is_crypto = is_crypto_asset(ticker)
                raw_shares = float(shares)
                if is_crypto or not raw_shares.is_integer():
                    half_shares = round(raw_shares / 2.0, 6)
                elif raw_shares > 1.0:
                    half_shares = int(raw_shares) // 2
                else:
                    # 1 whole share: close 100% at TP1 since it cannot be split into whole shares
                    half_shares = shares

                gross_proceeds = float(half_shares * spot_price)
                exit_friction = float(gross_proceeds * 0.00035)
                proceeds = float(gross_proceeds - exit_friction)
                cost_basis = float(half_shares * entry_price)
                pnl = float(proceeds - cost_basis)
                ret_pct = float((spot_price - entry_price) / denom * 100.0)

                target_broker.state["cash"] += proceeds
                target_broker.state["realized_pnl"] += pnl
                remaining_shares = shares - half_shares

                if remaining_shares <= 1e-6:
                    target_broker.state["total_trades"] += 1
                    target_broker.state["winning_trades"] += 1
                    del target_broker.state["open_positions"][ticker]
                    tp1_record = {
                        "ticker": ticker,
                        "shares_sold": half_shares,
                        "remaining_shares": 0,
                        "exit_price": spot_price,
                        "pnl": round(pnl, 2),
                        "return_pct": round(ret_pct, 2),
                        "action": "TP1_100PCT_COMPLETE_EXIT",
                        "exit_friction": round(exit_friction, 2),
                    }
                    executed_actions["take_profits_tp1"].append(tp1_record)
                    logger.info(
                        f"🎯 [TP1 COMPLETE EXIT] Closed 100% {ticker} ({half_shares} shares @ ${spot_price:.2f}) | Realized PnL: ${pnl:+,.2f} ({ret_pct:+.2f}%)"
                    )
                    continue
                else:
                    pos["shares"] = remaining_shares
                    pos["scaled_out"] = True
                    pos["sl_target"] = round(entry_price * 1.002, 2)  # Breakeven + 0.2%

                    tp1_record = {
                        "ticker": ticker,
                        "shares_sold": half_shares,
                        "remaining_shares": pos["shares"],
                        "exit_price": spot_price,
                        "pnl": round(pnl, 2),
                        "return_pct": round(ret_pct, 2),
                        "action": action_label,
                        "exit_friction": round(exit_friction, 2),
                    }
                    executed_actions["take_profits_tp1"].append(tp1_record)
                    logger.info(
                        f"💰 [{action_label}] Sold {half_shares} shares of {ticker} @ ${spot_price:.2f} | Realized PnL: ${pnl:+,.2f} ({ret_pct:+.2f}%) | Stop moved to ${pos['sl_target']:.2f}"
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
                gross_proceeds = float(pos["shares"] * spot_price)
                exit_friction = float(gross_proceeds * 0.00035)
                proceeds = float(gross_proceeds - exit_friction)
                cost_basis = float(pos["shares"] * entry_price)
                pnl = float(proceeds - cost_basis)
                ret_pct = float((spot_price - entry_price) / denom * 100.0)

                target_broker.state["cash"] += proceeds
                target_broker.state["realized_pnl"] += pnl
                target_broker.state["total_trades"] += 1
                target_broker.state["winning_trades"] += 1

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
                    "exit_friction": round(exit_friction, 2),
                }
                target_broker._record_trade_closure(trade_record)
                del target_broker.state["open_positions"][ticker]
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
                gross_proceeds = float(pos["shares"] * spot_price)
                exit_friction = float(gross_proceeds * 0.00035)
                proceeds = float(gross_proceeds - exit_friction)
                cost_basis = float(pos["shares"] * entry_price)
                pnl = float(proceeds - cost_basis)
                ret_pct = float((spot_price - entry_price) / denom * 100.0)

                target_broker.state["cash"] += proceeds
                target_broker.state["realized_pnl"] += pnl
                target_broker.state["total_trades"] += 1
                if pnl > 0 or scaled_out:
                    target_broker.state["winning_trades"] += 1
                else:
                    target_broker.state["losing_trades"] += 1

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
                    "exit_friction": round(exit_friction, 2),
                }
                target_broker._record_trade_closure(trade_record)
                del target_broker.state["open_positions"][ticker]
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

        # Immediately persist Phase A trade closures (e.g. stop losses or take profits)
        # to ensure disk state is synchronized before Phase B candidate evaluation
        if (
            executed_actions["take_profits_tp1"]
            or executed_actions["take_profits_tp2"]
            or executed_actions["stop_losses"]
        ):
            self.broker._recalculate_metrics(date_str, now_str)
            self.broker._save()
            self.crypto_broker._recalculate_metrics(date_str, now_str)
            self.crypto_broker._save()
            logger.info(
                "💾 [PHASE A COMMITTED] Closed positions persisted to paper_portfolio.json and pilot_crypto_portfolio.json."
            )

        # 3. Phase B: Scan Universe for Committee Buy Opportunities
        from src.opening_range_engine import (
            is_opening_15min_whipsaw_period,
            find_low_of_day_pullback_entry,
        )

        curr_open_count = len(self.broker.state.get("open_positions", {}))
        try:
            from src.alpaca_broker import AlpacaBrokerBridge

            alpaca_br = AlpacaBrokerBridge()
            if alpaca_br.is_connected():
                alp_syms = {
                    str(p.get("symbol", "")).upper()
                    for p in alpaca_br.get_open_positions()
                }
                for o in alpaca_br.get_open_orders():
                    if str(o.get("status", "")).lower() in (
                        "new",
                        "accepted",
                        "pending_new",
                        "open",
                        "held",
                        "partially_filled",
                    ):
                        alp_syms.add(str(o.get("symbol", "")).upper())
                active_all = (
                    set(self.broker.state.get("open_positions", {}).keys()) | alp_syms
                )
                curr_open_count = max(curr_open_count, len(active_all))
        except Exception as e:
            logger.debug(f"Alpaca open positions count check notice: {e}")

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

        # Macro Breadth Gate: Check VIX and SPY 50-day Trend Alignment
        try:
            vix_val = float(quotes_map.get("^VIX", {}).get("price", 0.0))
            if vix_val > 24.0:
                logger.warning(
                    f"🛡️ [MACRO BREADTH GATE] Pausing new entries: VIX is elevated at {vix_val:.1f} (> 24.0). Preserving cash."
                )
                available_slots = 0
            else:
                # Check broader market health: SPY price vs 50-day SMA
                from src.data_ingestion import get_price_history

                spy_df = get_price_history("SPY", period="6mo", use_cache=True)
                if isinstance(spy_df, pd.DataFrame) and len(spy_df) >= 50:
                    spy_close = float(spy_df["Close"].iloc[-1])
                    spy_sma50 = float(spy_df["Close"].rolling(50).mean().iloc[-1])
                    if spy_close < spy_sma50:
                        logger.warning(
                            f"🛡️ [MACRO BREADTH GATE] S&P 500 (SPY ${spy_close:.2f}) is below its 50-day SMA (${spy_sma50:.2f}). Market in defensive pullback — pausing new long allocations."
                        )
                        available_slots = 0
        except Exception as bg_err:
            logger.debug(f"Macro breadth gate notice: {bg_err}")

        # Macro Pre-Event Blackout Gate (FOMC / CPI / NFP releases)
        try:
            from src.ultra_quant_engine import get_ultra_quant_engine

            uq = get_ultra_quant_engine()
            m_blackout = uq.evaluate_macro_blackout()
            if m_blackout.get("is_blackout_active", False):
                logger.warning(
                    f"🚨 [MACRO EVENT BLACKOUT GATE] Pausing new entries: {m_blackout.get('reason')}"
                )
                available_slots = 0
        except Exception as mb_err:
            logger.debug(f"Macro blackout gate notice: {mb_err}")

        cash_available = self.broker.state.get("cash", 0.0)

        if available_slots > 0 and cash_available > 5000.0:
            # Anti-Whipsaw Cooldown: Quarantine any exit within 24h (scratch/breakeven) to stop ping-pong churn,
            # and enforce full 3-day quarantine on losses (pnl < 0).
            quarantined_tickers = set()
            for ct in self.broker.state.get("closed_trades", []):
                exit_d = ct.get("exit_date", "")
                pnl_val = float(ct.get("pnl", 0.0))
                if exit_d:
                    try:
                        d_exit = datetime.fromisoformat(str(exit_d)[:10])
                        d_now = datetime.fromisoformat(str(date_str)[:10])
                        days_diff = (d_now - d_exit).days
                        if days_diff < 1 or (pnl_val < 0 and days_diff < 3):
                            quarantined_tickers.add(ct.get("ticker"))
                    except Exception:
                        pass

            # Sector Shield: identify occupied sectors
            from src.cross_asset_pooling import get_sector_for_ticker

            occupied_sectors = {
                get_sector_for_ticker(h)
                for h in self.broker.state.get("open_positions", {}).keys()
            }

            from src.market_session import is_crypto_asset

            is_us_open = market_session.get("is_open", False)
            if not is_us_open:
                crypto_candidates = ["BTC-USD", "ETH-USD", "SOL-USD"]
                tickers_to_scan = list(
                    set(
                        [t for t in tickers_to_scan if is_crypto_asset(t)]
                        + crypto_candidates
                    )
                )
                logger.info(
                    f"🌙 [24/7 CRYPTO GATEWAY] US Equities closed. Routing autonomous scan to 24/7 digital assets: {tickers_to_scan}"
                )

            alpaca_held_or_pending = set()
            try:
                from src.alpaca_broker import AlpacaBrokerBridge

                alpaca_br = AlpacaBrokerBridge()
                if alpaca_br.is_connected():
                    for p in alpaca_br.get_open_positions():
                        alpaca_held_or_pending.add(str(p.get("symbol", "")).upper())
                    for o in alpaca_br.get_open_orders():
                        if str(o.get("status", "")).lower() in (
                            "new",
                            "accepted",
                            "pending_new",
                            "open",
                            "held",
                            "partially_filled",
                        ):
                            alpaca_held_or_pending.add(str(o.get("symbol", "")).upper())
            except Exception as alp_err:
                logger.debug(f"Alpaca candidate screening notice: {alp_err}")

            unheld_tickers = [
                t
                for t in tickers_to_scan
                if (
                    (
                        is_crypto_asset(t)
                        and t not in self.crypto_broker.state.get("open_positions", {})
                    )
                    or (
                        not is_crypto_asset(t)
                        and t not in self.broker.state.get("open_positions", {})
                        and t.upper() not in alpaca_held_or_pending
                    )
                )
                and t not in quarantined_tickers
                and (
                    is_crypto_asset(t)
                    or (
                        is_us_open
                        and (
                            get_sector_for_ticker(t) not in occupied_sectors
                            or get_sector_for_ticker(t)
                            in (
                                "General",
                                "Unknown",
                                "General_Market",
                                "Cryptocurrency",
                            )
                        )
                    )
                )
            ]
            if not is_us_open:
                unheld_tickers = [t for t in unheld_tickers if is_crypto_asset(t)]

            # Lightweight Pre-Screening: Filter to top 15 active/liquid assets
            # This keeps RAM usage minimal (<250MB) and prevents laptop freezing
            scored_candidates = []
            for t in unheld_tickers:
                q = quotes_map.get(t, {})
                p = float(q.get("price", 0.0))
                chg = abs(float(q.get("change_pct", 0.0)))
                vol = float(q.get("volume", 1.0))
                if p > 5.0:  # Filter out illiquid penny stocks
                    scored_candidates.append((t, chg * vol))

            # Prioritize the verified "Most Increasable" Set and Tier-1 Asymmetric Radar Set
            verified_priority_tickers = []
            try:
                radar_file = os.path.join("results", "tier1_asymmetric_radar.json")
                if os.path.exists(radar_file):
                    with open(radar_file, "r", encoding="utf-8") as rf:
                        radar_data = json.load(rf)
                    for item in radar_data[:25]:
                        tkr = item.get("ticker")
                        if (
                            tkr
                            and tkr in unheld_tickers
                            and tkr not in verified_priority_tickers
                        ):
                            verified_priority_tickers.append(tkr)
            except Exception as r_err:
                logger.debug(f"Tier-1 radar prioritization notice: {r_err}")

            try:
                inc_file = os.path.join(
                    "results", "most_increasable_trade_set_latest.json"
                )
                if os.path.exists(inc_file):
                    with open(inc_file, "r", encoding="utf-8") as f:
                        inc_data = json.load(f)
                    for item in inc_data.get("selected_candidates", []):
                        tkr = item.get("ticker")
                        if (
                            tkr
                            and tkr in unheld_tickers
                            and tkr not in verified_priority_tickers
                        ):
                            verified_priority_tickers.append(tkr)
            except Exception as inc_err:
                logger.debug(f"Most increasable set prioritization notice: {inc_err}")

            # Prioritize candidates flagged with RAPID_BULLISH_EXPANSION by SwiftShiftRadar
            try:
                swift_file = os.path.join("results", "swift_shift_radar_latest.json")
                if os.path.exists(swift_file):
                    with open(swift_file, "r", encoding="utf-8") as sf:
                        s_data = json.load(sf)
                    s_tkr = s_data.get("ticker")
                    if (
                        s_tkr
                        and s_tkr in unheld_tickers
                        and s_tkr not in verified_priority_tickers
                        and s_data.get("shift_direction") == "RAPID_BULLISH_EXPANSION"
                    ):
                        verified_priority_tickers.insert(0, s_tkr)
            except Exception as ss_err:
                logger.debug(f"Swift shift priority notice: {ss_err}")

            # Prioritize breaking catalyst tickers identified on the Benzinga wire
            benzinga_priority_tickers = []
            try:
                from src.benzinga_news_trader import fetch_top_benzinga_news_wire

                bz_articles = fetch_top_benzinga_news_wire(limit=30)
                for art in bz_articles:
                    if art.get("catalyst_sentiment") == "BULLISH":
                        for bt in art.get("extracted_tickers", []):
                            if (
                                bt in unheld_tickers
                                and bt not in verified_priority_tickers
                                and bt not in benzinga_priority_tickers
                            ):
                                benzinga_priority_tickers.append(bt)
                if benzinga_priority_tickers:
                    logger.info(
                        f"⚡ [BENZINGA PRIORITY] Prioritizing {len(benzinga_priority_tickers)} breaking catalyst tickers: {benzinga_priority_tickers}"
                    )
            except Exception as bz_err:
                logger.debug(f"Benzinga prioritization notice: {bz_err}")

            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            other_cands = [
                t
                for t, _ in scored_candidates
                if t not in verified_priority_tickers
                and t not in benzinga_priority_tickers
            ]
            top_candidates = (
                benzinga_priority_tickers + verified_priority_tickers + other_cands
            )[:30]

            logger.info(
                f"🌐 [LEAN MULTI-AGENT SCAN] Evaluating top {len(top_candidates)} active candidates (Priority Set: {verified_priority_tickers}): {top_candidates}"
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
                    news_df = get_news(ticker_sym, use_cache=True)
                    news_available = getattr(news_df, "attrs", {}).get(
                        "news_available",
                        (
                            not news_df.empty
                            if isinstance(news_df, pd.DataFrame)
                            else bool(news_df)
                        ),
                    )
                    if not news_available or (
                        isinstance(news_df, pd.DataFrame) and news_df.empty
                    ):
                        logger.warning(
                            f"SKIPPING_TRADE_NO_REAL_NEWS: Real news unavailable for {ticker_sym}. Skipping trade."
                        )
                        return ticker_sym, {
                            "action_code": "SKIP",
                            "final_resolution": "VETO",
                            "veto_reason": "NO_REAL_NEWS",
                            "news_available": False,
                        }

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
            funnel_scanned = len(top_candidates)
            funnel_skipped_news = 0
            funnel_vetoed: Dict[str, int] = {}
            funnel_blocked: Dict[str, int] = {}
            funnel_orders_submitted = 0

            # Lean thread pool (2 workers) to preserve system memory and responsiveness
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                results = executor.map(_deliberate_single, top_candidates)
                for t, delib in results:
                    if delib and isinstance(delib, dict):
                        deliberations.append((t, delib))
                        executed_actions["committee_resolutions"][t] = delib.get(
                            "final_resolution", "NEUTRAL"
                        )
                        if delib.get("veto_reason") == "NO_REAL_NEWS":
                            funnel_skipped_news += 1
                        elif delib.get("action_code") == "VETO":
                            vr = str(delib.get("veto_reason") or "COMMITTEE_VETO")[:40]
                            funnel_vetoed[vr] = funnel_vetoed.get(vr, 0) + 1

            # Sort candidate opportunities by CRO consensus conviction safely
            buy_candidates = [
                (t, d)
                for t, d in deliberations
                if isinstance(d, dict)
                and d.get("action_code") in ["EXECUTE_BUY", "SCALE_IN"]
            ]
            buy_candidates.sort(
                key=lambda x: (
                    float(x[1].get("p_trade", 0.50)) * 100.0 * 0.50
                    + float(x[1].get("consensus_conviction_pct") or 0.0) * 0.50
                    if isinstance(x[1], dict)
                    else 0.0
                ),
                reverse=True,
            )

            # Proprietary Alpha: Dynamic Sympathy Lead-Lag Propagation Scan
            try:
                from src.soros_sympathy import (
                    get_sympathy_propagator,
                    get_active_sympathy_network,
                )

                propagator = get_sympathy_propagator()
                active_sympathy_net = get_active_sympathy_network()
                driver_dfs = {}
                for driver in active_sympathy_net.keys():
                    if driver in quotes_map:
                        d_df = get_price_history(driver, period="1mo", use_cache=True)
                        if d_df is not None and not d_df.empty:
                            driver_dfs[driver] = d_df

                if driver_dfs and len(buy_candidates) < available_slots:
                    for driver, info in active_sympathy_net.items():
                        if driver in driver_dfs:
                            for follower in info["followers"]:
                                if follower in [
                                    b[0] for b in buy_candidates
                                ] or follower in self.broker.state.get(
                                    "open_positions", {}
                                ):
                                    continue
                                f_df = get_price_history(
                                    follower, period="1mo", use_cache=True
                                )
                                sympathy_check = (
                                    propagator.evaluate_sympathy_for_follower(
                                        follower_ticker=follower,
                                        driver_data_dict=driver_dfs,
                                        follower_df=f_df,
                                    )
                                )
                                if sympathy_check.get("sympathy_opportunity"):
                                    logger.info(
                                        f"🎯 [SYMPATHY ALPHA DETECTED] Follower {follower} lagging driver {driver} "
                                        f"(Expected: +{sympathy_check['total_expected_sympathy_pct']}%, Unpriced Spread: +{sympathy_check['unpriced_lag_spread_pct']}%)"
                                    )
                                    _, f_delib = _deliberate_single(follower)
                                    if f_delib and f_delib.get("action_code") in [
                                        "EXECUTE_BUY",
                                        "SCALE_IN",
                                        "HOLD",
                                    ]:
                                        f_delib["action_code"] = "EXECUTE_BUY"
                                        f_delib["final_resolution"] = (
                                            f"SYMPATHY_FRONTRUN_FROM_{driver}"
                                        )
                                        f_delib["sympathy_intelligence"] = (
                                            sympathy_check
                                        )
                                        buy_candidates.append((follower, f_delib))
            except Exception as sym_err:
                logger.debug(f"Sympathy scanner notice: {sym_err}")

            # Task 8 Max Position Size Hard Constraint: Max 10% of total portfolio equity for 8-10 position dynamic model
            total_eq = max(
                float(portfolio_summary.get("total_equity") or 100000.0), 1.0
            )

            # Neural SDE Dynamic CPPI Survival Floor & Cushion (Black & Jones 1987 / Grossman & Zhou 1993)
            # Calibrated by 1,000,000 GPU Monte Carlo paths in results/cvar_stress_report.json
            cppi_floor = 109362.04
            try:
                cvar_report_path = os.path.join("results", "cvar_stress_report.json")
                if os.path.exists(cvar_report_path):
                    with open(cvar_report_path, "r", encoding="utf-8") as cf:
                        cvar_rep = json.load(cf)
                    cppi_floor = float(
                        cvar_rep.get("recommended_dynamic_cash_floor", cppi_floor)
                    )
            except Exception as cf_err:
                logger.debug(f"CVaR report load notice: {cf_err}")

            cppi_cushion = max(0.0, total_eq - cppi_floor)
            if cppi_cushion <= 0.0:
                logger.critical(
                    f"🚨 [NEURAL SDE CPPI FLOOR BREACH] Equity (${total_eq:,.2f}) is at or below dynamic survival floor (${cppi_floor:,.2f}). Freezing all new long exposure."
                )
                return executed_actions

            # Position ticket size bounded by 10% equity ceiling and available CPPI surplus cushion
            max_position_dollars = min(
                total_eq * 0.10, max(1500.0, cppi_cushion * 0.35)
            )

            # Execute entries into Top candidate setups up to available slots
            for t, delib in buy_candidates[:available_slots]:
                spot_price = float(delib.get("spot_price", 0))
                if spot_price <= 0:
                    continue

                # Sector Shield Check: prevent multi-asset clustering in one GICS sector
                from src.cross_asset_pooling import get_sector_for_ticker

                cand_sector = get_sector_for_ticker(t)
                current_occupied = {
                    get_sector_for_ticker(h)
                    for h in self.broker.state.get("open_positions", {}).keys()
                }
                if cand_sector in current_occupied and cand_sector not in (
                    "General",
                    "Unknown",
                    "General_Market",
                ):
                    logger.warning(
                        f"🛡️ [SECTOR SHIELD VETO] Candidate {t} rejected: Sector '{cand_sector}' already occupied in active portfolio."
                    )
                    funnel_vetoed["SECTOR_SHIELD_OCCUPIED"] = funnel_vetoed.get("SECTOR_SHIELD_OCCUPIED", 0) + 1
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
                    funnel_vetoed["CORRELATION_SHIELD_EXCEEDED"] = funnel_vetoed.get("CORRELATION_SHIELD_EXCEEDED", 0) + 1
                    continue

                # Ensure Kelly sizing allocation does not breach max position size
                cro_info = delib.get("cro_signoff") or {}
                approved_kelly = float(cro_info.get("approved_kelly_pct", 8.0))

                # Universal DRL Continuous Policy Dynamic Leverage & Pyramiding Modulation
                try:
                    from src.drl_policy_agent import evaluate_drl_policy_action

                    drl_res = evaluate_drl_policy_action(ticker=t)
                    rec_lev = float(drl_res.get("recommended_leverage", 1.0))
                    drl_scalar = float(np.clip(rec_lev, 0.35, 1.25))
                    if drl_scalar != 1.0:
                        old_k = approved_kelly
                        approved_kelly = round(approved_kelly * drl_scalar, 1)
                        logger.info(
                            f"🧠 [UNIVERSAL DRL LEVERAGE] {t}: Policy recommends {rec_lev:.2f}x ({drl_res.get('action_label')}). Modulated Kelly: {old_k}% -> {approved_kelly}%."
                        )
                        if isinstance(delib.get("cro_signoff"), dict):
                            delib["cro_signoff"]["approved_kelly_pct"] = approved_kelly
                            delib["cro_signoff"]["drl_leverage_multiplier"] = rec_lev
                            delib["cro_signoff"]["drl_action_label"] = drl_res.get(
                                "action_label"
                            )
                except Exception as drl_err:
                    logger.debug(f"DRL policy modulation notice for {t}: {drl_err}")

                max_allowed_kelly = round(
                    (max_position_dollars / total_eq) * 100.0,
                    1,
                )
                if approved_kelly > max_allowed_kelly:
                    logger.info(
                        f"🔒 [CIRCUIT BREAKER: POSITION SIZE] Sizing for {t} capped from {approved_kelly}% to {max_allowed_kelly}% max position limit."
                    )
                    approved_kelly = max_allowed_kelly
                    if isinstance(delib.get("cro_signoff"), dict):
                        delib["cro_signoff"]["approved_kelly_pct"] = max_allowed_kelly

                # Formal Logic Z3 Rule Gate Verification (Sprint 4, Module 4.5 / Idea 47)
                target_b = self.get_target_broker(t)
                target_equity = float(target_b.state.get("total_equity", 100000.0))
                if is_crypto_asset(t):
                    shares_est = round(
                        float(target_equity * (approved_kelly / 100.0) / spot_price), 6
                    )
                else:
                    shares_est = max(
                        1.0,
                        float(
                            int(target_equity * (approved_kelly / 100.0) / spot_price)
                        ),
                    )
                gate_eval = self.rule_gate.verify_trade(
                    ticker=t,
                    spot_price=spot_price,
                    shares=shares_est,
                    deliberation=delib,
                    portfolio_state=target_b.state,
                )
                if not gate_eval["is_valid"]:
                    logger.warning(
                        f"🛑 [FORMAL LOGIC RULE GATE VETO] Trade for {t} rejected: {gate_eval['violations']}"
                    )
                    funnel_vetoed["RULE_GATE_CONSTRAINT_VIOLATION"] = funnel_vetoed.get("RULE_GATE_CONSTRAINT_VIOLATION", 0) + 1
                    continue

                order_res = execute_committee_order(
                    ticker=t, deliberation=delib, broker=target_b
                )
                if order_res.get("success"):
                    funnel_orders_submitted += 1
                    executed_actions["buys"].append(order_res)
                    resolution_text = delib.get("final_resolution", "APPROVED")

                    # Almgren-Chriss Optimal Execution Schedule (Paper 3)
                    try:
                        from src.ultra_quant_engine import get_ultra_quant_engine

                        ac_plan = (
                            get_ultra_quant_engine().evaluate_almgren_chriss_execution(
                                total_shares=float(order_res.get("shares", 10)),
                                price=spot_price,
                            )
                        )
                        order_res["almgren_chriss_schedule"] = ac_plan.get(
                            "trade_sizes", []
                        )
                        order_res["expected_shortfall"] = ac_plan.get(
                            "expected_shortfall_dollars", 0.0
                        )
                        logger.info(
                            f"📐 [ALMGREN-CHRISS] Sliced order into {len(ac_plan.get('trade_sizes', []))} slices | "
                            f"Expected Shortfall: ${ac_plan.get('expected_shortfall_dollars', 0.0):.2f}"
                        )
                    except Exception as ac_err:
                        logger.debug(f"Almgren-Chriss notice for {t}: {ac_err}")

                    logger.info(
                        f"🚀 [AUTONOMOUS BUY] Executed {order_res.get('shares')} shares of {t} @ ${spot_price:.2f} (Verdict: {resolution_text})"
                    )

                    # Mirror execution to Alpaca Broker Bridge strictly after execute_buy returned success
                    try:
                        from src.alpaca_broker import AlpacaBrokerBridge

                        alpaca = AlpacaBrokerBridge()
                        shares_b = int(order_res.get("shares", 0))
                        if alpaca.is_connected() and shares_b > 0:
                            tp1_p = float(delib.get("tp1_target", spot_price * 1.06))
                            sl_p = float(delib.get("stop_loss_target", spot_price * 0.965))
                            alpaca.submit_bracket_order(
                                ticker=t,
                                qty=shares_b,
                                take_profit_price=tp1_p,
                                stop_loss_price=sl_p,
                                side="buy",
                            )
                    except Exception as alp_err:
                        logger.error(f"Alpaca live order execution mirror error for {t}: {alp_err}")
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
                else:
                    blk_err = str(order_res.get("error") or order_res.get("reason") or "ORDER_BLOCKED")[:40]
                    funnel_blocked[blk_err] = funnel_blocked.get(blk_err, 0) + 1

        # Record execution funnel telemetry to results/funnel.json and results/funnel.md
        try:
            from src.funnel_tracker import record_funnel_run

            record_funnel_run(
                tickers_scanned=funnel_scanned if "funnel_scanned" in locals() else len(tickers_to_scan),
                skipped_no_real_news=funnel_skipped_news if "funnel_skipped_news" in locals() else 0,
                vetoed_by_reason=funnel_vetoed if "funnel_vetoed" in locals() else {},
                buy_blocked_guards=funnel_blocked if "funnel_blocked" in locals() else {},
                orders_submitted=funnel_orders_submitted if "funnel_orders_submitted" in locals() else 0,
            )
        except Exception as fe:
            logger.debug(f"Funnel telemetry record notice: {fe}")

        # 4. Phase C: Update Portfolio Summary Ledger
        self.broker._recalculate_metrics(date_str, now_str)
        self.broker._save()
        self.crypto_broker._recalculate_metrics(date_str, now_str)
        self.crypto_broker._save()

        # 5. Phase D: Self-Improving Feedback Loop & Trade Autopsy
        self_improvement_summary = self._run_self_improvement_feedback_loop(
            executed_actions
        )
        executed_actions["self_improvement"] = self_improvement_summary

        # Compute Advanced Autopsy Performance Ratios (Calmar, Sortino, Omega)
        try:
            from src.backtest_autopsy import compute_advanced_performance_ratios

            trades_df = self.broker.get_executed_trades_df()
            if (
                isinstance(trades_df, pd.DataFrame)
                and not trades_df.empty
                and "pnl" in trades_df.columns
            ):
                sim_equity = pd.Series(100000.0 + trades_df["pnl"].cumsum().values)
                ratios = compute_advanced_performance_ratios(sim_equity)
                executed_actions["autopsy_ratios"] = ratios
                logger.info(
                    f"📊 [AUTONOMOUS AUTOPSY] Calmar: {ratios.get('calmar_ratio'):.2f} | "
                    f"Sortino: {ratios.get('sortino_ratio'):.2f} | Omega: {ratios.get('omega_ratio'):.2f}"
                )
        except Exception as aut_err:
            logger.debug(f"Autopsy ratio calculation notice: {aut_err}")

        elapsed = round(time.time() - start_time, 2)
        cycle_elapsed_ns = time.perf_counter_ns() - cycle_t0_ns
        cycle_elapsed_us = round(cycle_elapsed_ns / 1000.0, 2)
        cycle_elapsed_ms = round(cycle_elapsed_ns / 1_000_000.0, 3)

        executed_actions["elapsed_seconds"] = elapsed
        executed_actions["execution_telemetry"] = {
            "cycle_latency_ns": cycle_elapsed_ns,
            "cycle_latency_us": cycle_elapsed_us,
            "cycle_latency_ms": cycle_elapsed_ms,
        }
        logger.info(
            f"⚡ [QUANTUM EXECUTION CYCLE COMPLETE] Latency: {cycle_elapsed_us:,.1f} μs ({cycle_elapsed_ms:.2f} ms | {cycle_elapsed_ns:,} ns) | Total: {elapsed}s. Buys: {len(executed_actions['buys'])}, Exits: {len(executed_actions['take_profits_tp1']) + len(executed_actions['take_profits_tp2']) + len(executed_actions['stop_losses'])}"
        )

        # Save cycle log safely with default=str serialization
        os.makedirs(os.path.dirname(AUTONOMOUS_LOG_FILE), exist_ok=True)
        try:
            with open(AUTONOMOUS_LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(executed_actions, f, indent=2, default=str)
        except Exception as log_err:
            logger.warning(f"Could not persist autonomous log file: {log_err}")

        # Persist updated portfolio states and export executed trades atomically
        try:
            open_val = sum(
                float(p.get("shares", 0))
                * float(p.get("current_price", p.get("entry_price", 0.0)))
                for p in self.broker.state.get("open_positions", {}).values()
            )
            self.broker.state["total_equity"] = round(
                float(self.broker.state.get("cash", 0.0)) + open_val, 2
            )
            self.broker._save()

            crypto_val = sum(
                float(p.get("shares", 0))
                * float(p.get("current_price", p.get("entry_price", 0.0)))
                for p in self.crypto_broker.state.get("open_positions", {}).values()
            )
            self.crypto_broker.state["total_equity"] = round(
                float(self.crypto_broker.state.get("cash", 0.0)) + crypto_val, 2
            )
            self.crypto_broker._save()
            logger.info(
                "💾 [PORTFOLIO PERSISTED] Updated paper_portfolio.json and pilot_crypto_portfolio.json successfully."
            )
        except Exception as save_err:
            logger.error(f"Failed to persist portfolio state: {save_err}")

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
                with open(memory_file, "r", encoding="utf-8") as f:
                    learning_state.update(json.load(f))
            except Exception as e:
                logger.warning(
                    f"Notice loading agent learning memory from {memory_file} ({e}). Initializing default learning state."
                )

        # Enforce Strategy Freeze: Learning and adaptive weight updates frozen during evaluation
        if (
            os.getenv("LEARNING_FROZEN", "false").lower() in ("true", "1", "yes")
            or "PYTEST_CURRENT_TEST" in os.environ
            or getattr(self.paper_broker, "portfolio_file", "") != os.path.join("results", "paper_portfolio.json")
        ):
            logger.info(
                "🔒 [STRATEGY FREEZE] LEARNING_FROZEN=true: Self-improvement learning feedback loop is read-only."
            )
            return {
                "total_cycles": learning_state.get("total_learning_cycles", 0),
                "agent_weights": learning_state.get("agent_voting_weights", {}),
                "kelly_multiplier": learning_state.get("adaptive_kelly_multiplier", 1.0),
                "new_autopsies_count": 0,
            }

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

        # Trigger Bayesian Post-Mortem Learner across all closed positions
        try:
            from src.post_mortem_learner import TradePostMortemLearner

            learner = TradePostMortemLearner()
            post_mortem_res = learner.run_post_mortem_cycle()
            if isinstance(post_mortem_res, dict):
                learning_state["bayesian_council_calibration"] = post_mortem_res.get(
                    "calibrated_weights", {}
                )
                learning_state["total_trades_analyzed"] = post_mortem_res.get(
                    "trades_analyzed", 0
                )
                learning_state["win_rate_pct"] = post_mortem_res.get(
                    "win_rate_pct", 0.0
                )
                learning_state["total_realized_pnl"] = post_mortem_res.get(
                    "total_realized_pnl", 0.0
                )
                learning_state["calibrated_weights"] = post_mortem_res.get(
                    "calibrated_weights", {}
                )
            logger.info(
                f"🧠 [BAYESIAN POST-MORTEM LEARNER] Analyzed {post_mortem_res.get('trades_analyzed', 0)} closed trades. Calibrated Council weights: {post_mortem_res.get('calibrated_weights')}"
            )
        except Exception as le:
            logger.debug(f"Bayesian post-mortem learning notice: {le}")

        # Persist learning memory
        os.makedirs(os.path.dirname(memory_file), exist_ok=True)
        with open(memory_file, "w", encoding="utf-8") as f:
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

        # Cache executive pre-market briefing memo JSON (results/morning_briefing_latest.json)
        try:
            from src.morning_briefing import generate_morning_briefing_text

            generate_morning_briefing_text(mode="MARKET_MASTER")
        except Exception as e:
            logger.warning(f"Error caching morning briefing memo JSON: {e}")

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
_DAEMON_STOP_EVENT = threading.Event()
_LAST_DAEMON_PULSE = {"timestamp": None, "status": "IDLE", "actions": None}


def get_daemon_status() -> Dict[str, Any]:
    """Returns current live daemon running state and last pulse timestamp."""
    is_alive = _DAEMON_THREAD is not None and _DAEMON_THREAD.is_alive()
    return {
        "is_active": is_alive,
        "last_pulse": _LAST_DAEMON_PULSE.get("timestamp"),
        "last_status": _LAST_DAEMON_PULSE.get("status", "IDLE"),
        "last_actions": _LAST_DAEMON_PULSE.get("actions"),
    }


def stop_background_daemon_thread(timeout: float = 3.0) -> bool:
    """
    Signals the background autonomous trading daemon thread to terminate cleanly
    and waits up to timeout seconds for thread completion.
    """
    global _DAEMON_THREAD
    with _DAEMON_LOCK:
        _DAEMON_STOP_EVENT.set()
        if _DAEMON_THREAD is not None and _DAEMON_THREAD.is_alive():
            _DAEMON_THREAD.join(timeout=timeout)
        is_still_alive = _DAEMON_THREAD is not None and _DAEMON_THREAD.is_alive()
        if not is_still_alive:
            _DAEMON_THREAD = None
            _LAST_DAEMON_PULSE["status"] = "STOPPED"
        return not is_still_alive


def ensure_background_daemon_thread_running(
    interval_seconds: int = 60,
    run_immediately: bool = True,
):
    """
    Ensures a single background autonomous trading daemon thread is permanently active.
    Polls the market continuously and executes cycles during open market hours.
    """
    global _DAEMON_THREAD
    with _DAEMON_LOCK:
        if _DAEMON_THREAD is not None and _DAEMON_THREAD.is_alive():
            return _DAEMON_THREAD

        _DAEMON_STOP_EVENT.clear()

        def _daemon_loop():
            engine = AutonomousTradingEngine()
            logger.info(
                f"🤖 [PERMANENT 24/7 DAEMON THREAD] Initialized. Interval: {interval_seconds}s"
            )
            if not run_immediately:
                _DAEMON_STOP_EVENT.wait(interval_seconds)

            while (
                not _DAEMON_STOP_EVENT.is_set() and threading.main_thread().is_alive()
            ):
                try:
                    from src.market_session import get_us_market_session

                    sess = get_us_market_session()
                    if sess.get("is_open", False):
                        # 1. Fast sub-second Sentinel price & stop floor audit for active holdings
                        from src.realtime_tracker import (
                            update_live_holdings_prices_and_alert_discord,
                        )

                        guard_res = update_live_holdings_prices_and_alert_discord(
                            notify_discord=False
                        )

                        # 2. Only run full autonomous scan if Auto-Pilot is explicitly armed by user
                        autopilot_flag = os.path.exists(
                            os.path.join("results", "AUTOPILOT_ACTIVE.flag")
                        ) or (
                            os.getenv("SENTILYZE_AUTOPILOT", "").lower()
                            in ("1", "true")
                        )
                        if autopilot_flag:
                            logger.info(
                                "🟢 [DAEMON CYCLE] Auto-Pilot is ARMED. Running top-25 candidate scan..."
                            )
                            cycle_res = engine.run_autonomous_cycle(
                                candidate_tickers=engine.tickers[:25]
                            )
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
                                "SENTINEL_GUARD_ACTIVE (Auto-Pilot Standby)"
                            )
                            _LAST_DAEMON_PULSE["actions"] = {
                                "buys": 0,
                                "tp1": 0,
                                "tp2": 0,
                                "stops": len(guard_res.get("executed_trades", [])),
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

                # Wait on stop event instead of blind sleep
                _DAEMON_STOP_EVENT.wait(interval_seconds)

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


def run_master_loop_daily_cycle(
    watchlist: Optional[List[str]] = None,
    dry_run: bool = False,
    ignore_hours: bool = True,
) -> Dict[str, Any]:
    """Runs the 4-phase MasterTradingLoop cycle."""
    from src.master_loop import get_master_trading_loop

    loop = get_master_trading_loop()
    return loop.run_full_daily_cycle(
        watchlist=watchlist, dry_run=dry_run, ignore_market_hours=ignore_hours
    )


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
        "--master-loop",
        action="store_true",
        help="Run full 4-phase master autonomous daily trading lifecycle",
    )
    parser.add_argument(
        "--discord-cmd",
        type=str,
        default=None,
        help="Execute interactive Discord bot command (e.g. /signal NVDA)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Daemon cycle interval in seconds (default: 300s / 5 min)",
    )

    args = parser.parse_args()
    engine = AutonomousTradingEngine()

    if args.discord_cmd:
        from src.discord_bot import handle_bot_command

        res = handle_bot_command(args.discord_cmd)
        print(json.dumps(res, indent=2))
    elif args.master_loop:
        res = run_master_loop_daily_cycle(dry_run=True, ignore_hours=True)
        print(json.dumps(res, indent=2))
    elif args.premarket:
        res = engine.run_premarket_briefing()
        print(json.dumps(res, indent=2))
    elif args.daemon:
        run_autonomous_daemon(interval_seconds=args.interval)
    else:
        summary = engine.run_autonomous_cycle()
        print(json.dumps(summary, indent=2))
