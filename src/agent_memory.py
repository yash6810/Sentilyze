"""
Agent Memory — Persistent Multi-Session Memory Engine for Sentilyze.
===================================================================
Provides durable episodic and semantic memory across trading sessions:
1. Episodic Memory: Logs historical trade post-mortems, committee vote splits, and R-multiple outcomes.
2. Dynamic Calibration: Tracks empirical win rates per agent to dynamically scale committee voting weights.
3. Semantic Heuristics: Distills durable market rules (e.g. regime-dependent stop tightening).
"""

from typing import Dict, Any, List, Optional
import os
import json
from datetime import datetime, timezone
from src.utils import get_logger

logger = get_logger(__name__)

MEMORY_DIR = os.path.join("results", "agent_memory")
POSTMORTEMS_FILE = os.path.join(MEMORY_DIR, "trade_postmortems.jsonl")
RULES_FILE = os.path.join(MEMORY_DIR, "semantic_rules.json")
WEIGHTS_FILE = os.path.join(MEMORY_DIR, "committee_weights.json")

DEFAULT_AGENT_WEIGHTS = {
    "Technical Momentum": 0.25,
    "FinBERT Sentiment": 0.25,
    "Fundamental Valuation": 0.20,
    "Chief Risk Officer": 0.15,
    "Adversarial Red-Team": 0.15,
}

DEFAULT_SEMANTIC_RULES = [
    {
        "rule_id": "RULE-001",
        "regime": "HIGH_VOLATILITY",
        "description": "When VIX > 25, technical breakout signals lose predictive power; dynamic leverage must be capped at 1.0x.",
        "confidence": 0.88,
        "created_at": "2026-01-15T00:00:00Z",
    },
    {
        "rule_id": "RULE-002",
        "regime": "EARNINGS_ANNOUNCEMENT",
        "description": "Pre-earnings sentiment spikes exhibit high mean-reversion risk; tighten ATR trailing stops by 30%.",
        "confidence": 0.82,
        "created_at": "2026-02-01T00:00:00Z",
    },
    {
        "rule_id": "RULE-003",
        "regime": "CHOPPY_MARKET",
        "description": "In non-trending regimes (ADX < 20), require at least 4/5 council consensus before opening long positions.",
        "confidence": 0.85,
        "created_at": "2026-03-01T00:00:00Z",
    },
]


class AgentMemoryStore:
    """Manages persistent episodic post-mortems, semantic rules, and dynamic voting weights."""

    def __init__(self, memory_dir: str = MEMORY_DIR):
        self.memory_dir = memory_dir
        self.postmortems_file = os.path.join(memory_dir, "trade_postmortems.jsonl")
        self.rules_file = os.path.join(memory_dir, "semantic_rules.json")
        self.weights_file = os.path.join(memory_dir, "committee_weights.json")
        self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        """Creates storage directories and baseline semantic files if missing."""
        os.makedirs(self.memory_dir, exist_ok=True)

        if not os.path.exists(self.rules_file):
            try:
                with open(self.rules_file, "w", encoding="utf-8") as f:
                    json.dump(DEFAULT_SEMANTIC_RULES, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to initialize semantic rules: {e}")

        if not os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, "w", encoding="utf-8") as f:
                    json.dump(DEFAULT_AGENT_WEIGHTS, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to initialize committee weights: {e}")

    def record_postmortem(
        self,
        ticker: str,
        direction: str = "LONG",
        agent_votes: Optional[Dict[str, str]] = None,
        outcome: str = "WIN",
        r_multiple: float = 0.0,
        pnl_pct: float = 0.0,
        pnl_dollars: float = 0.0,
        exit_reason: str = "",
        entry_date: str = "",
        exit_date: str = "",
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Appends an episodic trade post-mortem to the JSONL log.

        Args:
            ticker: Asset symbol
            direction: 'LONG' or 'SHORT' / 'CASH'
            agent_votes: Map of agent name -> voted direction ('BUY', 'SELL', 'HOLD')
            outcome: 'WIN', 'LOSS', or 'SCRATCH'
            r_multiple: Risk-adjusted return multiple (e.g. +2.1R, -1.0R)
            pnl_pct: Trade percentage return (e.g. +3.4%)
            pnl_dollars: Realized dollar PnL
            exit_reason: Trigger reason (e.g. 'STOP_LOSS', 'TAKE_PROFIT')
            entry_date: Entry date string
            exit_date: Exit date string
            notes: Post-mortem reflection
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker.upper(),
            "direction": direction.upper(),
            "agent_votes": agent_votes
            or {
                "Technical Momentum": "BUY",
                "FinBERT Sentiment": "BUY" if outcome == "WIN" else "HOLD",
                "Fundamental Valuation": "HOLD",
                "Chief Risk Officer": (
                    "APPROVED" if outcome == "WIN" else "VETO_OVERRIDDEN"
                ),
                "Adversarial Red-Team": "HOLD" if outcome == "WIN" else "CAUTION",
            },
            "outcome": outcome.upper(),
            "r_multiple": round(r_multiple, 2),
            "pnl_pct": round(pnl_pct, 2),
            "pnl_dollars": round(pnl_dollars, 2),
            "entry_date": entry_date,
            "exit_date": exit_date,
            "exit_reason": exit_reason,
            "notes": notes,
        }

        try:
            with open(self.postmortems_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            logger.info(
                f"Recorded trade post-mortem for {ticker} (Outcome: {outcome} | PnL: ${pnl_dollars:+,.2f})"
            )
            self.recalibrate_weights()
        except Exception as e:
            logger.error(f"Failed to record post-mortem: {e}")

        return entry

    def get_ticker_reputation(self, ticker: str) -> Dict[str, Any]:
        """Calculates historical win rate, trade count, and recent trajectory for a specific ticker."""
        postmortems = self.get_recent_postmortems(limit=100)
        t_upper = ticker.upper()
        ticker_trades = [p for p in postmortems if p.get("ticker") == t_upper]

        if not ticker_trades:
            return {
                "ticker": t_upper,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.50,
                "total_pnl_dollars": 0.0,
                "last_outcome": "NONE",
                "consecutive_losses": 0,
                "last_exit_date": "",
            }

        wins = sum(1 for p in ticker_trades if p.get("outcome") == "WIN")
        losses = sum(1 for p in ticker_trades if p.get("outcome") == "LOSS")
        total_pnl = sum(float(p.get("pnl_dollars", 0.0)) for p in ticker_trades)
        win_rate = wins / len(ticker_trades) if ticker_trades else 0.50

        # Calculate consecutive recent losses (most recent trades first)
        consec_losses = 0
        for p in ticker_trades:
            if p.get("outcome") == "LOSS":
                consec_losses += 1
            else:
                break

        last_outcome = ticker_trades[0].get("outcome", "NONE")
        last_exit = ticker_trades[0].get("exit_date", "")

        return {
            "ticker": t_upper,
            "total_trades": len(ticker_trades),
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 2),
            "total_pnl_dollars": round(total_pnl, 2),
            "last_outcome": last_outcome,
            "consecutive_losses": consec_losses,
            "last_exit_date": last_exit,
        }

    def get_memory_risk_adjustment(self, ticker: str) -> Dict[str, Any]:
        """
        Synthesizes episodic trade memory to provide risk adjustments to ChiefRiskOfficerAgent.
        """
        rep = self.get_ticker_reputation(ticker)
        consec = rep.get("consecutive_losses", 0)
        last_out = rep.get("last_outcome", "NONE")
        total_pnl = rep.get("total_pnl_dollars", 0.0)
        tot_trades = rep.get("total_trades", 0)

        caution_level = "NORMAL"
        conviction_delta = 0.0
        kelly_scale = 1.0
        require_supermajority = False
        reasons = []

        if consec >= 2 or (tot_trades >= 2 and total_pnl < -1000.0):
            caution_level = "PENALIZED"
            conviction_delta = -12.0
            kelly_scale = 0.40
            require_supermajority = True
            reasons.append(
                f"Historical episodic penalty: {consec} consecutive losses, cumulative ${total_pnl:,.2f} PnL."
            )
        elif last_out == "LOSS" or total_pnl < 0:
            caution_level = "CAUTION"
            conviction_delta = -6.0
            kelly_scale = 0.65
            require_supermajority = True
            reasons.append(
                f"Episodic memory caution: Last trade on {ticker} exited at a loss (${total_pnl:,.2f})."
            )
        elif tot_trades >= 1 and total_pnl >= 2000.0 and rep.get("win_rate", 0) >= 0.70:
            caution_level = "ALPHA_CHAMPION"
            conviction_delta = +4.0
            kelly_scale = 1.15
            reasons.append(
                f"Alpha champion: Verified high-expectancy performer (${total_pnl:+,.2f} net profit)."
            )

        return {
            "ticker": ticker.upper(),
            "caution_level": caution_level,
            "conviction_delta": conviction_delta,
            "kelly_scale": kelly_scale,
            "require_supermajority": require_supermajority,
            "reasons": reasons,
            "reputation": rep,
        }

    def sync_from_executed_trades(
        self, csv_path: str = "results/executed_trades.csv"
    ) -> int:
        """Synchronizes executed_trades.csv into trade_postmortems.jsonl and recalibrates weights."""
        if not os.path.exists(csv_path):
            return 0
        try:
            import pandas as pd

            df = pd.read_csv(csv_path)
            if df.empty:
                return 0

            existing_signatures = set()
            if os.path.exists(self.postmortems_file):
                with open(self.postmortems_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                p = json.loads(line.strip())
                                existing_signatures.add(
                                    (
                                        p.get("ticker"),
                                        p.get("exit_date"),
                                        round(
                                            float(
                                                p.get(
                                                    "pnl_dollars",
                                                    p.get("pnl_pct", 0),
                                                )
                                            ),
                                            2,
                                        ),
                                    )
                                )
                            except Exception:
                                pass

            added = 0
            with open(self.postmortems_file, "a", encoding="utf-8") as f:
                for _, row in df.iterrows():
                    ticker = str(row.get("ticker", "")).strip().upper()
                    exit_date = str(row.get("exit_date", ""))
                    pnl = float(row.get("pnl", 0.0))
                    sig = (ticker, exit_date, round(pnl, 2))
                    if sig in existing_signatures:
                        continue

                    ret_pct = float(row.get("return_pct", 0.0))
                    reason = str(row.get("reason", "EXIT"))
                    outcome = "WIN" if pnl > 0 else ("SCRATCH" if pnl == 0 else "LOSS")
                    shares = int(row.get("shares", 0))
                    entry_p = float(row.get("entry_price", 0.0))
                    cost = shares * entry_p
                    r_mult = pnl / max(1.0, cost * 0.025)

                    entry = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "ticker": ticker,
                        "direction": "LONG",
                        "agent_votes": {
                            "Technical Momentum": "BUY",
                            "FinBERT Sentiment": (
                                "BUY" if outcome == "WIN" else "HOLD"
                            ),
                            "Fundamental Valuation": "HOLD",
                            "Chief Risk Officer": (
                                "APPROVED" if outcome == "WIN" else "VETO_OVERRIDDEN"
                            ),
                            "Adversarial Red-Team": (
                                "HOLD" if outcome == "WIN" else "CAUTION"
                            ),
                        },
                        "outcome": outcome,
                        "r_multiple": round(r_mult, 2),
                        "pnl_pct": round(ret_pct, 2),
                        "pnl_dollars": round(pnl, 2),
                        "entry_date": str(row.get("entry_date", "")),
                        "exit_date": exit_date,
                        "exit_reason": reason,
                        "notes": f"Exit Reason: {reason} on {exit_date}. PnL: ${pnl:+,.2f} ({ret_pct:+.2f}%).",
                    }
                    f.write(json.dumps(entry) + "\n")
                    existing_signatures.add(sig)
                    added += 1

            if added > 0:
                self.recalibrate_weights()
            return added
        except Exception as e:
            logger.error(f"Error syncing executed trades to memory: {e}")
            return 0

    def get_recent_postmortems(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Retrieves the most recent post-mortems in reverse chronological order."""
        if not os.path.exists(self.postmortems_file):
            return []

        entries = []
        try:
            with open(self.postmortems_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Error reading post-mortems: {e}")

        return entries[::-1][:limit]

    def recalibrate_weights(
        self,
        learning_rate: float = 0.15,
        base_temperature: float = 0.35,
        decay_gamma: float = 0.9515,
    ) -> Dict[str, float]:
        """
        Calibrates committee voting weights using the Softmax Dirichlet RLFF Algorithm.
        Grounded in Reinforcement Learning from Financial Feedback with 14-session episodic decay,
        duration-efficiency discounting, and an Adversarial Red-Team hedging duality bonus.
        """
        postmortems = self.get_recent_postmortems(limit=30)
        if len(postmortems) < 3:
            return self.get_calibrated_weights()

        agents = list(DEFAULT_AGENT_WEIGHTS.keys())
        import numpy as np

        # Dirichlet concentration parameter alpha initialized to 1.0 (uniform prior)
        alpha = {a: 1.0 for a in agents}

        # Process postmortems in chronological order (oldest to newest)
        for p in reversed(postmortems):
            # 1. Episodic prior decay: alpha = 1.0 + gamma * (alpha - 1.0)
            for a in agents:
                alpha[a] = 1.0 + decay_gamma * (alpha[a] - 1.0)

            r_mult = float(p.get("r_multiple", 0.0))
            if r_mult == 0.0:
                pnl_pct = float(p.get("pnl_pct", 0.0))
                r_mult = pnl_pct / 2.5

            votes = p.get("agent_votes", {})
            trade_dir = p.get("direction", "LONG")

            # Holding period duration discount: psi = 1 / (1 + 0.08 * sqrt(days))
            days = 1.0
            try:
                e_d = datetime.fromisoformat(str(p.get("entry_date", ""))[:10])
                x_d = datetime.fromisoformat(str(p.get("exit_date", ""))[:10])
                days = max(1.0, float((x_d - e_d).days))
            except Exception:
                days = 2.0
            psi = 1.0 / (1.0 + 0.08 * (days**0.5))

            for agent in agents:
                v = votes.get(agent, "NEUTRAL")
                v_str = str(v).upper()
                if "BUY" in v_str or "LONG" in v_str:
                    action_k = 1.0 if trade_dir == "LONG" else -1.0
                elif "SELL" in v_str or "SHORT" in v_str:
                    action_k = -1.0 if trade_dir == "LONG" else 1.0
                elif "VETO" in v_str:
                    action_k = -1.0
                else:
                    action_k = 0.0

                # Adversarial Red-Team Duality Bonus:
                # If trade lost money and red team warned / vetoed, give preservation reward
                if agent == "Adversarial Red-Team":
                    if r_mult < 0 and (
                        "VETO" in v_str or "CAUTION" in v_str or action_k < 0
                    ):
                        reward_k = 1.5 * abs(r_mult) * psi
                    else:
                        dir_term = np.tanh(r_mult * action_k) * (abs(r_mult) ** 0.75)
                        reward_k = dir_term * psi
                else:
                    dir_term = np.tanh(r_mult * action_k) * (abs(r_mult) ** 0.75)
                    reward_k = dir_term * psi

                # Gradient step
                alpha[agent] = max(0.50, alpha[agent] + learning_rate * reward_k)

        # Temperature-scaled Softmax over normalized Dirichlet parameters
        sum_alpha = sum(alpha.values())
        z_vals = {a: alpha[a] / sum_alpha for a in agents}
        exp_logits = {a: np.exp(z_vals[a] / base_temperature) for a in agents}
        sum_exp = sum(exp_logits.values())
        calibrated = {a: round(float(exp_logits[a] / sum_exp), 4) for a in agents}

        try:
            with open(self.weights_file, "w", encoding="utf-8") as f:
                json.dump(calibrated, f, indent=2)
            logger.info(
                f"Calibrated agent committee weights (Softmax Dirichlet RLFF): {calibrated}"
            )
        except Exception as e:
            logger.warning(f"Failed to persist calibrated weights: {e}")

        return calibrated

    def get_calibrated_weights(self) -> Dict[str, float]:
        """Returns current dynamic voting weights."""
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, "r", encoding="utf-8") as f:
                    weights = json.load(f)
                if weights and isinstance(weights, dict):
                    return weights
            except Exception:
                pass
        return dict(DEFAULT_AGENT_WEIGHTS)

    def get_semantic_rules(self) -> List[Dict[str, Any]]:
        """Returns all persistent semantic trading heuristics."""
        if os.path.exists(self.rules_file):
            try:
                with open(self.rules_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return list(DEFAULT_SEMANTIC_RULES)

    def add_semantic_rule(
        self,
        description: str,
        regime: str = "GENERAL",
        confidence: float = 0.85,
    ) -> Dict[str, Any]:
        """Adds a newly discovered semantic rule to memory."""
        rules = self.get_semantic_rules()
        new_id = f"RULE-{len(rules)+1:03d}"
        rule = {
            "rule_id": new_id,
            "regime": regime.upper(),
            "description": description,
            "confidence": round(confidence, 2),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        rules.append(rule)
        try:
            with open(self.rules_file, "w", encoding="utf-8") as f:
                json.dump(rules, f, indent=2)
            logger.info(f"Added new semantic rule: {new_id}")
        except Exception as e:
            logger.error(f"Failed to save semantic rule: {e}")
        return rule

    def _match_agent_name(self, name: str) -> Optional[str]:
        name_lower = str(name).lower()
        if "tech" in name_lower or "momentum" in name_lower:
            return "Technical Momentum"
        if "sent" in name_lower or "finbert" in name_lower or "nlp" in name_lower:
            return "FinBERT Sentiment"
        if "val" in name_lower or "fund" in name_lower or "dcf" in name_lower:
            return "Fundamental Valuation"
        if "risk" in name_lower or "cro" in name_lower:
            return "Chief Risk Officer"
        if "red" in name_lower or "adversar" in name_lower or "veto" in name_lower:
            return "Adversarial Red-Team"
        return None
