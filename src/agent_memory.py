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
        direction: str,
        agent_votes: Dict[str, str],
        outcome: str,
        r_multiple: float = 0.0,
        pnl_pct: float = 0.0,
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
            notes: Post-mortem reflection
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker.upper(),
            "direction": direction.upper(),
            "agent_votes": agent_votes,
            "outcome": outcome.upper(),
            "r_multiple": round(r_multiple, 2),
            "pnl_pct": round(pnl_pct, 2),
            "notes": notes,
        }

        try:
            with open(self.postmortems_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            logger.info(f"Recorded trade post-mortem for {ticker} (Outcome: {outcome})")
            self.recalibrate_weights()
        except Exception as e:
            logger.error(f"Failed to record post-mortem: {e}")

        return entry

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

    def recalibrate_weights(self) -> Dict[str, float]:
        """Calculates empirical win rates per agent and updates committee voting weights."""
        postmortems = self.get_recent_postmortems(limit=30)
        if len(postmortems) < 3:
            return self.get_calibrated_weights()

        agent_scores = {
            agent: {"correct": 0, "total": 0} for agent in DEFAULT_AGENT_WEIGHTS
        }

        for p in postmortems:
            outcome = p.get("outcome", "SCRATCH")
            votes = p.get("agent_votes", {})
            trade_dir = p.get("direction", "LONG")
            is_win = outcome == "WIN" or p.get("pnl_pct", 0) > 0

            for agent, vote in votes.items():
                clean_agent = self._match_agent_name(agent)
                if not clean_agent:
                    continue

                agent_scores[clean_agent]["total"] += 1
                v_up = str(vote).upper()

                if is_win:
                    if ("BUY" in v_up and trade_dir == "LONG") or (
                        "SELL" in v_up and trade_dir == "SHORT"
                    ):
                        agent_scores[clean_agent]["correct"] += 1
                else:
                    if "SELL" in v_up or "HOLD" in v_up or "VETO" in v_up:
                        agent_scores[clean_agent]["correct"] += 1

        raw_weights = {}
        for agent, counts in agent_scores.items():
            tot = counts["total"]
            win_rate = (counts["correct"] / tot) if tot > 0 else 0.50
            scaled_weight = max(0.10, min(0.40, win_rate))
            raw_weights[agent] = scaled_weight

        total_raw = sum(raw_weights.values())
        calibrated = {k: round(v / total_raw, 4) for k, v in raw_weights.items()}

        try:
            with open(self.weights_file, "w", encoding="utf-8") as f:
                json.dump(calibrated, f, indent=2)
            logger.info(f"Calibrated agent committee weights: {calibrated}")
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
