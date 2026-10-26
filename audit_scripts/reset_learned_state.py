import json
import os
import sys

sys.path.insert(0, ".")

# 1. Reset results/agent_learning_memory.json to default constants in src/autonomous_trader.py
# Default state defined at src/autonomous_trader.py:1724
default_learning_memory = {
    "total_learning_cycles": 0,
    "last_updated": "2026-11-02T00:00:00+00:00",
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

with open("results/agent_learning_memory.json", "w", encoding="utf-8") as f:
    json.dump(default_learning_memory, f, indent=2)

print("Reset results/agent_learning_memory.json to code defaults.")

# 2. Reset results/agent_memory/committee_weights.json to DEFAULT_WEIGHTS from src/post_mortem_learner.py:27
from src.post_mortem_learner import DEFAULT_WEIGHTS

with open("results/agent_memory/committee_weights.json", "w", encoding="utf-8") as f:
    json.dump(DEFAULT_WEIGHTS, f, indent=2)

print("Reset results/agent_memory/committee_weights.json to DEFAULT_WEIGHTS.")

# 3. Reset results/agent_memory/semantic_rules.json to DEFAULT_SEMANTIC_RULES from src/agent_memory.py:33
from src.agent_memory import DEFAULT_SEMANTIC_RULES

with open("results/agent_memory/semantic_rules.json", "w", encoding="utf-8") as f:
    json.dump(DEFAULT_SEMANTIC_RULES, f, indent=2)

print("Reset results/agent_memory/semantic_rules.json to DEFAULT_SEMANTIC_RULES.")

# 4. Reset results/agent_memory/trade_postmortems.jsonl to empty file (unlearned, neutral 0 trades)
with open("results/agent_memory/trade_postmortems.jsonl", "w", encoding="utf-8") as f:
    pass

print("Reset results/agent_memory/trade_postmortems.jsonl to empty (0 trades).")

# 5. Remove or empty results/agent_memory/session_episodic_memory.json
if os.path.exists("results/agent_memory/session_episodic_memory.json"):
    with open(
        "results/agent_memory/session_episodic_memory.json", "w", encoding="utf-8"
    ) as f:
        json.dump(
            {"project_name": "Sentilyze", "status": "RESET_TO_DEFAULTS"}, f, indent=2
        )
    print("Reset results/agent_memory/session_episodic_memory.json.")
