import sys, os

sys.path.insert(0, ".")
import json
from src.agent_committee import ChiefRiskOfficerAgent

cro = ChiefRiskOfficerAgent()
mock_reports = [
    {
        "agent_name": "Technical Momentum Specialist",
        "vote": "BUY",
        "conviction_score": 80.0,
    },
    {
        "agent_name": "Sentiment & Alternative Data Specialist",
        "vote": "BUY",
        "conviction_score": 75.0,
    },
    {
        "agent_name": "Forensic & Valuation Auditor",
        "vote": "BUY",
        "conviction_score": 70.0,
        "key_metrics": {"data_available": True},
    },
]

signoff = cro.evaluate_and_sign_off("NVDA", 220.0, mock_reports, vix_level=15.0)
print("ACTION_CODE:", signoff.get("action_code"))
print("VETO_REASON:", signoff.get("veto_reason"))
print("FINAL_RESOLUTION:", signoff.get("final_resolution"))
for k, v in signoff.items():
    if "veto" in k or "trigger" in k:
        print(f"  {k}: {v}")
