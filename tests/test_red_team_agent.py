"""
Tests for Adversarial Red-Team Specialist Agent (src/red_team_agent.py).
Verifies vulnerability checks, stress-testing, and CRO veto flags.
"""

import pytest
from src.red_team_agent import AdversarialRedTeamAgent


def test_red_team_agent_initialization():
    agent = AdversarialRedTeamAgent()
    assert "Red-Team" in agent.name


def test_red_team_evaluation_structure():
    agent = AdversarialRedTeamAgent()
    result = agent.evaluate("NVDA", spot_price=120.0)

    assert result["agent_name"] == "Adversarial Red-Team Specialist"
    assert result["vote"] in ["VETO", "CAUTION", "CLEAR"]
    assert 0.0 <= result["conviction_score"] <= 100.0
    assert 0.0 <= result["severity_score"] <= 100.0
    assert isinstance(result["risk_factors"], list)
    assert "key_metrics" in result
    assert "vulnerabilities_detected" in result["key_metrics"]
    assert "tail_risk_status" in result["key_metrics"]
    assert "thesis" in result


def test_red_team_stress_scenario():
    agent = AdversarialRedTeamAgent()
    result = agent.evaluate("TSLA", spot_price=200.0)
    assert result["vote"] in ["VETO", "CAUTION", "CLEAR"]
    assert "academic_grounding" in result
