import pytest
from src.autonomous_trader import FormalLogicZ3RuleGate


def test_formal_logic_rule_gate_satisfaction():
    gate = FormalLogicZ3RuleGate(min_cash_buffer=1000.0, max_position_pct=0.20)

    portfolio_state = {
        "cash": 50000.0,
        "total_equity": 100000.0,
    }

    deliberation = {
        "tp1_target": 110.0,
        "tp2_target": 120.0,
        "stop_loss_target": 95.0,
    }

    # Valid trade: 50 shares @ $100 = $5,000 cost (10% of cash, 5% of equity)
    res = gate.verify_trade(
        ticker="NVDA",
        spot_price=100.0,
        shares=50.0,
        deliberation=deliberation,
        portfolio_state=portfolio_state,
    )

    assert res["is_valid"] is True
    assert res["status"] == "SATISFIED"
    assert len(res["violations"]) == 0
    assert res["invariants_checked"]["solvency_cash_buffer"] is True
    assert res["invariants_checked"]["bracket_monotonicity"] is True


def test_formal_logic_rule_gate_violations():
    gate = FormalLogicZ3RuleGate(min_cash_buffer=5000.0, max_position_pct=0.15)

    portfolio_state = {
        "cash": 10000.0,
        "total_equity": 100000.0,
    }

    # Violates bracket monotonicity: stop_loss > entry_price
    invalid_deliberation = {
        "tp1_target": 105.0,
        "tp2_target": 110.0,
        "stop_loss_target": 102.0,  # Invalid!
    }

    # Cost = 100 * 100 = $10,000 > cash ($10,000) - buffer ($5,000) -> Solvency violation!
    res = gate.verify_trade(
        ticker="AAPL",
        spot_price=100.0,
        shares=100.0,
        deliberation=invalid_deliberation,
        portfolio_state=portfolio_state,
    )

    assert res["is_valid"] is False
    assert res["status"] == "VIOLATED"
    assert len(res["violations"]) >= 2
