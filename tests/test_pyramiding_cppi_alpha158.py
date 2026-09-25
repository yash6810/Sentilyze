"""
Unit Tests for:
1. CPPI Dynamic Cushion Scaling (src/cppi_insurance.py)
2. Turtle 0.5N Pyramiding Engine (src/compound_engine.py & src/paper_broker.py)
3. Microsoft Qlib Top 25 Orthogonal Alpha158 Factors (src/alpha158_factors.py)
4. Softmax Dirichlet RLFF Committee Calibration (src/agent_memory.py)

STRICT PORTFOLIO PRESERVATION: All broker and memory tests use tmp_path.
"""

import numpy as np
import pandas as pd


def test_cppi_cushion_multiplier_full_capacity():
    from src.cppi_insurance import get_cppi_cushion_multiplier

    # When current equity is at peak equity ($150k), cushion is 5% of equity (cushion = $7.5k).
    # multiplier = 2.85 -> risky exposure = 2.85 * 0.05 = 14.25% (or full 1.0 factor when cushion is wide)
    eval_res = get_cppi_cushion_multiplier(
        portfolio_value=150000.0,
        peak_equity=150000.0,
        floor_pct=0.90,  # 10% cushion = 2.85 * 0.10 = 0.285
        multiplier=2.85,
    )
    assert "allocation_factor" in eval_res
    assert eval_res["allocation_factor"] > 0.0
    assert eval_res["cushion"] > 0.0
    assert eval_res["floor_value"] == 135000.0


def test_cppi_cushion_multiplier_at_floor():
    from src.cppi_insurance import get_cppi_cushion_multiplier

    # When current equity is at or below the 95% floor, allocation factor must be 0.0
    eval_res = get_cppi_cushion_multiplier(
        portfolio_value=94000.0,
        peak_equity=100000.0,
        floor_pct=0.95,
        multiplier=2.85,
    )
    assert eval_res["allocation_factor"] == 0.0
    assert eval_res["cushion"] == 0.0
    assert eval_res["regime"] == "CAPITAL_PRESERVATION_FLOOR"


def test_turtle_pyramid_plan():
    from src.compound_engine import calculate_turtle_pyramid_plan

    plan = calculate_turtle_pyramid_plan(
        entry_price=100.0,
        atr=4.0,
        target_allocation_dollars=15000.0,
    )
    assert plan["total_target_dollars"] == 15000.0
    assert plan["total_units"] == 3
    assert plan["seed_unit_dollars"] == 5000.0

    # Unit 1 trigger should be 100 + 4 = 104
    assert plan["unit_triggers"]["unit_1_trigger"] == 104.0
    # Unit 2 trigger should be 100 + 8 = 108
    assert plan["unit_triggers"]["unit_2_trigger"] == 108.0


def test_turtle_pyramiding_step_evaluation():
    from src.compound_engine import evaluate_pyramiding_step

    pos_state = {
        "entry_price": 100.0,
        "pyramid_units_filled": 1,
        "sl_target": 94.0,
        "highest_price_seen": 100.0,
    }

    # At price 102 (below +1 ATR = 104), action should be HOLD
    step1 = evaluate_pyramiding_step(102.0, pos_state, atr=4.0)
    assert step1["action"] == "HOLD"
    assert step1["trigger_met"] is False

    # At price 104.5 (above +1 ATR = 104), action should be ADD_UNIT_1
    step2 = evaluate_pyramiding_step(104.5, pos_state, atr=4.0)
    assert step2["action"] == "ADD_UNIT_1"
    assert step2["trigger_met"] is True
    assert step2["target_unit"] == 2
    # SL should be raised to breakeven (> 100.0)
    assert step2["new_sl"] >= 100.0

    # Now simulate position at 2 units, price reaching 108.5 (above +2 ATR = 108)
    pos_state["pyramid_units_filled"] = 2
    pos_state["sl_target"] = 100.2
    pos_state["highest_price_seen"] = 108.5
    step3 = evaluate_pyramiding_step(108.5, pos_state, atr=4.0)
    assert step3["action"] == "ADD_UNIT_2"
    assert step3["trigger_met"] is True
    assert step3["target_unit"] == 3


def test_paper_broker_execute_pyramid_unit(tmp_path):
    from src.paper_broker import PaperBroker

    # Use tmp_path to strictly protect live results/paper_portfolio.json
    p_file = str(tmp_path / "test_portfolio.json")
    t_file = str(tmp_path / "test_trades.csv")

    broker = PaperBroker(portfolio_file=p_file, trades_file=t_file)
    broker.state["cash"] = 100000.0

    # Open Seed Unit: 50 shares @ $100
    buy_res = broker.execute_buy("NVDA", shares=50, price=100.0)
    assert buy_res["success"] is True
    assert broker.state["open_positions"]["NVDA"]["shares"] == 50
    assert broker.state["open_positions"]["NVDA"]["entry_price"] == 100.0

    # Execute Pyramid Unit 2: Add 50 shares @ $104, raise SL to $100.20
    pyr_res = broker.execute_pyramid_unit(
        ticker="NVDA",
        unit_num=2,
        shares=50,
        price=104.0,
        new_sl=100.20,
    )
    assert pyr_res["success"] is True
    assert pyr_res["total_shares"] == 100
    # Blended basis: (50*100 + 50*104) / 100 = $102.00
    assert pyr_res["blended_entry"] == 102.0
    assert broker.state["open_positions"]["NVDA"]["entry_price"] == 102.0
    assert broker.state["open_positions"]["NVDA"]["pyramid_units_filled"] == 2
    assert broker.state["open_positions"]["NVDA"]["sl_target"] == 100.20


def test_alpha158_top25_calculation():
    from src.alpha158_factors import compute_alpha158_top25, get_alpha158_feature_names

    # Create synthetic OHLCV data of length 40
    dates = pd.date_range("2026-01-01", periods=40, freq="B")
    np.random.seed(42)
    c = 100.0 + np.cumsum(np.random.randn(40) * 1.5)
    h = c + np.random.rand(40) * 2.0
    lo = c - np.random.rand(40) * 2.0
    o = lo + (h - lo) * np.random.rand(40)
    v = np.random.randint(1000000, 5000000, size=40)

    df = pd.DataFrame(
        {"Open": o, "High": h, "Low": lo, "Close": c, "Volume": v},
        index=dates,
    )

    factors = compute_alpha158_top25(df)
    expected_names = get_alpha158_feature_names()

    assert len(factors.columns) == 25
    assert len(factors) == 40
    for name in expected_names:
        assert name in factors.columns
    # Check that there are no NaNs or Infs
    assert not factors.isna().any().any()
    assert not np.isinf(factors.values).any()


def test_agent_memory_softmax_dirichlet_recalibration(tmp_path):
    from src.agent_memory import AgentMemoryStore

    mem_dir = str(tmp_path / "memory")
    store = AgentMemoryStore(memory_dir=mem_dir)

    # Seed 3 post-mortems
    store.record_postmortem(
        ticker="NVDA",
        direction="LONG",
        agent_votes={"Technical Momentum": "BUY", "Adversarial Red-Team": "CAUTION"},
        outcome="WIN",
        r_multiple=2.5,
        pnl_pct=6.5,
        pnl_dollars=1200.0,
    )
    store.record_postmortem(
        ticker="AAPL",
        direction="LONG",
        agent_votes={"Technical Momentum": "BUY", "Adversarial Red-Team": "VETO"},
        outcome="LOSS",
        r_multiple=-1.0,
        pnl_pct=-2.5,
        pnl_dollars=-450.0,
    )
    store.record_postmortem(
        ticker="PLTR",
        direction="LONG",
        agent_votes={"FinBERT Sentiment": "BUY", "Fundamental Valuation": "HOLD"},
        outcome="WIN",
        r_multiple=3.0,
        pnl_pct=8.0,
        pnl_dollars=1800.0,
    )

    calibrated = store.recalibrate_weights()
    assert isinstance(calibrated, dict)
    assert len(calibrated) == 5
    # Simplex property: weights must sum to 1.0 (+- 0.01 rounding)
    assert abs(sum(calibrated.values()) - 1.0) < 0.01
    for w in calibrated.values():
        assert w > 0.0
