from src.quant_engine import run_unified_institutional_pipeline


def test_run_unified_institutional_pipeline():
    res = run_unified_institutional_pipeline("NVDA", account_equity=100000.0)

    assert res["ticker"] == "NVDA"
    assert res["spot_price"] > 0.0
    assert 0.0 <= res["master_composite_score"] <= 100.0
    assert "institutional_verdict" in res
    assert "verdict_color" in res
    assert res["pipeline_state"] == "SYNCHRONIZED_MACHINE_FLOW"


def test_all_8_pillars_present_in_output():
    res = run_unified_institutional_pipeline("AAPL", account_equity=150000.0)

    pillars = res["pillars"]
    assert "p1_ai_alpha" in pillars
    assert "p2_alternative_data" in pillars
    assert "p3_options_microstructure" in pillars
    assert "p4_supply_chain" in pillars
    assert "p5_risk_management" in pillars
    assert "p6_smart_execution" in pillars
    assert "p7_omnichannel_mobile" in pillars
    assert "p8_forensics_valuation" in pillars

    # Verify specific metrics inside pillars
    assert "tft_1d_target" in pillars["p1_ai_alpha"]
    assert "sec_status" in pillars["p2_alternative_data"]
    assert "max_pain_strike" in pillars["p3_options_microstructure"]
    assert "half_kelly_allocation_dollars" in pillars["p5_risk_management"]
    assert "vwap_child_orders" in pillars["p6_smart_execution"]
    assert "piotroski_f_score" in pillars["p8_forensics_valuation"]
