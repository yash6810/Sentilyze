from src.ai_copilot import AICopilotEngine


def test_copilot_portfolio_query():
    copilot = AICopilotEngine()
    res = copilot.answer_query(
        "Show me my portfolio balance and profit", context_ticker="AVGO"
    )
    assert res["query_category"] == "PORTFOLIO_STATUS"
    assert "Live Portfolio Status" in res["markdown_response"]
    assert "total_equity" in res["structured_data"]


def test_copilot_committee_query(mocker):
    mocker.patch(
        "src.ai_copilot.convene_trading_committee",
        return_value={
            "ticker": "AVGO",
            "final_resolution": "🚀 CONVICTION INSTITUTIONAL BUY",
            "consensus_conviction_pct": 82.5,
            "tp1_target": 390.0,
            "tp2_target": 415.0,
            "stop_loss_target": 335.0,
            "agent_testimonies": [
                {
                    "agent_name": "Technical Momentum Specialist",
                    "role": "Pillar 1",
                    "vote": "BUY",
                    "conviction_score": 85.0,
                    "thesis": "Healthy trend.",
                }
            ],
            "cro_signoff": {
                "buy_votes": 3,
                "approved_leverage": 1.5,
                "kelly_allocation_pct": 12.5,
                "cro_thesis": "Approved.",
            },
        },
    )
    copilot = AICopilotEngine()
    res = copilot.answer_query(
        "What does the committee debate say about AVGO?", context_ticker="AVGO"
    )
    assert res["query_category"] == "COMMITTEE_VERDICT"
    assert "CONVICTION INSTITUTIONAL BUY" in res["markdown_response"]


def test_copilot_stress_query():
    copilot = AICopilotEngine()
    res = copilot.answer_query(
        "Simulate a 10% drop crash in my portfolio", context_ticker="AVGO"
    )
    assert res["query_category"] == "STRESS_SIMULATION"
    assert "Stress-Test Simulation" in res["markdown_response"]
    assert res["structured_data"]["drop_pct"] == 10.0


def test_copilot_ticker_analysis_query():
    copilot = AICopilotEngine()
    res = copilot.answer_query(
        "Why should I buy AVGO and what is the stop loss?", context_ticker="AVGO"
    )
    assert res["query_category"] == "TICKER_ANALYSIS"
    assert "Deep Quantitative Diagnosis" in res["markdown_response"]
    assert "tp1" in res["structured_data"]
