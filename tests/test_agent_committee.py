from src.agent_committee import (
    TechnicalAlphaAgent,
    SentimentCatalystAgent,
    ForensicFundamentalAgent,
    ChiefRiskOfficerAgent,
    compute_fractional_kelly_sizing,
    convene_trading_committee,
    audit_full_universe_committee,
)


def test_fractional_kelly_sizing():
    # 1. Positive Edge Test (53.3% win rate, 1.75 payoff ratio, Quarter-Kelly)
    res_pos = compute_fractional_kelly_sizing(
        win_rate=0.533, payoff_ratio=1.75, kelly_fraction=0.25
    )
    assert res_pos["status"] == "POSITIVE_EXPECTANCY"
    assert res_pos["full_kelly_pct"] > 0.0
    assert 0.0 < res_pos["fractional_kelly_pct"] <= 15.0
    assert res_pos["edge"] > 0.0

    # 2. Negative Edge Test (30% win rate, 1.0 payoff ratio)
    res_neg = compute_fractional_kelly_sizing(
        win_rate=0.30, payoff_ratio=1.0, kelly_fraction=0.25
    )
    assert res_neg["status"] == "NEGATIVE_EXPECTANCY_NO_ALLOCATION"
    assert res_neg["fractional_kelly_pct"] == 0.0


def test_technical_alpha_agent():
    agent = TechnicalAlphaAgent()
    rep = agent.evaluate("NVDA", spot_price=220.0)
    assert rep["agent_name"] == "Technical Momentum Specialist"
    assert rep["vote"] in ["BUY", "NEUTRAL", "HOLD"]
    assert 0.0 <= rep["conviction_score"] <= 100.0
    assert "estimated_rsi" in rep["key_metrics"]


def test_sentiment_catalyst_agent():
    agent = SentimentCatalystAgent()
    rep = agent.evaluate("NVDA")
    assert rep["agent_name"] == "Sentiment & Alternative Data Specialist"
    assert rep["vote"] in ["BUY", "HOLD", "SELL"]
    assert 0.0 <= rep["conviction_score"] <= 100.0


def test_forensic_fundamental_agent():
    agent = ForensicFundamentalAgent()
    rep = agent.evaluate("NVDA", spot_price=220.0)
    assert rep["agent_name"] == "Forensic & Valuation Auditor"
    assert "piotroski_f_score" in rep["key_metrics"]
    assert "altman_z_score" in rep["key_metrics"]


def test_cro_agent_approval_and_veto():
    cro = ChiefRiskOfficerAgent()

    # 1. Normal Approval
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
    assert signoff["action_code"] in ["EXECUTE_BUY", "SCALE_IN"]
    assert signoff["approved_leverage"] >= 1.0
    assert signoff["vix_veto_triggered"] is False
    assert signoff["kelly_allocation_pct"] > 0.0

    # 2. VIX Panic Veto
    signoff_panic = cro.evaluate_and_sign_off(
        "NVDA", 220.0, mock_reports, vix_level=32.0, vix_change_pct=15.0
    )
    assert signoff_panic["action_code"] == "VETO"
    assert signoff_panic["approved_leverage"] == 0.0
    assert signoff_panic["vix_veto_triggered"] is True


def test_convene_trading_committee(tmp_path, mocker):
    mocker.patch(
        "src.agent_committee.COMMITTEE_FILE", str(tmp_path / "test_resolutions.json")
    )
    res = convene_trading_committee("NVDA", vix_level=16.0)
    assert res["ticker"] == "NVDA"
    assert "final_resolution" in res
    assert len(res["agent_testimonies"]) == 4
    assert "cro_signoff" in res


def test_audit_full_universe_committee(tmp_path, mocker):
    mocker.patch(
        "src.agent_committee.COMMITTEE_FILE",
        str(tmp_path / "test_resolutions_universe.json"),
    )
    summary = audit_full_universe_committee(universe_tickers=["NVDA", "AAPL"])
    assert summary["total_audited"] == 2
    assert "NVDA" in summary["resolutions"]
    assert "AAPL" in summary["resolutions"]
