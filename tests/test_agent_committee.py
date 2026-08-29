from src.agent_committee import (
    TechnicalAlphaAgent,
    SentimentCatalystAgent,
    ForensicFundamentalAgent,
    InstitutionalFlowAgent,
    MacroSectorRegimeAgent,
    CatalystMoatAgent,
    ChiefRiskOfficerAgent,
    convene_trading_committee,
    audit_full_universe_committee,
)


def test_new_specialist_agents():
    flow = InstitutionalFlowAgent()
    rep_f = flow.evaluate("NVDA", spot_price=220.0)
    assert rep_f["agent_name"] == "Institutional Flow & Dark Pool Tracker"
    assert "insider_score" in rep_f["key_metrics"]

    macro = MacroSectorRegimeAgent()
    rep_m = macro.evaluate("NVDA", spot_price=220.0, vix_level=16.5)
    assert rep_m["agent_name"] == "Macro Regime & Sector Strategist"
    assert rep_m["vote"] in ["BUY", "HOLD"]

    moat = CatalystMoatAgent()
    rep_moat = moat.evaluate("NVDA", spot_price=220.0)
    assert rep_moat["agent_name"] == "Catalyst & Competitive Moat Specialist"
    assert "patent_index" in rep_moat["key_metrics"]


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
    assert "beneish_m_score" in rep["key_metrics"]


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
            "key_metrics": {"beneish_m_score": -2.4},
        },
    ]
    signoff = cro.evaluate_and_sign_off("NVDA", 220.0, mock_reports, vix_level=15.0)
    assert signoff["action_code"] in ["EXECUTE_BUY", "SCALE_IN"]
    assert signoff["approved_leverage"] >= 1.0
    assert signoff["vix_veto_triggered"] is False

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
    assert len(res["agent_testimonies"]) == 6
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
