from src.forensic_accounting import (
    calculate_beneish_m_score,
    analyze_debt_maturity_wall,
)


def test_beneish_m_score():
    # Pristine accounting inputs
    res_pristine = calculate_beneish_m_score(
        "NVDA",
        dsri=1.01,
        gmi=0.98,
        aqi=0.92,
        sgi=1.10,
        depi=1.00,
        sgai=0.95,
        lvgi=0.90,
        tata=0.01,
    )
    assert res_pristine["ticker"] == "NVDA"
    assert "beneish_m_score" in res_pristine
    assert res_pristine["beneish_m_score"] < -1.78
    assert "PRISTINE" in res_pristine["verdict"]

    # Red flag manipulation inputs
    res_red = calculate_beneish_m_score(
        "ENRON",
        dsri=2.5,
        gmi=1.8,
        aqi=2.0,
        sgi=2.5,
        depi=1.5,
        sgai=1.2,
        lvgi=1.8,
        tata=0.35,
    )
    assert res_red["beneish_m_score"] > -1.78
    assert "HIGH FORENSIC RED FLAGS" in res_red["verdict"]


def test_debt_maturity_wall():
    wall = analyze_debt_maturity_wall("NVDA")
    assert wall["ticker"] == "NVDA"
    assert "total_debt_billions" in wall
    assert len(wall["maturities"]) >= 3
    assert wall["interest_coverage_ratio"] > 10.0
