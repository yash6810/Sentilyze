from src.sec_filing_diff import analyze_sec_filing_diff
from src.earnings_sentiment import analyze_earnings_call_transcript
from src.social_sentiment import (
    calculate_social_buzz_metrics,
    fetch_social_sentiment_tracker,
)
from src.insider_tracker import (
    track_corporate_insider_filings,
    track_congressional_stock_disclosures,
    compute_smart_money_insider_score,
)
from src.patent_contract_radar import (
    track_federal_contract_awards,
    track_uspto_patent_momentum,
    compute_government_and_patent_index,
)


def test_sec_filing_diff_analysis():
    res = analyze_sec_filing_diff("NVDA")
    assert res["ticker"] == "NVDA"
    assert "similarity_score" in res
    assert 0.0 <= res["similarity_score"] <= 1.0
    assert "status" in res
    assert "material_risks_added" in res


def test_earnings_call_sentiment():
    res = analyze_earnings_call_transcript("NVDA", quarter="Q2 2026")
    assert res["ticker"] == "NVDA"
    assert 0.0 <= res["executive_optimism_score"] <= 100.0
    assert 0.0 <= res["analyst_skepticism_score"] <= 100.0
    assert "verdict" in res


def test_social_sentiment_velocity():
    metrics = calculate_social_buzz_metrics(
        "NVDA",
        mention_volume_today=2500,
        avg_7d_mentions=1000,
        bullish_posts=1800,
        bearish_posts=700,
    )
    assert metrics["mention_velocity_ratio"] == 2.5
    assert metrics["bullish_sentiment_pct"] > 70.0
    assert "regime" in metrics

    tracker = fetch_social_sentiment_tracker("TSLA")
    assert tracker["ticker"] == "TSLA"
    assert tracker["mention_volume_24h"] > 0
    assert "reddit_stream" in tracker
    assert "stocktwits_stream" in tracker


def test_insider_and_congressional_tracker():
    insiders = track_corporate_insider_filings("NVDA")
    assert len(insiders) >= 2
    assert "insider_name" in insiders[0]

    congress = track_congressional_stock_disclosures("NVDA")
    assert len(congress) >= 1
    assert "politician" in congress[0]

    smart_money = compute_smart_money_insider_score("NVDA")
    assert 0.0 <= smart_money["smart_money_score"] <= 100.0
    assert "sentiment_verdict" in smart_money


def test_patent_and_contract_radar():
    contracts = track_federal_contract_awards("PLTR")
    assert len(contracts) >= 2
    assert "award_value" in contracts[0]

    patents = track_uspto_patent_momentum("NVDA")
    assert patents["granted_patents_90d"] > 0
    assert "ai_patents_pct" in patents

    gov_index = compute_government_and_patent_index("PLTR")
    assert 0.0 <= gov_index["composite_innovation_score"] <= 100.0
    assert "badge" in gov_index
