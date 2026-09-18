import json
from unittest.mock import patch, MagicMock

from src.sec_crawler import (
    SECCatalystCrawler,
    get_recent_sec_catalysts,
)


def test_sec_cik_map_fallback():
    crawler = SECCatalystCrawler()
    with patch("urllib.request.urlopen", side_effect=Exception("Network offline")):
        ciks = crawler.load_sec_cik_map()
        assert isinstance(ciks, dict)
        assert ciks.get("NVDA") == "0001045810"
        assert ciks.get("AAPL") == "0000320193"
        assert ciks.get("MSFT") == "0000789019"


def test_score_catalyst_sentiment():
    crawler = SECCatalystCrawler()

    # Positive catalyst: Record revenue beat
    pos_score = crawler._score_catalyst_sentiment(
        text="Company reports record revenue and beat consensus expectations",
        tags=["EARNINGS_ANNOUNCEMENT (Quarterly / Annual Results)"],
    )
    assert pos_score > 0.20

    # Risk catalyst: Investigation / Delisting
    risk_score = crawler._score_catalyst_sentiment(
        text="Company received formal subpoena and investigation regarding material weakness",
        tags=["OTHER_MATERIAL_EVENTS (FDA Decisions, Buybacks, Litigation)"],
    )
    assert risk_score < -0.20

    # Neutral / generic announcement
    neutral_score = crawler._score_catalyst_sentiment(
        text="General corporate update and routine proxy mailing",
        tags=["GENERAL_MATERIAL_EVENT"],
    )
    assert -0.20 <= neutral_score <= 0.20


def test_fetch_recent_8k_filings_mocked():
    crawler = SECCatalystCrawler()
    crawler._ticker_to_cik = {"NVDA": "0001045810"}
    crawler._cik_loaded = True

    mock_submissions = {
        "filings": {
            "recent": {
                "form": ["8-K", "10-Q", "8-K"],
                "filingDate": ["2026-09-10", "2026-09-08", "2026-09-01"],
                "primaryDocDescription": [
                    "Item 1.01 Entry into Definitive Agreement with Global Partner",
                    "Quarterly Report",
                    "Item 5.02 Departure of Director or Certain Officers",
                ],
                "items": [["1.01"], [], ["5.02"]],
            }
        }
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_submissions).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        catalysts = crawler.fetch_recent_8k_filings("NVDA", days_lookback=30)
        assert len(catalysts) == 2  # 10-Q should be skipped
        first = catalysts[0]
        assert first["ticker"] == "NVDA"
        assert first["form_type"] == "8-K"
        assert any("MATERIAL_AGREEMENT" in t for t in first["catalyst_types"])
        assert first["sentiment_score"] > 0.0


def test_get_recent_sec_catalysts_helper():
    with patch.object(
        SECCatalystCrawler,
        "fetch_recent_8k_filings",
        return_value=[{"ticker": "NVDA", "sentiment_score": 0.5}],
    ):
        res = get_recent_sec_catalysts("NVDA", days=7)
        assert len(res) == 1
        assert res[0]["ticker"] == "NVDA"
