"""
Unit Tests for Fast-Track Catalyst Lane & Live Breaking News Ingestion.
Validates:
1. SentimentCatalystAgent accepts override_headlines and returns BUY with high conviction.
2. convene_trading_committee exposes verdict and conviction aliases.
3. Event-driven catalyst trades bypass conservative DCF valuation downvotes.
4. benzinga_news_trader unblocks trade routing for EXECUTE_BUY and SCALE_IN.

STRICT PORTFOLIO PRESERVATION: Uses tmp_path for all broker states.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.agent_committee import (
    SentimentCatalystAgent,
    ChiefRiskOfficerAgent,
    convene_trading_committee,
)
from src.benzinga_news_trader import scan_and_trade_benzinga_catalysts
from src.paper_broker import PaperBroker


def test_sentiment_catalyst_override_headlines():
    agent = SentimentCatalystAgent()
    breaking_news = [
        "Akamai to Acquire Leading Edge Cloud Provider in Transformational $11.6 Billion Deal",
        "Analysts Upgrade Akamai with Price Target Boost to $145",
    ]
    report = agent.evaluate(ticker="AKAM", override_headlines=breaking_news)
    assert report["vote"] == "BUY"
    assert report["conviction_score"] >= 65.0
    assert report["key_metrics"]["is_material"] is True
    assert report["key_metrics"]["headlines_analyzed"] == 2
    assert "HIGH-IMPACT CATALYST" in report["thesis"] or "M_AND_A" in report["thesis"]


def test_catalyst_lane_dcf_bypass():
    cro = ChiefRiskOfficerAgent()
    spot_price = 100.0

    # Simulate specialist reports where DCF votes AVOID/SELL due to high valuation
    # but Sentiment specialist has a material catalyst with strong BUY
    agent_reports = [
        {
            "agent_name": "Technical Momentum Specialist",
            "vote": "BUY",
            "conviction_score": 68.0,
            "key_metrics": {"rsi": 62.0, "trend_status": "BULLISH_ABOVE_SMA200"},
            "thesis": "Bullish momentum breakout above 20 EMA.",
        },
        {
            "agent_name": "Sentiment & Alternative Data Specialist",
            "vote": "BUY",
            "conviction_score": 75.0,
            "key_metrics": {
                "finbert_polarity": 0.35,
                "is_material": True,
                "event_type": "M&A",
            },
            "thesis": "Major M&A buyout catalyst.",
        },
        {
            "agent_name": "Fundamental Valuation Specialist",
            "vote": "AVOID",
            "conviction_score": 35.0,
            "key_metrics": {
                "dcf_margin_of_safety_pct": -0.25,
                "piotroski_f_score": 6,
            },
            "thesis": "DCF models indicate 25% overvaluation.",
        },
        {
            "agent_name": "Price Scout & Microstructure Specialist",
            "vote": "BUY",
            "conviction_score": 65.0,
            "key_metrics": {"poc_proximity_pct": 0.5},
            "thesis": "Price holding above Point of Control.",
        },
        {
            "agent_name": "Adversarial Red-Team Specialist",
            "vote": "CAUTION",
            "conviction_score": 45.0,
            "key_metrics": {"vulnerability_count": 1},
            "thesis": "Watch post-earnings volatility.",
        },
    ]

    signoff = cro.evaluate_and_sign_off(
        ticker="TEST",
        spot_price=spot_price,
        agent_reports=agent_reports,
        vix_level=16.0,
        vix_change_pct=-0.5,
    )
    # The catalyst lane ensures that with 3 buy votes (Technical, Sentiment, Scout)
    # and material catalyst, action_code is EXECUTE_BUY
    assert signoff["action_code"] in ["EXECUTE_BUY", "SCALE_IN"]
    assert signoff["approved_leverage"] >= 1.0


def test_convene_trading_committee_verdict_alias():
    breaking_news = ["Company announces blowout record revenue and raises guidance."]
    with patch(
        "src.agent_committee.ChiefRiskOfficerAgent.evaluate_and_sign_off"
    ) as mock_cro:
        mock_cro.return_value = {
            "action_code": "EXECUTE_BUY",
            "final_resolution": "🚀 HIGH CONVICTION COMMITTEE BUY",
            "consensus_conviction_pct": 78.5,
            "approved_leverage": 1.25,
            "kelly_allocation_pct": 8.0,
            "dynamic_take_profit_tp1": 115.0,
            "dynamic_take_profit_tp2": 125.0,
            "dynamic_stop_loss": 95.0,
            "agent_reports": [],
        }
        res = convene_trading_committee(
            "NVDA", spot_price=105.0, catalyst_headlines=breaking_news
        )
        assert res["action_code"] == "EXECUTE_BUY"
        assert res["verdict"] == "EXECUTE_BUY"
        assert res["conviction"] == 78.5


from src.benzinga_news_trader import scan_and_trade_benzinga_catalysts


def test_benzinga_news_trader_unblocked_routing(tmp_path):
    p_file = str(tmp_path / "test_port.json")
    t_file = str(tmp_path / "test_trades.csv")

    broker = PaperBroker(portfolio_file=p_file, trades_file=t_file)
    broker.state["cash"] = 50000.0

    mock_news = [
        {
            "ticker": "TEST",
            "extracted_tickers": ["TEST"],
            "title": "TEST Reports $10B Transformative Cloud Deal",
            "link": "https://example.com",
            "published": "2026-10-02",
            "catalyst_sentiment": "BULLISH",
            "sentiment_score": 0.85,
        }
    ]

    mock_committee = {
        "action_code": "EXECUTE_BUY",
        "verdict": "EXECUTE_BUY",
        "consensus_conviction_pct": 78.0,
        "conviction": 78.0,
        "dynamic_take_profit_tp1": 115.0,
        "dynamic_take_profit_tp2": 125.0,
        "dynamic_stop_loss": 95.0,
    }

    with (
        patch(
            "src.benzinga_news_trader.fetch_top_benzinga_news_wire",
            return_value=mock_news,
        ),
        patch(
            "src.benzinga_news_trader.fetch_live_quote",
            return_value={"price": 100.0, "change_pct": 2.5},
        ),
        patch(
            "src.benzinga_news_trader.convene_trading_committee",
            return_value=mock_committee,
        ),
    ):
        report = scan_and_trade_benzinga_catalysts(
            execute_paper=True, broker=broker, max_stocks_to_evaluate=1
        )
        assert len(report.get("trade_actions_executed", [])) == 1
        action = report["trade_actions_executed"][0]
        assert action["ticker"] == "TEST"
        assert action["action"] == "BUY"
        assert "TEST" in broker.state["open_positions"]
