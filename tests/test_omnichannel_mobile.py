from src.whatsapp_alerts import (
    format_whatsapp_trade_alert,
    send_whatsapp_notification,
)
from src.smartwatch_api import generate_smartwatch_glance_payload
from src.financial_qa_agent import answer_financial_query


def test_whatsapp_alerts():
    msg = format_whatsapp_trade_alert(
        ticker="NVDA",
        action="BUY",
        price=128.50,
        shares=100,
        stage="TP1 Scale-Out (50%)",
        pnl_dollars=450.0,
    )
    assert "NVDA" in msg
    assert "TP1" in msg

    dispatch = send_whatsapp_notification(msg)
    assert "SUCCESS" in dispatch["status"] or "DELIVERED" in dispatch["status"]


def test_smartwatch_api():
    payload = generate_smartwatch_glance_payload(
        total_equity=108500.0,
        daily_pnl_pct=3.10,
        top_active_ticker="NVDA",
        top_active_pnl_pct=5.40,
    )
    assert "complications" in payload
    assert "circular_gauge" in payload["complications"]
    assert "modular_large" in payload["complications"]
    assert payload["complications"]["circular_gauge"]["value_text"] == "+3.1%"


def test_financial_qa_agent():
    ans1 = answer_financial_query("What is our portfolio VaR if Semis drop 5% today?")
    assert "Stress-Test" in ans1["answer_markdown"] or "VaR" in ans1["answer_markdown"]

    ans2 = answer_financial_query("What is NVDA options max pain?")
    assert "Max Pain" in ans2["answer_markdown"]

    ans3 = answer_financial_query("Which stock has highest Piotroski F-score?")
    assert "Piotroski" in ans3["answer_markdown"]
