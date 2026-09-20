from src.financial_qa_agent import answer_financial_query


def test_financial_qa_agent():
    ans1 = answer_financial_query("What is our portfolio VaR if Semis drop 5% today?")
    assert "Stress-Test" in ans1["answer_markdown"] or "VaR" in ans1["answer_markdown"]

    ans2 = answer_financial_query("What is NVDA options max pain?")
    assert "Max Pain" in ans2["answer_markdown"]

    ans3 = answer_financial_query("Which stock has highest Piotroski F-score?")
    assert "Piotroski" in ans3["answer_markdown"]
