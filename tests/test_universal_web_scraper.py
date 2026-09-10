from unittest.mock import patch, MagicMock
import pandas as pd
from src.universal_web_scraper import (
    detect_stock_tickers,
    extract_financial_tables_from_html,
    _score_text_with_finbert,
    _extract_summary_sentences,
    scrape_url,
    batch_scrape_urls,
    scrape_ticker_news_from_web,
)


def test_detect_stock_tickers():
    text = "We are bullish on $NVDA and AAPL after the recent earnings surge. However, TSLA and MSFT face headwinds."
    tickers = detect_stock_tickers(text)
    assert "NVDA" in tickers
    assert "AAPL" in tickers
    assert "TSLA" in tickers
    assert "MSFT" in tickers


def test_extract_financial_tables_from_html():
    html = """
    <html>
        <body>
            <table id="financials">
                <tr><th>Metric</th><th>Q3 2026</th><th>Q4 2026</th></tr>
                <tr><td>Revenue</td><td>$35.1B</td><td>$38.4B</td></tr>
                <tr><td>Gross Margin</td><td>75.2%</td><td>76.0%</td></tr>
            </table>
        </body>
    </html>
    """
    tables = extract_financial_tables_from_html(html)
    assert len(tables) == 1
    df = tables[0]
    assert isinstance(df, pd.DataFrame)
    assert "Metric" in df.columns
    assert len(df) == 2


def test_extract_summary_sentences():
    text = "Nvidia reports record earnings beating all analyst expectations.\n\nData center revenue expanded by 112% year-over-year.\n\nManagement announced an increase in share buyback authorizations."
    summary = _extract_summary_sentences(text, max_sentences=2)
    assert "Nvidia reports record earnings" in summary


def test_score_text_with_finbert():
    bull_text = "Nvidia reports record quarterly earnings surge beating all Wall Street estimates with massive profit growth."
    score_res = _score_text_with_finbert(bull_text)
    assert "sentiment_label" in score_res
    assert score_res["sentiment_label"] in ["positive", "neutral", "negative"]
    assert "sentiment_score" in score_res


def test_scrape_url_invalid():
    res = scrape_url("ftp://invalid-url.com")
    assert res["success"] is False
    assert "Invalid URL" in res["error"]


@patch("requests.Session.get")
def test_scrape_url_mock(mock_get):
    sample_html = """
    <html>
        <head>
            <title>Nvidia Unveils Next-Gen Blackwell Ultra GPUs at GTC</title>
            <meta name="description" content="Nvidia announced record AI chip shipments and forward guidance.">
        </head>
        <body>
            <h1>Nvidia Unveils Next-Gen Blackwell Ultra GPUs at GTC</h1>
            <p>Nvidia CEO Jensen Huang announced the new architecture during the keynote speech. The new platform delivers 4x inference speedup for generative AI reasoning models.</p>
            <p>Wall Street analysts from Goldman Sachs reiterated their buy rating on $NVDA with a $200 price target.</p>
            <table>
                <tr><th>Architecture</th><th>Compute FP8</th></tr>
                <tr><td>Blackwell Ultra</td><td>20 PetaFLOPS</td></tr>
            </table>
        </body>
    </html>
    """
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.text = sample_html
    mock_get.return_value = mock_res

    res = scrape_url("https://example.com/nvidia-gtc-news")
    assert res["success"] is True
    assert "Nvidia" in res["title"]
    assert "NVDA" in res["detected_tickers"]
    assert res["word_count"] > 10
    assert len(res["tables"]) >= 1


@patch("src.universal_web_scraper._get_browser_session")
def test_scrape_ticker_news_from_web(mock_session_fn):
    mock_session = MagicMock()
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.text = """
    <html>
        <body>
            <table id="news-table">
                <tr>
                    <td>Sep-08-26 10:00AM</td>
                    <td>
                        <a href="https://reuters.com/nvda-deal">Nvidia Signs Multi-Billion Enterprise AI Infrastructure Partnership</a>
                        <span>(Reuters)</span>
                    </td>
                </tr>
            </table>
        </body>
    </html>
    """
    mock_session.get.return_value = mock_res
    mock_session_fn.return_value = mock_session

    df = scrape_ticker_news_from_web("NVDA", limit=5)
    assert not df.empty
    assert "Title" in df.columns
    assert "publishedAt" in df.columns
    assert "Nvidia Signs Multi-Billion" in df["Title"].iloc[0]
    assert df["source"].iloc[0]["name"] == "Reuters"
