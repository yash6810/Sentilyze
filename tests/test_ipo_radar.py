import pytest
from unittest.mock import patch, MagicMock
from src.ipo_radar import (
    fetch_pre_ipo_radar_summary,
    fetch_sec_edgar_ipo_filings,
    auto_register_ipo_ticker,
    PRE_IPO_UNIVERSE,
)


def test_pre_ipo_universe_structure():
    assert len(PRE_IPO_UNIVERSE) >= 4
    names = [p["name"] for p in PRE_IPO_UNIVERSE]
    assert "OpenAI" in names
    assert "Anthropic" in names
    assert "SpaceX" in names
    assert "Stripe" in names


def test_fetch_pre_ipo_radar_summary():
    summary = fetch_pre_ipo_radar_summary()
    assert isinstance(summary, dict)
    assert "pre_ipo_targets" in summary
    assert "recent_s1_filings" in summary
    assert summary["total_targets_tracked"] >= 4


@patch("src.ipo_radar.requests.get")
def test_fetch_sec_edgar_ipo_filings(mock_get):
    mock_xml = b"""<?xml version="1.0" encoding="utf-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
        <entry>
            <title>S-1 - OpenAI Inc. (0001999999)</title>
            <updated>2026-08-27T12:00:00-04:00</updated>
            <link href="https://www.sec.gov/Archives/edgar/data/1999999/000199999926000001/s1.htm"/>
            <summary>Registration statement under the Securities Act of 1933</summary>
        </entry>
    </feed>
    """
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.content = mock_xml
    mock_get.return_value = mock_res

    filings = fetch_sec_edgar_ipo_filings()
    assert len(filings) >= 1
    assert "OpenAI" in filings[0]["title"]
    assert filings[0]["filing_type"] == "SEC Form S-1 (IPO Registration)"


def test_auto_register_ipo_ticker(tmp_path, monkeypatch):
    test_stocks_file = tmp_path / "test_stocks.txt"
    test_stocks_file.write_text("NVDA\nAAPL\n")
    monkeypatch.setattr("src.ipo_radar.STOCKS_FILE", str(test_stocks_file))

    # Register new ticker
    res = auto_register_ipo_ticker("OPAI", "OpenAI Inc.")
    assert res is True
    content = test_stocks_file.read_text()
    assert "OPAI" in content
    assert "OpenAI" in content
