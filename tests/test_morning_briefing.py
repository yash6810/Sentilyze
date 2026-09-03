import pytest
from src.morning_briefing import (
    generate_morning_briefing_text,
    synthesize_briefing_audio,
    scan_top_alpha_stocks,
    load_universe_candidates,
    get_portfolio_intelligence,
)


def test_load_universe_candidates():
    candidates = load_universe_candidates(max_count=10)
    assert isinstance(candidates, list)
    assert len(candidates) > 0
    assert all(isinstance(t, str) and len(t) > 0 for t in candidates)


def test_scan_top_alpha_stocks():
    top_picks = scan_top_alpha_stocks(
        candidate_tickers=["NVDA", "AAPL", "MSFT"], top_k=2
    )
    assert isinstance(top_picks, list)
    assert len(top_picks) > 0
    pick = top_picks[0]
    assert "ticker" in pick
    assert "last_price" in pick
    assert "conviction_pct" in pick
    assert "tp1_target" in pick
    assert "sl_target" in pick


def test_get_portfolio_intelligence():
    intel = get_portfolio_intelligence()
    assert "total_equity" in intel
    assert "cash_reserves" in intel
    assert intel["total_equity"] >= 100000.0
    assert intel["win_rate"] > 0


def test_generate_morning_briefing_market_master_mode():
    memo = generate_morning_briefing_text(mode="MARKET_MASTER", ticker="NVDA")
    assert "headline" in memo
    assert "executive_summary" in memo
    assert "audio_script" in memo
    assert len(memo["audio_script"]) > 100
    assert "macro_posture" in memo
    assert "portfolio_status" in memo
    assert "top_stocks_in_play" in memo


def test_generate_morning_briefing_modes():
    modes = ["TOP_STOCKS", "PORTFOLIO_RADAR", "SINGLE_TICKER"]
    for m in modes:
        res = generate_morning_briefing_text(mode=m, ticker="NVDA")
        assert "audio_script" in res
        assert len(res["audio_script"]) > 50


def test_synthesize_briefing_audio_execution():
    test_script = (
        "Good morning. This is your Sentilyze quantitative morning test briefing."
    )
    path = synthesize_briefing_audio(
        test_script,
        output_path="results/test_briefing.mp3",
        voice_key="UK_LONDON",
        slow=False,
    )
    assert path is not None
