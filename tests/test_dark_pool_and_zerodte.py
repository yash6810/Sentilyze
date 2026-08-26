from src.dark_pool_radar import (
    scan_dark_pool_blocks,
    scan_abnormal_options_vol_oi,
    compute_dark_pool_sentiment,
)
from src.zerodte_scalper import generate_0dte_scalp_signal


def test_dark_pool_radar():
    blocks = scan_dark_pool_blocks("NVDA")
    assert len(blocks) >= 2
    assert "notional_value" in blocks[0]

    opts = scan_abnormal_options_vol_oi("NVDA")
    assert len(opts) >= 2
    assert "vol_to_oi_ratio" in opts[0]

    sentiment = compute_dark_pool_sentiment("NVDA")
    assert 0.0 <= sentiment["dark_pool_activity_score"] <= 100.0
    assert "regime" in sentiment


def test_zerodte_scalper():
    # Bullish breakout
    sig_bull = generate_0dte_scalp_signal(
        index_ticker="SPY",
        current_index_price=565.0,
        opening_range_high=562.0,
        opening_range_low=559.0,
        vwap_price=561.0,
    )
    assert sig_bull["direction"] == "BULLISH CALL SCALP"
    assert sig_bull["option_type"] == "CALL"
    assert "take_profit_1 (+50%)" in sig_bull

    # Bearish breakdown
    sig_bear = generate_0dte_scalp_signal(
        index_ticker="QQQ",
        current_index_price=470.0,
        opening_range_high=480.0,
        opening_range_low=475.0,
        vwap_price=478.0,
    )
    assert sig_bear["direction"] == "BEARISH PUT SCALP"
    assert sig_bear["option_type"] == "PUT"
