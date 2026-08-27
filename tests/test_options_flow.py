import pandas as pd
from src.options_flow import (
    fetch_option_chain,
    calculate_max_pain,
    calculate_put_call_ratios,
    estimate_gamma_exposure,
    recommend_option_spreads,
)


def _generate_test_chain():
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    calls = pd.DataFrame(
        {
            "strike": strikes,
            "lastPrice": [12.0, 7.5, 4.0, 1.8, 0.6],
            "openInterest": [1000, 2500, 5000, 3000, 1200],
            "volume": [200, 600, 1500, 800, 300],
            "impliedVolatility": [0.40, 0.38, 0.35, 0.36, 0.39],
        }
    )
    puts = pd.DataFrame(
        {
            "strike": strikes,
            "lastPrice": [0.8, 1.9, 4.2, 7.8, 12.5],
            "openInterest": [1500, 3500, 4500, 2000, 800],
            "volume": [300, 900, 1200, 500, 150],
            "impliedVolatility": [0.42, 0.39, 0.35, 0.37, 0.40],
        }
    )
    return calls, puts


def test_fetch_option_chain():
    res = fetch_option_chain("NVDA")
    assert "ticker" in res
    assert "spot_price" in res
    assert "calls_df" in res
    assert "puts_df" in res
    assert not res["calls_df"].empty
    assert not res["puts_df"].empty


def test_calculate_max_pain():
    calls, puts = _generate_test_chain()
    max_pain, loss_df = calculate_max_pain(calls, puts)

    assert isinstance(max_pain, float)
    assert max_pain in [90.0, 95.0, 100.0, 105.0, 110.0]
    assert not loss_df.empty
    assert "total_loss" in loss_df.columns


def test_calculate_put_call_ratios():
    calls, puts = _generate_test_chain()
    pcr = calculate_put_call_ratios(calls, puts)

    assert "pcr_open_interest" in pcr
    assert "pcr_volume" in pcr
    assert pcr["pcr_open_interest"] > 0
    assert "sentiment_verdict" in pcr


def test_estimate_gamma_exposure():
    calls, puts = _generate_test_chain()
    gex = estimate_gamma_exposure(calls, puts, spot_price=100.0)

    assert "net_gex" in gex
    assert "total_call_gex" in gex
    assert "total_put_gex" in gex
    assert "regime_verdict" in gex
    assert not gex["gex_by_strike"].empty


def test_recommend_option_spreads():
    calls, puts = _generate_test_chain()
    spreads_buy = recommend_option_spreads("NVDA", "BUY", 100.0, 100.0, calls, puts)
    assert len(spreads_buy) >= 2
    assert "Bull Call" in spreads_buy[0]["name"]
    assert "max_profit" in spreads_buy[0]
    assert "max_loss" in spreads_buy[0]

    spreads_hold = recommend_option_spreads("NVDA", "HOLD", 100.0, 100.0, calls, puts)
    assert len(spreads_hold) >= 2
    assert "Bear Put" in spreads_hold[0]["name"]
