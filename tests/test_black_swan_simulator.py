from src.black_swan_simulator import (
    simulate_portfolio_crises,
    calculate_kelly_sizing,
    estimate_market_impact_slippage,
    HISTORICAL_CRISES,
)


def test_historical_crises_definition():
    assert len(HISTORICAL_CRISES) >= 4
    for c in HISTORICAL_CRISES:
        assert "name" in c
        assert "market_drawdown_pct" in c
        assert c["market_drawdown_pct"] < 0


def test_simulate_portfolio_crises():
    positions = {
        "NVDA": 25000.0,
        "AAPL": 20000.0,
        "MSFT": 15000.0,
        "JPM": 10000.0,
    }
    results = simulate_portfolio_crises(positions, total_equity=100000.0)

    assert len(results) == len(HISTORICAL_CRISES)
    for r in results:
        assert "crisis_name" in r
        assert "portfolio_drawdown_pct" in r
        assert "projected_dollar_loss" in r
        assert "simulated_equity_after" in r
        assert r["simulated_equity_after"] <= 100000.0
        assert r["projected_dollar_loss"] > 0


def test_calculate_kelly_sizing():
    res = calculate_kelly_sizing(win_rate=0.55, win_loss_ratio=1.5)

    assert "full_kelly_pct" in res
    assert "half_kelly_pct" in res
    assert "recommended_leverage" in res
    assert res["half_kelly_pct"] == round(res["full_kelly_pct"] * 0.5, 1)
    assert 1.0 <= res["recommended_leverage"] <= 2.0


def test_estimate_market_impact_slippage():
    res = estimate_market_impact_slippage(
        order_size_dollars=25000.0,
        daily_volume_dollars=1_000_000_000.0,
        daily_volatility_pct=0.02,
    )

    assert "estimated_slippage_bps" in res
    assert "estimated_slippage_dollars" in res
    assert "liquidity_status" in res
    assert res["estimated_slippage_bps"] >= 0.0
