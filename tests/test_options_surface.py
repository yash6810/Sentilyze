from src.options_surface import (
    generate_volatility_surface_mesh,
    calculate_multileg_payoff,
)


def test_generate_volatility_surface_mesh():
    mesh = generate_volatility_surface_mesh("AVGO", spot_price=350.0)
    assert mesh["ticker"] == "AVGO"
    assert len(mesh["strikes"]) == 15
    assert len(mesh["dtes"]) == 7
    assert len(mesh["iv_matrix"]) == 7
    assert len(mesh["iv_matrix"][0]) == 15
    assert mesh["atm_iv_pct"] > 0.0


def test_calculate_multileg_payoff_bull_call_spread():
    res = calculate_multileg_payoff("BULL_CALL_SPREAD", spot_price=350.0)
    assert res["strategy_type"] == "BULL_CALL_SPREAD"
    assert len(res["legs"]) == 2
    assert len(res["price_range"]) == 50
    assert len(res["payoff_curve"]) == 50
    assert res["max_loss"] > 0.0


def test_calculate_multileg_payoff_iron_condor():
    res = calculate_multileg_payoff("IRON_CONDOR", spot_price=350.0)
    assert res["strategy_type"] == "IRON_CONDOR"
    assert len(res["legs"]) == 4
    assert res["max_profit"] > 0.0


def test_calculate_multileg_payoff_long_straddle():
    res = calculate_multileg_payoff("LONG_STRADDLE", spot_price=350.0)
    assert res["strategy_type"] == "LONG_STRADDLE"
    assert len(res["legs"]) == 2
    assert res["max_profit"] == "Unlimited"


def test_solve_pinn_local_volatility():
    from src.options_surface import solve_pinn_local_volatility

    res = solve_pinn_local_volatility(
        spot_price=100.0,
        strikes=[90.0, 95.0, 100.0, 105.0, 110.0],
        dtes=[14, 30, 60],
        epochs=15,
    )
    assert res["status"] == "SUCCESS"
    assert res["is_arbitrage_free"] is True
    assert len(res["local_vol_matrix_pct"]) == 3
    assert len(res["local_vol_matrix_pct"][0]) == 5
    assert res["atm_local_vol_pct"] > 0.0
