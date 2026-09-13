"""
Unit tests for Options Gamma Exposure (GEX) & Strike Walls Engine.
"""

import pandas as pd

from src.options_gex import (
    calculate_black_scholes_gamma,
    compute_gamma_exposure_profile,
    calculate_options_gex,
)


def test_black_scholes_gamma_atm_peak():
    """Verify Black-Scholes gamma is positive and peaks near ATM."""
    spot = 100.0
    dte = 30.0
    iv = 0.25

    gamma_atm = calculate_black_scholes_gamma(spot=spot, strike=100.0, dte=dte, iv=iv)
    gamma_otm = calculate_black_scholes_gamma(spot=spot, strike=120.0, dte=dte, iv=iv)
    gamma_itm = calculate_black_scholes_gamma(spot=spot, strike=80.0, dte=dte, iv=iv)

    assert gamma_atm > 0.0
    assert gamma_atm > gamma_otm
    assert gamma_atm > gamma_itm


def test_black_scholes_gamma_edge_cases():
    """Verify gamma calculation handles zero or negative inputs gracefully."""
    assert (
        calculate_black_scholes_gamma(spot=0.0, strike=100.0, dte=30.0, iv=0.25) == 0.0
    )
    assert (
        calculate_black_scholes_gamma(spot=100.0, strike=0.0, dte=30.0, iv=0.25) == 0.0
    )
    assert (
        calculate_black_scholes_gamma(spot=100.0, strike=100.0, dte=0.0, iv=0.25) == 0.0
    )
    assert (
        calculate_black_scholes_gamma(spot=100.0, strike=100.0, dte=30.0, iv=0.0) == 0.0
    )


def test_compute_gamma_exposure_profile_structure():
    """Verify profile calculation yields standard keys and sensible levels."""
    profile = compute_gamma_exposure_profile("NVDA", max_expiries=2)

    assert "ticker" in profile
    assert profile["ticker"] == "NVDA"
    assert "spot_price" in profile
    assert profile["spot_price"] > 0.0
    assert "call_wall" in profile
    assert "put_wall" in profile
    assert "gamma_flip_line" in profile
    assert "total_net_gex_m" in profile
    assert "market_maker_regime" in profile
    assert "strikes_df" in profile

    df_strikes = profile["strikes_df"]
    assert isinstance(df_strikes, pd.DataFrame)
    assert not df_strikes.empty
    assert "strike" in df_strikes.columns
    assert "call_gex_m" in df_strikes.columns
    assert "put_gex_m" in df_strikes.columns
    assert "net_gex_m" in df_strikes.columns

    # Call GEX must be >= 0 and Put GEX must be <= 0
    assert (df_strikes["call_gex_m"] >= 0).all()
    assert (df_strikes["put_gex_m"] <= 0).all()


def test_calculate_options_gex_alias():
    """Verify calculate_options_gex alias behaves identically to compute_gamma_exposure_profile."""
    res = calculate_options_gex("AAPL")
    assert res["ticker"] == "AAPL"
    assert "spot_price" in res
    assert "market_maker_regime" in res
