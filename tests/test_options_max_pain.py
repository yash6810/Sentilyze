"""
Unit tests for Options Max Pain & Strike Pinning Radar (Option 4).
"""

import pandas as pd
import numpy as np
import pytest
from src.options_gex import (
    calculate_max_pain,
    calculate_options_max_pain,
    compute_gamma_exposure_profile,
)


def test_calculate_max_pain_analytical_minimum():
    # Construct a simple options board where strike 100 minimizes holder payout
    # Calls at 90, 100, 110
    # Puts at 90, 100, 110
    calls = pd.DataFrame(
        {
            "strike": [90.0, 100.0, 110.0],
            "openInterest": [1000, 200, 50],
            "volume": [100, 20, 5],
            "dte": [5, 5, 5],
        }
    )
    puts = pd.DataFrame(
        {
            "strike": [90.0, 100.0, 110.0],
            "openInterest": [50, 200, 1000],
            "volume": [5, 20, 100],
            "dte": [5, 5, 5],
        }
    )

    res = calculate_max_pain(df_calls=calls, df_puts=puts, spot_price=99.0, avg_dte=5.0)

    assert res["max_pain_strike"] == 100.0
    assert abs(res["distance_to_spot_pct"]) < 2.0
    assert res["pinning_probability_pct"] >= 50.0
    assert "PINNING" in res["pinning_badge"]
    assert res["total_holder_payout_m"] >= 0.0


def test_calculate_options_max_pain_profile():
    # Test high-level wrapper
    res = calculate_options_max_pain("NVDA", max_expiries=1)
    assert res["ticker"] == "NVDA"
    assert "max_pain" in res
    assert res["max_pain"]["max_pain_strike"] > 0
    assert "call_wall" in res
    assert "put_wall" in res
    assert "gamma_flip" in res


def test_compute_gamma_exposure_profile_contains_max_pain():
    gex = compute_gamma_exposure_profile("AAPL", max_expiries=1)
    assert "max_pain" in gex
    assert "pinning_probability_pct" in gex["max_pain"]
    assert gex["max_pain"]["max_pain_strike"] > 0
