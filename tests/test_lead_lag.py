import numpy as np
import pandas as pd
from src.lead_lag import compute_lead_lag_matrix, rank_market_price_leaders


def _generate_synthetic_price_series():
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=100)
    # Asset A is the leader
    noise_a = np.random.normal(0, 1, 100)
    series_a = pd.Series(100.0 + np.cumsum(noise_a), index=dates)

    # Asset B lags Asset A by 1 day
    series_b = pd.Series(100.0 + np.cumsum(np.roll(noise_a, 1) + np.random.normal(0, 0.2, 100)), index=dates)

    # Asset C is independent
    series_c = pd.Series(100.0 + np.cumsum(np.random.normal(0, 1, 100)), index=dates)

    return {"LEAD_A": series_a, "LAG_B": series_b, "IND_C": series_c}


def test_compute_lead_lag_matrix():
    data = _generate_synthetic_price_series()
    matrix = compute_lead_lag_matrix(data, max_lag=2)

    assert matrix.shape == (3, 3)
    assert set(matrix.index) == {"LEAD_A", "LAG_B", "IND_C"}
    assert matrix.loc["LEAD_A", "LEAD_A"] == 1.0


def test_rank_market_price_leaders():
    data = _generate_synthetic_price_series()
    matrix = compute_lead_lag_matrix(data, max_lag=2)
    ranks = rank_market_price_leaders(matrix)

    assert len(ranks) == 3
    for r in ranks:
        assert "ticker" in r
        assert "leads_count" in r
        assert "influence_score" in r
        assert "status" in r
