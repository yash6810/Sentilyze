import numpy as np
import pandas as pd
import pytest
from src.microstructure import (
    compute_vpin,
    compute_corwin_schultz_spread,
    compute_roll_spread,
    compute_amihud_illiquidity,
    evaluate_microstructure_health,
)


@pytest.fixture
def sample_market_df():
    np.random.seed(42)
    n = 60
    base_price = 150.0
    returns = np.random.normal(0.0005, 0.015, n)
    closes = base_price * np.cumprod(1 + returns)
    highs = closes * (1 + np.abs(np.random.normal(0, 0.008, n)))
    lows = closes * (1 - np.abs(np.random.normal(0, 0.008, n)))
    opens = (highs + lows) / 2.0
    volumes = np.random.uniform(500_000, 2_000_000, n)

    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        }
    )
    return df


def test_compute_vpin(sample_market_df):
    res = compute_vpin(sample_market_df, n_buckets=10)
    assert "vpin" in res
    assert 0.0 <= res["vpin"] <= 1.0
    assert res["toxicity_regime"] in [
        "CLEAN_FLOW",
        "MODERATE_TOXICITY",
        "HIGH_TOXICITY",
    ]
    assert "is_toxic" in res
    assert 0.0 <= res["buyer_volume_ratio"] <= 1.0
    assert 0.0 <= res["seller_volume_ratio"] <= 1.0


def test_compute_corwin_schultz_spread(sample_market_df):
    spread = compute_corwin_schultz_spread(sample_market_df, window=10)
    assert isinstance(spread, pd.Series)
    assert len(spread) == len(sample_market_df)
    # Spreads must be non-negative
    valid_spreads = spread.dropna()
    assert (valid_spreads >= 0.0).all()


def test_compute_roll_spread(sample_market_df):
    spread = compute_roll_spread(sample_market_df, window=10)
    assert isinstance(spread, pd.Series)
    assert len(spread) == len(sample_market_df)
    valid_spreads = spread.dropna()
    assert (valid_spreads >= 0.0).all()


def test_compute_amihud_illiquidity(sample_market_df):
    illiq = compute_amihud_illiquidity(sample_market_df, window=10)
    assert isinstance(illiq, pd.Series)
    assert len(illiq) == len(sample_market_df)
    valid_illiq = illiq.dropna()
    assert (valid_illiq >= 0.0).all()


def test_evaluate_microstructure_health(sample_market_df):
    health = evaluate_microstructure_health("NVDA", df=sample_market_df)
    assert health["ticker"] == "NVDA"
    assert health["status"] == "AUDIT_COMPLETE"
    assert 0.0 <= health["microstructure_score"] <= 100.0
    assert 0.0 <= health["vpin"] <= 1.0
    assert health["corwin_schultz_spread_bps"] >= 0.0
    assert health["roll_spread_bps"] >= 0.0
    assert 0.0 < health["action_penalty_multiplier"] <= 1.0
    assert "execution_recommendation" in health


def test_microstructure_insufficient_data():
    empty_df = pd.DataFrame()
    res_vpin = compute_vpin(empty_df)
    assert res_vpin["is_toxic"] is False
    assert res_vpin["vpin"] == 0.35

    health = evaluate_microstructure_health("DUMMY", df=empty_df)
    assert health["status"] == "INSUFFICIENT_DATA"
    assert health["microstructure_score"] == 50.0
