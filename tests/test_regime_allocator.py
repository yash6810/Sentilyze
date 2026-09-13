"""
Unit tests for 3-State Gaussian Hidden Markov Model & Meta-Regime Allocator.
"""

import pytest
import numpy as np
import pandas as pd

from src.regime_allocator import (
    MetaRegimeAllocator,
    MetaRegimeReport,
    REGIME_BULL,
    REGIME_CHOP,
    REGIME_CRISIS,
)


@pytest.fixture
def mock_price_history():
    """Generate 100 days of price history with bull and volatile phases."""
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    rng = np.random.RandomState(42)
    rets = np.concatenate(
        [
            rng.normal(0.001, 0.008, 60),  # Bull phase
            rng.normal(-0.002, 0.025, 60),  # High vol phase
        ]
    )
    prices = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {
            "Open": prices * 0.999,
            "High": prices * 1.005,
            "Low": prices * 0.995,
            "Close": prices,
            "Volume": 50000000,
        },
        index=dates,
    )


def test_regime_allocator_fit_and_states(mock_price_history):
    """Verify HMM discovers 3 regimes and sorts them properly."""
    allocator = MetaRegimeAllocator(
        benchmark_ticker="TEST", n_regimes=3, random_state=42
    )
    allocator.fit(mock_price_history)

    assert allocator._fitted is True
    assert len(allocator.state_map) == 3
    state_names = set(allocator.state_map.values())
    assert REGIME_BULL in state_names
    assert REGIME_CHOP in state_names
    assert REGIME_CRISIS in state_names

    # Check transition matrix
    T = allocator.transition_matrix
    assert T.shape == (3, 3)
    # Rows should sum to approximately 1.0
    row_sums = T.sum(axis=1)
    np.testing.assert_allclose(row_sums, [1.0, 1.0, 1.0], atol=1e-3)


def test_regime_allocator_analysis_report(mock_price_history):
    """Verify analysis report structure and allocation bounds."""
    allocator = MetaRegimeAllocator(
        benchmark_ticker="TEST", n_regimes=3, random_state=42
    )
    report = allocator.analyze(mock_price_history)

    assert isinstance(report, MetaRegimeReport)
    assert report.benchmark_ticker == "TEST"
    assert report.current_regime in [REGIME_BULL, REGIME_CHOP, REGIME_CRISIS]
    assert 0.0 <= report.current_confidence <= 1.0
    assert 0.0 <= report.recommended_allocation <= 1.0

    # Probabilities should sum to 1.0
    total_p = sum(report.state_probabilities.values())
    assert abs(total_p - 1.0) < 1e-3

    # Historical regime dataframe
    assert not report.historical_regimes.empty
    assert "Regime" in report.historical_regimes.columns
    assert "Prob_Bull" in report.historical_regimes.columns


def test_regime_allocator_fallback_generation():
    """Verify fallback generation works if remote API fails."""
    allocator = MetaRegimeAllocator(
        benchmark_ticker="NONEXISTENT_TICKER_XYZ", random_state=42
    )
    df_fb = allocator._generate_fallback_history(days=100)

    assert isinstance(df_fb, pd.DataFrame)
    assert len(df_fb) == 100
    assert "Close" in df_fb.columns
    assert (df_fb["Close"] > 0).all()
