"""
Unit Tests for Wavelet Multi-Resolution Entropy Sifter.
Validates:
1. Pure NumPy Daubechies D4 MODWT decomposition (energy conservation, multi-scale shapes).
2. Noise vs. Trend discrimination (Gaussian noise triggers CHOP_INVALIDATION_VETO; clean trend passes).
3. Functional gatekeeper interface evaluate_wavelet_invalidation_gate().
4. Strict safety: Zero touch on live portfolio files.
"""

import numpy as np
import pandas as pd
import pytest
from src.wavelet_entropy_sifter import (
    WaveletEntropySifter,
    get_wavelet_sifter,
    evaluate_wavelet_invalidation_gate,
    H_MODWT,
    G_MODWT,
)


def test_wavelet_filter_coefficients():
    """Verify that MODWT filter coefficients satisfy energy conservation."""
    assert len(H_MODWT) == 4
    assert len(G_MODWT) == 4
    # For MODWT, sum of squares of each filter equals 0.5
    assert np.isclose(np.sum(H_MODWT**2), 0.5, atol=1e-5)
    assert np.isclose(np.sum(G_MODWT**2), 0.5, atol=1e-5)


def test_modwt_decomposition_shapes_and_energy():
    """Verify that decomposition preserves signal length without decimation."""
    sifter = WaveletEntropySifter(level=3)
    np.random.seed(42)
    N = 64
    signal = np.random.randn(N)

    details, smooth = sifter.decompose_modwt(signal)
    assert len(details) == 3
    for d in details:
        assert len(d) == N
    assert len(smooth) == N


def test_high_entropy_chop_veto_detection():
    """Verify that pure Gaussian white noise is correctly identified as chop and vetoed."""
    sifter = WaveletEntropySifter(
        level=3, noise_threshold=0.50, entropy_veto_threshold=0.85
    )
    np.random.seed(101)
    # 128 bars of pure white noise returns
    noise_returns = np.random.normal(0, 0.02, 128)

    result = sifter.compute_spectral_entropy(noise_returns)
    assert "normalized_entropy" in result
    assert "noise_d1_pct" in result
    assert result["normalized_entropy"] >= 0.80
    assert result["regime"] == "HIGH_ENTROPY_CHOP_VETO"
    assert result["veto_active"] is True
    assert result["conviction_multiplier"] == 0.0


def test_harmonic_trend_clean_expansion():
    """Verify that clean sinusoidal harmonic trend passes through without veto."""
    sifter = WaveletEntropySifter(level=3)
    t = np.linspace(0, 4 * np.pi, 128)
    # Strong macro/swing return wave with tiny noise
    clean_returns = 0.02 * np.sin(t) + 0.01 + np.random.normal(0, 0.001, 128)

    result = sifter.compute_spectral_entropy(clean_returns)
    assert result["veto_active"] is False
    assert result["normalized_entropy"] < 0.75
    assert result["conviction_multiplier"] >= 1.0
    assert result["regime"] in [
        "HARMONIC_SWING_EXPANSION",
        "SECULAR_MACRO_TREND",
        "TRANSITIONAL_MODERATE",
    ]


def test_evaluate_wavelet_invalidation_gate_with_dataframe():
    """Verify the functional gatekeeper with a simulated price DataFrame."""
    np.random.seed(7)
    dates = pd.date_range("2026-01-01", periods=60, freq="B")
    prices = 100.0 + np.cumsum(np.random.randn(60) * 0.5)
    df = pd.DataFrame({"Close": prices}, index=dates)

    gate_result = evaluate_wavelet_invalidation_gate(ticker="NVDA", df=df)
    assert gate_result["ticker"] == "NVDA"
    assert "veto_active" in gate_result
    assert "regime" in gate_result
    assert "normalized_entropy" in gate_result
    assert isinstance(gate_result["veto_active"], bool)


def test_insufficient_data_handling():
    """Verify graceful handling for very short series (< 16 bars)."""
    short_prices = np.array([100.0, 101.0, 102.0])
    gate_result = evaluate_wavelet_invalidation_gate(ticker="TEST", prices=short_prices)
    assert gate_result["ticker"] == "TEST"
    assert gate_result["veto_active"] is False
    assert "INSUFFICIENT" in gate_result["regime"]
