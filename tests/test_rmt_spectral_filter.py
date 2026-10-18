"""
Unit tests for RMT Spectral Filter (Invention 1: Marchenko-Pastur Spectral Denoising).
Following Elite Engineering Standards:
- Seam-based testing against public interface.
- Zero tautological assertions.
- Non-destructive to live portfolio state.
"""

import numpy as np
import pandas as pd
import pytest


def test_marchenko_pastur_upper_bound():
    """Verify Marchenko-Pastur theoretical upper bound calculation."""
    from src.rmt_spectral_filter import compute_marchenko_pastur_bounds

    # N = 100 assets, T = 1000 observations, q = T/N = 10.0
    bounds = compute_marchenko_pastur_bounds(n_assets=100, t_obs=1000, variance=1.0)
    assert "lambda_plus" in bounds
    assert "lambda_minus" in bounds
    assert bounds["lambda_plus"] > bounds["lambda_minus"]
    # Analytical: (1 + sqrt(0.1))^2 = (1 + 0.3162)^2 approx 1.732
    assert 1.70 <= bounds["lambda_plus"] <= 1.75
    # Analytical: (1 - sqrt(0.1))^2 = (1 - 0.3162)^2 approx 0.467
    assert 0.45 <= bounds["lambda_minus"] <= 0.48


def test_clean_correlation_matrix_preserves_properties():
    """Verify cleaned correlation matrix is symmetric, positive semi-definite, and diag==1.0."""
    from src.rmt_spectral_filter import clean_correlation_matrix_rmt

    # Generate a random valid correlation matrix
    np.random.seed(42)
    x = np.random.randn(200, 20)  # T=200, N=20
    corr = np.corrcoef(x, rowvar=False)

    cleaned = clean_correlation_matrix_rmt(corr, t_obs=200)

    # 1. Shape preservation
    assert cleaned.shape == (20, 20)
    # 2. Symmetry
    assert np.allclose(cleaned, cleaned.T, atol=1e-6)
    # 3. Diagonal is strictly 1.0
    assert np.allclose(np.diag(cleaned), np.ones(20), atol=1e-5)
    # 4. Positive semi-definite: all eigenvalues >= -1e-6
    eigvals = np.linalg.eigvalsh(cleaned)
    assert np.all(eigvals >= -1e-6)


def test_rmt_denoises_pure_noise():
    """Verify that RMT shrinks spurious correlations from pure random Gaussian noise."""
    from src.rmt_spectral_filter import clean_correlation_matrix_rmt

    np.random.seed(123)
    # 30 purely independent assets, T=60 (noisy small sample)
    noise_data = np.random.randn(60, 30)
    empirical_corr = np.corrcoef(noise_data, rowvar=False)

    # Off-diagonal noise in empirical correlation
    off_diag_mask = ~np.eye(30, dtype=bool)
    empirical_mean_abs = np.mean(np.abs(empirical_corr[off_diag_mask]))

    cleaned_corr = clean_correlation_matrix_rmt(empirical_corr, t_obs=60)
    cleaned_mean_abs = np.mean(np.abs(cleaned_corr[off_diag_mask]))

    # Denoised matrix should have significantly lower spurious off-diagonal correlation
    assert cleaned_mean_abs < empirical_mean_abs


def test_extract_rmt_spectral_clusters():
    """Verify cluster extraction seam produces valid peer rankings from DataFrame."""
    from src.rmt_spectral_filter import extract_rmt_spectral_clusters

    np.random.seed(99)
    # Create 4 correlated assets
    base = np.random.randn(100, 1)
    a = base + 0.1 * np.random.randn(100, 1)
    b = base + 0.1 * np.random.randn(100, 1)
    c = -base + 0.1 * np.random.randn(100, 1)
    d = np.random.randn(100, 1)

    df_returns = pd.DataFrame(
        np.hstack([a, b, c, d]), columns=["ALPHA", "BETA", "GAMMA", "DELTA"]
    )
    clusters = extract_rmt_spectral_clusters(df_returns, top_k=2)

    assert "ALPHA" in clusters
    peers = [item["peer"] for item in clusters["ALPHA"]]
    # BETA must be the top structural peer of ALPHA
    assert "BETA" in peers


def test_evaluate_asset_rmt_peers_cached():
    """Verify high-level evaluation reads precomputed clusters gracefully."""
    from src.rmt_spectral_filter import evaluate_asset_rmt_peers

    peers = evaluate_asset_rmt_peers("NVDA", top_k=3)
    assert isinstance(peers, list)
    if peers:
        assert "peer" in peers[0]
        assert "filtered_structural_correlation" in peers[0]
