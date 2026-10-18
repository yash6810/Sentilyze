"""
Random Matrix Theory (RMT) Spectral Denoising Engine.
=====================================================
Implementation of Marchenko-Pastur Spectral Filtering (Invention 1):
1. Purges Wishart noise from empirical asset covariance/correlation matrices.
2. Identifies structural macroeconomic and sector eigenvectors vs. noise spectrum.
3. Accelerated with PyTorch CUDA tensors when available, with zero-copy NumPy fallback.
4. Deep module design: clean public seams hiding complex eigenvalue transformations.

References:
- Bouchaud & Potters (2000), "Theory of Financial Risk and Derivative Pricing"
- Laloux, Cizeau, Bouchaud, Potters (1999), "Noise Dressing of Financial Correlation Matrices"
- Marcos López de Prado (2018), "Advances in Financial Machine Learning", Chapter 2
- Zura Kakushadze & Juan Andrés Serur (2018), "151 Trading Strategies", Chapter 3
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import json
import os
import numpy as np
import pandas as pd
import torch

from src.utils import get_logger

logger = get_logger(__name__)

DEFAULT_CLUSTERS_PATH = os.path.join("results", "rmt_spectral_clusters.json")


def compute_marchenko_pastur_bounds(
    n_assets: int,
    t_obs: int,
    variance: float = 1.0,
) -> Dict[str, float]:
    """
    Computes theoretical minimum and maximum eigenvalues under the Marchenko-Pastur distribution.

    Args:
        n_assets: Number of assets (N)
        t_obs: Number of temporal observations (T)
        variance: Theoretical residual noise variance (sigma^2)

    Returns:
        Dict containing lambda_minus, lambda_plus, and q_ratio (T/N).
    """
    if n_assets <= 0 or t_obs <= 0:
        raise ValueError(
            f"n_assets and t_obs must be positive, got N={n_assets}, T={t_obs}"
        )

    q_ratio = float(t_obs) / float(n_assets)
    sqrt_inv_q = np.sqrt(1.0 / q_ratio)

    lambda_plus = float(variance * (1.0 + sqrt_inv_q) ** 2)
    lambda_minus = float(variance * max(0.0, 1.0 - sqrt_inv_q) ** 2)

    return {
        "lambda_minus": lambda_minus,
        "lambda_plus": lambda_plus,
        "q_ratio": q_ratio,
    }


def clean_correlation_matrix_rmt(
    corr: np.ndarray,
    t_obs: int,
    variance: Optional[float] = None,
    use_cuda: bool = True,
) -> np.ndarray:
    """
    Purges random Wishart noise from an empirical correlation matrix using Marchenko-Pastur clipping.

    All eigenvalues lambda <= lambda_plus are identified as empirical noise and replaced
    by their spectral trace average, preserving total matrix variance and positive semi-definiteness.

    Args:
        corr: (N, N) empirical correlation matrix.
        t_obs: Number of historical return observations T used to compute corr.
        variance: Estimated variance of the noise subspace (computed automatically if None).
        use_cuda: If True, uses CUDA GPU acceleration if available.

    Returns:
        Cleaned, positive-definite (N, N) correlation matrix with exact 1.0 diagonals.
    """
    if not isinstance(corr, np.ndarray):
        corr = np.array(corr, dtype=np.float64)

    n_assets = corr.shape[0]
    if corr.shape[0] != corr.shape[1]:
        raise ValueError(f"Correlation matrix must be square, got shape {corr.shape}")

    if n_assets <= 1:
        return np.ones((n_assets, n_assets), dtype=np.float64)

    # Accelerate with PyTorch on CUDA if available
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    try:
        corr_tensor = torch.tensor(corr, dtype=torch.float64, device=device)
        # SVD / Hermitian Eigendecomposition: corr is symmetric real
        eigvals, eigvecs = torch.linalg.eigh(corr_tensor)

        # Convert back to numpy for Marchenko-Pastur thresholding
        vals = eigvals.cpu().numpy()
        vecs = eigvecs.cpu().numpy()

        # Sort descending
        idx_desc = np.argsort(vals)[::-1]
        vals = vals[idx_desc]
        vecs = vecs[:, idx_desc]

        # Clamp tiny negative eigenvalues from numerical roundoff
        vals = np.maximum(vals, 1e-12)

        # Theoretical bounds
        if variance is None:
            # Estimate noise variance: 1.0 - variance explained by top structural eigenvalues
            q = float(t_obs) / float(n_assets)
            lam_plus_est = (1.0 + np.sqrt(1.0 / q)) ** 2
            noise_idx = np.where(vals <= lam_plus_est)[0]
            if len(noise_idx) > 0:
                variance = float(np.mean(vals[noise_idx]))
            else:
                variance = 1.0

        bounds = compute_marchenko_pastur_bounds(n_assets, t_obs, variance)
        lam_plus = bounds["lambda_plus"]

        # Identify noise vs signal eigenvalues
        is_noise = vals <= lam_plus
        n_noise = int(np.sum(is_noise))

        cleaned_vals = vals.copy()
        if n_noise > 0:
            mean_noise_val = float(np.mean(vals[is_noise]))
            cleaned_vals[is_noise] = mean_noise_val

        # Reconstruct cleaned correlation matrix: C_tilde = V * Lambda_tilde * V^T
        reconstructed = vecs @ np.diag(cleaned_vals) @ vecs.T

        # Normalize diagonal to exactly 1.0: C_norm = D * C_tilde * D where D = diag(1 / sqrt(diag))
        diag_elements = np.maximum(np.diag(reconstructed), 1e-12)
        inv_sqrt_diag = 1.0 / np.sqrt(diag_elements)
        cleaned_corr = np.diag(inv_sqrt_diag) @ reconstructed @ np.diag(inv_sqrt_diag)

        # Guarantee strict symmetry and exactly 1.0 on diagonal
        cleaned_corr = (cleaned_corr + cleaned_corr.T) / 2.0
        np.fill_diagonal(cleaned_corr, 1.0)

        return np.clip(cleaned_corr, -1.0, 1.0)

    except Exception as e:
        logger.warning(f"RMT spectral cleaning failed, returning fallback: {e}")
        # Safe fallback: shrinkage toward identity
        return 0.85 * corr + 0.15 * np.eye(n_assets)


def extract_rmt_spectral_clusters(
    returns_df: pd.DataFrame,
    top_k: int = 5,
    t_obs: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extracts structural peer clusters across assets after purging empirical noise.

    Args:
        returns_df: DataFrame of asset returns (columns = tickers, rows = timesteps).
        top_k: Number of highest correlated structural peers to retain per ticker.
        t_obs: Number of observations (defaults to len(returns_df)).

    Returns:
        Dict mapping ticker -> list of top structural peers with cleaned correlations.
    """
    tickers = list(returns_df.columns)
    t = t_obs or len(returns_df)
    n = len(tickers)

    if n < 2:
        return {}

    empirical_corr = returns_df.corr().fillna(0.0).values
    cleaned_corr = clean_correlation_matrix_rmt(empirical_corr, t_obs=t)

    clusters: Dict[str, List[Dict[str, Any]]] = {}
    for i, tkr in enumerate(tickers):
        corrs_i = cleaned_corr[i, :]
        peer_indices = np.argsort(corrs_i)[::-1]

        peers_list = []
        for p_idx in peer_indices:
            if p_idx == i:
                continue
            peer_sym = tickers[p_idx]
            struct_corr = float(corrs_i[p_idx])
            peers_list.append(
                {
                    "peer": peer_sym,
                    "filtered_structural_correlation": round(struct_corr, 4),
                }
            )
            if len(peers_list) >= top_k:
                break
        clusters[tkr] = peers_list

    return clusters


def evaluate_asset_rmt_peers(
    ticker: str,
    top_k: int = 5,
    clusters_path: str = DEFAULT_CLUSTERS_PATH,
) -> List[Dict[str, Any]]:
    """
    High-level seam for retrieving RMT structural peers for an active trading asset.
    Reads pre-computed GPU clusters from results/rmt_spectral_clusters.json for sub-millisecond inference.
    """
    if os.path.exists(clusters_path):
        try:
            with open(clusters_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if ticker in data:
                return data[ticker][:top_k]
        except Exception as e:
            logger.debug(
                f"Failed to read cached RMT clusters from {clusters_path}: {e}"
            )

    return []
