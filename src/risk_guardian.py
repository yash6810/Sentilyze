"""
Mahalanobis Anomaly Sentinel & Dynamic Regime Guardian (Sprint 1, Module 1.8 / Idea 49)

Monitors incoming market feature vectors against historical multivariate reference distributions.
Calculates Mahalanobis statistical distance D_M(x) and Chi-Square p-values to catch
out-of-distribution market dislocations (flash crashes, unprecedented liquidity dry-ups, regime drift).
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.data_ingestion import get_price_history

logger = logging.getLogger("Sentilyze.RiskGuardian")
logging.basicConfig(level=logging.INFO)


class MahalanobisAnomalySentinel:
    """
    Multivariate statistical anomaly detector based on Mahalanobis distance D_M(x).
    Evaluates whether recent market features for an asset belong to historical normal regimes.
    """

    FEATURE_COLUMNS = [
        "return_5d",
        "volatility_20d",
        "rsi_14",
        "volume_ratio",
        "dist_to_sma50",
    ]

    def __init__(self, significance_level: float = 0.01):
        self.significance_level = significance_level

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts normalized technical feature set from OHLCV DataFrame."""
        if df.empty or len(df) < 50:
            return pd.DataFrame()

        feats = pd.DataFrame(index=df.index)
        closes = df["Close"]

        # 1. 5-day return
        feats["return_5d"] = closes.pct_change(5).fillna(0.0)

        # 2. 20-day realized volatility
        feats["volatility_20d"] = closes.pct_change().rolling(20).std().fillna(0.0)

        # 3. RSI 14
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        feats["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50.0)

        # 4. Volume ratio (Volume / 20d SMA Volume)
        if "Volume" in df.columns:
            vol = df["Volume"].replace(0, 1e-6)
            vol_sma = vol.rolling(20).mean().replace(0, 1e-6)
            feats["volume_ratio"] = (vol / vol_sma).fillna(1.0)
        else:
            feats["volume_ratio"] = 1.0

        # 5. Distance to 50-day SMA
        sma50 = closes.rolling(50).mean().bfill()
        feats["dist_to_sma50"] = ((closes - sma50) / sma50).fillna(0.0)

        return feats[self.FEATURE_COLUMNS].dropna()

    def fit_reference_distribution(
        self, feats: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes reference mean vector and robust inverted covariance matrix.
        Uses pseudo-inverse to avoid numerical instability on near-collinear features.
        """
        X = feats.values
        mu = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)

        # Regularize covariance diagonal with small epsilon for numerical stability
        cov += np.eye(cov.shape[0]) * 1e-6
        inv_cov = np.linalg.pinv(cov)
        return mu, inv_cov

    def compute_mahalanobis_distance(
        self, x: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray
    ) -> float:
        """D_M(x) = sqrt((x - mu)^T * Sigma^-1 * (x - mu))"""
        diff = x - mu
        d_sq = float(np.dot(np.dot(diff, inv_cov), diff.T))
        return float(np.sqrt(max(d_sq, 0.0)))

    def audit_asset_regime(
        self, ticker: str, spot_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Audits incoming features for an asset and flags abnormal distribution drift.
        """
        try:
            df = get_price_history(ticker, period="1y", use_cache=True)
        except Exception:
            df = pd.DataFrame()

        if df.empty or len(df) < 60:
            return {
                "ticker": ticker,
                "status": "INSUFFICIENT_DATA",
                "is_anomaly": False,
                "mahalanobis_distance": 0.0,
                "p_value": 1.0,
                "regime_verdict": "REFERENCE_DATA_UNAVAILABLE",
            }

        feats = self.extract_features(df)
        if len(feats) < 40:
            return {
                "ticker": ticker,
                "status": "INSUFFICIENT_FEATURE_ROWS",
                "is_anomaly": False,
                "mahalanobis_distance": 0.0,
                "p_value": 1.0,
                "regime_verdict": "REFERENCE_DATA_UNAVAILABLE",
            }

        # Train on history up to t-1
        train_feats = feats.iloc[:-1]
        latest_feat = feats.iloc[-1].values

        mu, inv_cov = self.fit_reference_distribution(train_feats)
        d_m = self.compute_mahalanobis_distance(latest_feat, mu, inv_cov)

        df_dim = len(self.FEATURE_COLUMNS)
        # Chi-Square survival function for squared distance
        d_sq = d_m**2
        p_val = float(chi2.sf(d_sq, df=df_dim))
        critical_d = float(np.sqrt(chi2.ppf(1.0 - self.significance_level, df=df_dim)))

        is_anomaly = bool(d_m > critical_d or p_val < self.significance_level)

        if is_anomaly:
            verdict = "SEVERE_REGIME_ANOMALY_UNSEEN_STATE"
            action = "CRO_VETO_OR_HAIRCUT_RECOMMENDED"
        elif d_m > (critical_d * 0.80):
            verdict = "ELEVATED_STATISTICAL_DISPERSION"
            action = "MONITOR_CLOSELY"
        else:
            verdict = "NORMAL_REGIME_STATISTICALLY_STABLE"
            action = "STANDARD_EXECUTION"

        return {
            "status": "SUCCESS",
            "ticker": ticker,
            "mahalanobis_distance": round(d_m, 3),
            "critical_threshold_99": round(critical_d, 3),
            "p_value": round(p_val, 5),
            "degrees_of_freedom": df_dim,
            "is_anomaly": is_anomaly,
            "regime_verdict": verdict,
            "action_recommendation": action,
        }


def evaluate_mahalanobis_sentinel(ticker: str) -> Dict[str, Any]:
    sentinel = MahalanobisAnomalySentinel()
    return sentinel.audit_asset_regime(ticker)
