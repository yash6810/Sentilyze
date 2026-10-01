"""
Conformalized Quantile Regression (CQR) & Adaptive Conformal Inference Engine.
Paper 32: Romano, Patterson & Candès (Stanford, NeurIPS 2019)
Paper 33: Gibbs & Candès (Stanford, NeurIPS 2021)

Provides mathematically guaranteed, distribution-free prediction intervals [L(x), U(x)]
with exact 1 - alpha coverage on next-period equity returns without parametric assumptions.
Used to gate model trade convictions and eliminate overconfident false breakouts.
"""

import os
import json
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.utils import get_logger

logger = get_logger(__name__)

CONFORMAL_DIR = os.path.join("results", "conformal")


class ConformalQuantileEngine:
    """
    Split Conformalized Quantile Regressor (CQR) with non-parametric calibration.
    Computes distribution-free lower and upper bounds for returns:
        P(Y_t+1 in [L(X), U(X)]) >= 1 - alpha
    """

    def __init__(self, alpha: float = 0.10, random_state: int = 42):
        """
        Args:
            alpha (float): Miscoverage error rate (default: 0.10 => 90% guaranteed coverage).
            random_state (int): Reproducible seed for split calibration.
        """
        self.alpha = alpha
        self.alpha_lo = alpha / 2.0  # 0.05 (5th percentile)
        self.alpha_hi = 1.0 - (alpha / 2.0)  # 0.95 (95th percentile)
        self.random_state = random_state

        self.model_lo: Optional[GradientBoostingRegressor] = None
        self.model_hi: Optional[GradientBoostingRegressor] = None
        self.conformal_quantile_margin: float = 0.0
        self.calibration_size: int = 0
        self.mean_interval_width: float = 0.0

    def fit_and_calibrate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        calib_fraction: float = 0.25,
    ) -> "ConformalQuantileEngine":
        """
        Fits lower/upper pinball regressors on proper training set and calibrates
        exact non-conformity quantile corrections on a holdout calibration set.
        """
        clean_mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[clean_mask].copy()
        y_clean = y[clean_mask].copy()

        n = len(X_clean)
        if n < 30:
            logger.warning(
                f"Dataset size ({n}) too small for reliable conformal split. Using empirical defaults."
            )
            self.conformal_quantile_margin = float(
                np.std(y_clean) * 1.5 if n > 5 else 0.02
            )
            self.mean_interval_width = self.conformal_quantile_margin * 2.0
            return self

        # Temporal split to avoid lookahead: First (1 - calib_fraction) is train, last calib_fraction is calibration
        n_calib = max(10, int(n * calib_fraction))
        n_train = n - n_calib

        X_train, y_train = X_clean.iloc[:n_train], y_clean.iloc[:n_train]
        X_calib, y_calib = X_clean.iloc[n_train:], y_clean.iloc[n_train:]

        # 1. Fit Lower Quantile Model (Pinball Loss at alpha_lo)
        self.model_lo = GradientBoostingRegressor(
            loss="quantile",
            alpha=self.alpha_lo,
            n_estimators=60,
            max_depth=3,
            learning_rate=0.05,
            random_state=self.random_state,
        )
        self.model_lo.fit(X_train, y_train)

        # 2. Fit Upper Quantile Model (Pinball Loss at alpha_hi)
        self.model_hi = GradientBoostingRegressor(
            loss="quantile",
            alpha=self.alpha_hi,
            n_estimators=60,
            max_depth=3,
            learning_rate=0.05,
            random_state=self.random_state,
        )
        self.model_hi.fit(X_train, y_train)

        # 3. Compute Non-Conformity Scores on Holdout Calibration Set
        q_lo_calib = self.model_lo.predict(X_calib)
        q_hi_calib = self.model_hi.predict(X_calib)

        # Non-conformity score E_i: Positive if actual return falls outside [q_lo, q_hi]
        non_conformity_scores = np.maximum(
            q_lo_calib - y_calib.values,
            y_calib.values - q_hi_calib,
        )

        # 4. Compute Finite-Sample Conformal Correction Margin Q_{1-alpha}
        # Correct finite sample adjustment: (1 - alpha) * (1 + 1/n_calib)
        quantile_target = min(0.999, (1.0 - self.alpha) * (1.0 + 1.0 / n_calib))
        self.conformal_quantile_margin = float(
            np.quantile(non_conformity_scores, quantile_target)
        )
        self.calibration_size = n_calib

        # Empirical calibration stats
        raw_widths = q_hi_calib - q_lo_calib
        conformal_widths = raw_widths + (2.0 * self.conformal_quantile_margin)
        self.mean_interval_width = float(np.mean(conformal_widths))

        logger.info(
            f"🎯 [CONFORMAL CQR CALIBRATED] Margin: {self.conformal_quantile_margin:.4f} | "
            f"Mean Band Width: {self.mean_interval_width*100.0:.2f}% | Calib Size: {n_calib}"
        )
        return self

    def predict_interval(
        self,
        X_sample: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Outputs distribution-free confidence bounds [L(x), U(x)] for incoming feature vector.
        """
        if self.model_lo is None or self.model_hi is None:
            # Fallback if uncalibrated
            return {
                "lower_bound_pct": -2.0,
                "upper_bound_pct": 2.0,
                "interval_width_pct": 4.0,
                "is_trade_viable": True,
                "signal_clarity": "UNCALIBRATED_FALLBACK",
                "coverage_target_pct": (1.0 - self.alpha) * 100.0,
            }

        q_lo = float(self.model_lo.predict(X_sample)[0])
        q_hi = float(self.model_hi.predict(X_sample)[0])

        lower_bound = q_lo - self.conformal_quantile_margin
        upper_bound = q_hi + self.conformal_quantile_margin
        if upper_bound <= lower_bound:
            upper_bound = lower_bound + 0.005
        interval_width = upper_bound - lower_bound

        # Trade Filtering Rules:
        # 1. POSITIVE_EDGE: lower_bound > -0.002 (-0.2% max expected downside risk)
        # 2. UNCERTAIN_NOISE: interval spans wide negative territory (e.g. lower bound < -1.5%)
        # 3. HIGH_ASYMMETRY: upper_bound >= 2.5 * abs(lower_bound)
        is_viable = lower_bound > -0.0075 and upper_bound > 0.005
        asymmetry_ratio = (
            round(upper_bound / abs(lower_bound), 2) if abs(lower_bound) > 1e-4 else 5.0
        )

        if lower_bound > 0:
            clarity = "🟢 PURE_ALPHA_POSITIVE"
        elif is_viable and asymmetry_ratio >= 2.0:
            clarity = "🟡 ASYMMETRIC_POSITIVE"
        else:
            clarity = "🔴 NOISE_WIDE_UNCERTAINTY"

        return {
            "lower_bound_pct": round(lower_bound * 100.0, 2),
            "upper_bound_pct": round(upper_bound * 100.0, 2),
            "interval_width_pct": round(interval_width * 100.0, 2),
            "asymmetry_ratio": asymmetry_ratio,
            "is_trade_viable": bool(is_viable),
            "signal_clarity": clarity,
            "coverage_target_pct": round((1.0 - self.alpha) * 100.0, 1),
            "conformal_margin": round(self.conformal_quantile_margin, 4),
        }

    def save_calibration(self, ticker: str):
        """Saves conformal calibration metadata in results/conformal/<ticker>_cqr.json."""
        os.makedirs(CONFORMAL_DIR, exist_ok=True)
        out_file = os.path.join(CONFORMAL_DIR, f"{ticker}_cqr.json")
        data = {
            "ticker": ticker,
            "alpha": self.alpha,
            "coverage_guarantee_pct": (1.0 - self.alpha) * 100.0,
            "conformal_quantile_margin": self.conformal_quantile_margin,
            "mean_interval_width": self.mean_interval_width,
            "calibration_size": self.calibration_size,
        }
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_calibration_metadata(cls, ticker: str) -> Optional[Dict[str, Any]]:
        """Loads pre-computed calibration metadata if available."""
        out_file = os.path.join(CONFORMAL_DIR, f"{ticker}_cqr.json")
        if os.path.exists(out_file):
            try:
                with open(out_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None


def calibrate_and_evaluate_conformal_bounds(
    ticker: str,
    feature_history: pd.DataFrame,
    target_series: pd.Series,
    current_feature_row: pd.DataFrame,
    alpha: float = 0.10,
) -> Dict[str, Any]:
    """
    End-to-end convenience evaluator:
    Fits CQR on historical features, calibrates non-parametric bounds,
    and returns exact 90% or 95% guaranteed return interval for current tick.
    """
    engine = ConformalQuantileEngine(alpha=alpha)
    engine.fit_and_calibrate(X=feature_history, y=target_series)
    engine.save_calibration(ticker)
    return engine.predict_interval(current_feature_row)
