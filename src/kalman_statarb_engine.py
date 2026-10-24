"""Dynamic Kalman-Filter Statistical Arbitrage & Cointegration Pairs Engine.

This module implements an online State-Space Kalman Filter for dynamic
hedge-ratio estimation between cointegrated asset pairs (e.g., NVDA/AMD,
MSFT/GOOGL, QQQ/SPY). Unlike static OLS regression which breaks down under
regime shifts, the Kalman Filter updates the state vector [alpha_t, beta_t]
in real-time, yielding instantaneous hedge ratios, prediction error spreads,
instantaneous spread variance, and Ornstein-Uhlenbeck half-life estimates.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("Sentilyze.KalmanStatArb")


@dataclass
class PairSpreadState:
    """Snapshot of Kalman Filter state for an asset pair."""

    asset_y: str
    asset_x: str
    alpha: float
    beta: float  # Dynamic hedge ratio (Qty X to hedge 1 unit of Y)
    spread: float  # Current error e_t = y_t - (alpha + beta * x_t)
    spread_std: float  # Instantaneous sqrt(S_t)
    z_score: float  # Standardized spread z_t = e_t / sqrt(S_t)
    half_life_days: float  # Ornstein-Uhlenbeck mean-reversion half-life
    signal: str  # 'LONG_SPREAD', 'SHORT_SPREAD', 'EXIT', 'NEUTRAL', 'STOP_LOSS'
    confidence: float
    cointegration_stable: bool


class DynamicKalmanFilter:
    """Online 2-State Kalman Filter for dynamic hedge ratio [alpha, beta]."""

    def __init__(
        self,
        delta: float = 1e-4,  # Transition variance scaling factor
        R: float = 1e-3,  # Observation measurement noise
    ) -> None:
        """Initialize state space parameters."""
        self.delta = delta
        self.R = R
        # State vector: [alpha, beta]^T
        self.theta = np.zeros((2, 1))
        # Initial state covariance matrix
        self.P = np.zeros((2, 2))
        self.initialized = False

    def update(self, y: float, x: float) -> Tuple[float, float, float, float]:
        """Update Kalman filter state with new observations y and x.

        Returns:
            Tuple of (alpha_t, beta_t, error_spread, spread_variance)
        """
        H = np.array([[1.0, x]])  # Observation matrix (1 x 2)

        if not self.initialized:
            # Initialize with reasonable starting values
            self.theta = np.array([[0.0], [y / max(x, 1e-6)]])
            self.P = np.eye(2) * 1.0
            self.initialized = True

        # System noise covariance Q_t
        # Standard formulation: Q = (delta / (1 - delta)) * eye(2)
        Q = (self.delta / (1.0 - self.delta)) * np.eye(2)

        # 1. State Prediction:
        # theta_{t|t-1} = theta_{t-1} (random walk model)
        # P_{t|t-1} = P_{t-1} + Q
        P_pred = self.P + Q

        # 2. Measurement Prediction & Residual:
        y_pred = float((H @ self.theta).item())
        error = float(y - y_pred)

        # 3. Innovation Covariance S_t:
        S = float((H @ P_pred @ H.T + self.R).item())

        # 4. Kalman Gain K_t:
        K = P_pred @ H.T / max(S, 1e-8)

        # 5. State Update:
        self.theta = self.theta + K * error
        self.P = (np.eye(2) - K @ H) @ P_pred

        alpha = float(self.theta[0, 0])
        beta = float(self.theta[1, 0])

        return alpha, beta, error, S


class KalmanStatArbEngine:
    """Institutional Statistical Arbitrage & Cointegration Pairs Engine."""

    DEFAULT_PAIRS: List[Tuple[str, str]] = [
        ("NVDA", "AMD"),
        ("MSFT", "GOOGL"),
        ("AAPL", "MSFT"),
        ("QQQ", "SPY"),
        ("AMZN", "META"),
        ("COIN", "MSTR"),
    ]

    def __init__(
        self,
        z_entry: float = 2.0,
        z_exit: float = 0.35,
        z_stop: float = 3.8,
        delta: float = 1e-4,
        R: float = 1e-3,
    ) -> None:
        """Initialize parameters for pairs trading engine."""
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_stop = z_stop
        self.delta = delta
        self.R = R

    def calculate_half_life(self, spread_series: np.ndarray) -> float:
        """Compute Ornstein-Uhlenbeck mean-reversion half-life via AR(1) OLS.

        Spread model: Delta e_t = -kappa * (e_{t-1} - mu) * dt + sigma * dW
        Half-life: tau = ln(2) / kappa
        """
        if len(spread_series) < 15:
            return 999.0

        spread = spread_series[~np.isnan(spread_series)]
        if len(spread) < 15 or np.all(spread == spread[0]):
            return 999.0

        delta_spread = np.diff(spread)
        lagged_spread = spread[:-1]

        # Regress delta_spread on lagged_spread with constant
        X = np.vstack([lagged_spread, np.ones(len(lagged_spread))]).T
        try:
            # OLS solution
            theta, _, _, _ = np.linalg.lstsq(X, delta_spread, rcond=None)
            beta = theta[0]

            # If beta >= 0, spread is non-stationary / trending, no mean-reversion
            if beta >= 0 or np.isclose(beta, 0):
                return 999.0

            half_life = -np.log(2.0) / beta
            return float(np.clip(half_life, 0.1, 999.0))
        except Exception as exc:
            logger.debug(f"Error calculating half-life: {exc}")
            return 999.0

    def evaluate_pair(
        self,
        prices_y: pd.Series,
        prices_x: pd.Series,
        asset_y_name: str = "Y",
        asset_x_name: str = "X",
    ) -> PairSpreadState:
        """Run full Kalman Filter history across price series and generate trading signal."""
        # Align series by index
        combined = pd.DataFrame({"y": prices_y, "x": prices_x}).dropna()
        if len(combined) < 20:
            return PairSpreadState(
                asset_y=asset_y_name,
                asset_x=asset_x_name,
                alpha=0.0,
                beta=1.0,
                spread=0.0,
                spread_std=1.0,
                z_score=0.0,
                half_life_days=999.0,
                signal="NEUTRAL",
                confidence=0.0,
                cointegration_stable=False,
            )

        kf = DynamicKalmanFilter(delta=self.delta, R=self.R)

        alphas = []
        betas = []
        errors = []
        variances = []

        for _, row in combined.iterrows():
            a, b, err, var = kf.update(row["y"], row["x"])
            alphas.append(a)
            betas.append(b)
            errors.append(err)
            variances.append(var)

        errors_arr = np.array(errors)
        variances_arr = np.array(variances)

        curr_alpha = alphas[-1]
        curr_beta = betas[-1]
        curr_error = errors_arr[-1]
        curr_std = np.sqrt(max(variances_arr[-1], 1e-6))

        # Calculate instantaneous z-score and rolling empirical z-score
        recent_window = min(len(errors_arr), 60)
        recent_rolling_mean = np.mean(errors_arr[-recent_window:])
        recent_rolling_std = np.std(errors_arr[-recent_window:])

        if recent_rolling_std > 1e-5:
            empirical_z = (curr_error - recent_rolling_mean) / recent_rolling_std
        else:
            empirical_z = curr_error / curr_std

        # Compute Ornstein-Uhlenbeck half-life over recent errors
        half_life = self.calculate_half_life(errors_arr[-recent_window:])

        # Cointegration stability checks:
        # Half-life must be reasonable (0.2 to 30.0 trading days)
        cointegration_stable = 0.2 <= half_life <= 30.0

        # Signal Logic
        signal = "NEUTRAL"
        confidence = 0.50

        if abs(empirical_z) >= self.z_stop:
            signal = "STOP_LOSS"
            confidence = 0.90
        elif empirical_z <= -self.z_entry:
            # Spread is deeply depressed: Long Y, Short beta * X
            signal = "LONG_SPREAD"
            confidence = min(0.95, 0.50 + abs(empirical_z) * 0.15)
        elif empirical_z >= self.z_entry:
            # Spread is excessively stretched: Short Y, Long beta * X
            signal = "SHORT_SPREAD"
            confidence = min(0.95, 0.50 + abs(empirical_z) * 0.15)
        elif abs(empirical_z) <= self.z_exit:
            signal = "EXIT"
            confidence = 0.80

        # Penalize confidence if cointegration has broken down or half-life is infinite
        if not cointegration_stable:
            confidence = min(confidence, 0.45)
            if signal in ("LONG_SPREAD", "SHORT_SPREAD"):
                signal = "NEUTRAL"

        return PairSpreadState(
            asset_y=asset_y_name,
            asset_x=asset_x_name,
            alpha=round(curr_alpha, 4),
            beta=round(curr_beta, 4),
            spread=round(curr_error, 4),
            spread_std=round(curr_std, 4),
            z_score=round(empirical_z, 2),
            half_life_days=round(half_life, 1),
            signal=signal,
            confidence=round(confidence, 2),
            cointegration_stable=cointegration_stable,
        )

    def scan_universe(
        self, price_matrix: pd.DataFrame, pairs: Optional[List[Tuple[str, str]]] = None
    ) -> List[Dict[str, Any]]:
        """Scan candidate pairs against a DataFrame of adjusted close prices."""
        target_pairs = pairs or self.DEFAULT_PAIRS
        results = []

        for asset_y, asset_x in target_pairs:
            if asset_y in price_matrix.columns and asset_x in price_matrix.columns:
                state = self.evaluate_pair(
                    price_matrix[asset_y],
                    price_matrix[asset_x],
                    asset_y_name=asset_y,
                    asset_x_name=asset_x,
                )
                results.append(asdict(state))

        # Sort by opportunity: active trading signals with highest absolute z-scores first
        results.sort(
            key=lambda item: (
                1 if item["signal"] in ("LONG_SPREAD", "SHORT_SPREAD") else 0,
                abs(item["z_score"]),
            ),
            reverse=True,
        )
        return results


def run_kalman_statarb_scan(
    price_matrix: pd.DataFrame, pairs: Optional[List[Tuple[str, str]]] = None
) -> List[Dict[str, Any]]:
    """Functional top-level API for running Kalman statarb pairs analysis."""
    engine = KalmanStatArbEngine()
    return engine.scan_universe(price_matrix, pairs=pairs)
