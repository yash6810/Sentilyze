"""
Regime-Conditioned Mixture of Experts (MoE) Architecture.

Trains 3 specialized XGBoost sub-models (Bull Momentum, Bear Volatility, Sideways Mean-Reversion)
with a dynamic Softmax Gating Network routing predictions based on macroeconomic and VIX states.
Reference: Jacobs & Jordan (1991), Neural Computation.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RegimeMixtureOfExperts:
    """
    3-Expert Gated Mixture of Experts classifier for financial regimes.
    """

    def __init__(self, base_params: Optional[Dict[str, Any]] = None):
        self.base_params = base_params or {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        self.expert_bull = xgb.XGBClassifier(**self.base_params)
        self.expert_bear = xgb.XGBClassifier(**self.base_params)
        self.expert_chop = xgb.XGBClassifier(**self.base_params)
        self.gating_model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.05,
            objective="multi:softprob",
            num_class=3,
        )

    def _determine_regimes(self, X: pd.DataFrame) -> np.ndarray:
        """
        Labels historical rows into 3 latent market regimes:
        0 = Bull Momentum (Price > SMA50 and RSI > 50)
        1 = Bear Volatility (Price < SMA50 or High Vol)
        2 = Chop / Mean Reversion (Sideways)
        """
        n = len(X)
        regimes = np.full(n, 2, dtype=int)  # Default chop

        # Check standard indicator columns if available
        if "RSI" in X.columns and "SMA_50" in X.columns and "Close" in X.columns:
            bull_cond = (X["Close"] > X["SMA_50"]) & (X["RSI"] >= 50)
            bear_cond = (X["Close"] < X["SMA_50"]) | (X["RSI"] <= 40)
            regimes[bull_cond.values] = 0
            regimes[bear_cond.values] = 1
        elif "RSI" in X.columns:
            regimes[X["RSI"] > 55] = 0
            regimes[X["RSI"] < 45] = 1

        return regimes

    def fit(self, X: pd.DataFrame, y: pd.Series):
        regimes = self._determine_regimes(X)
        self.classes_ = np.unique(regimes)

        # Train Bull Expert
        bull_idx = regimes == 0
        if np.sum(bull_idx) > 20 and len(np.unique(y[bull_idx])) > 1:
            self.expert_bull.fit(X[bull_idx], y[bull_idx])
        else:
            self.expert_bull.fit(X, y)

        # Train Bear Expert
        bear_idx = regimes == 1
        if np.sum(bear_idx) > 20 and len(np.unique(y[bear_idx])) > 1:
            self.expert_bear.fit(X[bear_idx], y[bear_idx])
        else:
            self.expert_bear.fit(X, y)

        # Train Chop Expert
        chop_idx = regimes == 2
        if np.sum(chop_idx) > 20 and len(np.unique(y[chop_idx])) > 1:
            self.expert_chop.fit(X[chop_idx], y[chop_idx])
        else:
            self.expert_chop.fit(X, y)

        # Train Gating Router
        if len(self.classes_) > 1:
            self.gating_model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=2,
                learning_rate=0.05,
                eval_metric="mlogloss",
                random_state=42,
            )
            self.gating_model.fit(X, regimes)
        else:
            self.gating_model = None

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        if self.gating_model is not None and hasattr(
            self.gating_model, "predict_proba"
        ):
            raw_gate = self.gating_model.predict_proba(X)
            gate_weights = np.zeros((n, 3))
            for idx, c in enumerate(self.gating_model.classes_):
                if c < 3:
                    gate_weights[:, int(c)] = raw_gate[:, idx]
        else:
            det_regimes = self._determine_regimes(X)
            gate_weights = np.zeros((n, 3))
            for i in range(3):
                gate_weights[det_regimes == i, i] = 1.0

        p_bull = self.expert_bull.predict_proba(X)[:, 1]
        p_bear = self.expert_bear.predict_proba(X)[:, 1]
        p_chop = self.expert_chop.predict_proba(X)[:, 1]

        # Gated mixture probability
        p_up = (
            gate_weights[:, 0] * p_bull
            + gate_weights[:, 1] * p_bear
            + gate_weights[:, 2] * p_chop
        )
        p_up = np.clip(p_up, 0.001, 0.999)
        p_down = 1.0 - p_up
        return np.column_stack([p_down, p_up])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)
