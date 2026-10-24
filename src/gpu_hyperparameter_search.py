"""GPU-Accelerated Hyperparameter & Strategy Grid Search Engine.

Implements high-throughput Walk-Forward Optimization (WFO) across
trading model hyperparameters using XGBoost's GPU histogram tree method
(`tree_method="hist"`, `device="cuda"` with graceful `"cpu"` fallback).
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

logger = logging.getLogger("Sentilyze.GPUHyperparameterSearch")


@dataclass
class HyperparameterTrialResult:
    """Outcome of a single hyperparameter evaluation trial."""

    trial_id: int
    params: Dict[str, Any]
    oos_auc: float
    oos_accuracy: float
    oos_sharpe: float
    total_trades: int
    win_rate: float
    is_optimal: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        return asdict(self)


class GPUHyperparameterSearcher:
    """Orchestrates GPU-accelerated strategy and hyperparameter search."""

    DEFAULT_PARAM_SPACE: Dict[str, List[Any]] = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.02, 0.05, 0.10],
        "subsample": [0.75, 0.85, 1.0],
        "colsample_bytree": [0.75, 0.85, 1.0],
        "reg_alpha": [0.0, 0.5, 1.5],
        "reg_lambda": [0.5, 1.0, 2.0],
        "n_estimators": [50, 100],
    }

    def __init__(
        self,
        device: Optional[str] = None,
        min_auc: float = 0.53,
        min_sharpe: float = 1.0,
    ) -> None:
        """Initialize search engine with target device."""
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.min_auc = min_auc
        self.min_sharpe = min_sharpe

    def _build_xgb_model(self, trial_params: Dict[str, Any]) -> xgb.XGBClassifier:
        """Construct XGBoost model with GPU hist tree method."""
        params = trial_params.copy()
        params["tree_method"] = "hist"
        params["device"] = self.device
        params["eval_metric"] = "logloss"
        params["random_state"] = 42
        return xgb.XGBClassifier(**params)

    def evaluate_params_walk_forward(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        forward_returns: pd.Series,
        trial_params: Dict[str, Any],
        train_window: int = 120,
        test_window: int = 20,
    ) -> HyperparameterTrialResult:
        """Evaluate parameter configuration via rolling walk-forward validation."""
        total_samples = len(X)
        if total_samples < (train_window + test_window):
            return HyperparameterTrialResult(
                trial_id=0,
                params=trial_params,
                oos_auc=0.50,
                oos_accuracy=0.50,
                oos_sharpe=0.0,
                total_trades=0,
                win_rate=0.0,
                is_optimal=False,
            )

        oos_preds: List[float] = []
        oos_true: List[int] = []
        strat_returns: List[float] = []

        for start in range(0, total_samples - train_window, test_window):
            end_train = start + train_window
            end_test = min(end_train + test_window, total_samples)

            X_train, y_train = X.iloc[start:end_train], y.iloc[start:end_train]
            X_test, y_test = X.iloc[end_train:end_test], y.iloc[end_train:end_test]
            ret_test = forward_returns.iloc[end_train:end_test]

            model = self._build_xgb_model(trial_params)
            model.fit(X_train, y_train)

            probs = model.predict_proba(X_test)[:, 1]
            oos_preds.extend(probs)
            oos_true.extend(y_test.values)

            # Strategy returns: +1 when prob > 0.55, -1 when prob < 0.45, 0 otherwise
            signals = np.where(probs > 0.55, 1.0, np.where(probs < 0.45, -1.0, 0.0))
            strat_returns.extend(signals * ret_test.values)

        oos_true_arr = np.array(oos_true)
        oos_preds_arr = np.array(oos_preds)
        strat_ret_arr = np.array(strat_returns)

        # Metrics
        try:
            auc = (
                roc_auc_score(oos_true_arr, oos_preds_arr)
                if len(np.unique(oos_true_arr)) > 1
                else 0.50
            )
        except Exception:
            auc = 0.50

        acc = accuracy_score(oos_true_arr, (oos_preds_arr >= 0.50).astype(int))

        active_trades = strat_ret_arr[strat_ret_arr != 0.0]
        total_trades = len(active_trades)
        win_rate = float(np.mean(active_trades > 0)) if total_trades > 0 else 0.0

        mean_ret = float(np.mean(strat_ret_arr))
        std_ret = float(np.std(strat_ret_arr))
        sharpe = (mean_ret / max(std_ret, 1e-5)) * np.sqrt(252.0)

        is_optimal = auc >= self.min_auc and sharpe >= self.min_sharpe

        return HyperparameterTrialResult(
            trial_id=0,
            params=trial_params,
            oos_auc=round(float(auc), 4),
            oos_accuracy=round(float(acc), 4),
            oos_sharpe=round(float(np.clip(sharpe, -10.0, 10.0)), 2),
            total_trades=total_trades,
            win_rate=round(win_rate, 4),
            is_optimal=is_optimal,
        )

    def search_grid(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        forward_returns: pd.Series,
        n_trials: int = 20,
        param_space: Optional[Dict[str, List[Any]]] = None,
        seed: int = 42,
    ) -> List[HyperparameterTrialResult]:
        """Perform randomized GPU grid search across hyperparameter combinations."""
        space = param_space or self.DEFAULT_PARAM_SPACE
        random.seed(seed)

        # Generate unique trial parameter combinations
        keys = list(space.keys())
        trials: List[HyperparameterTrialResult] = []

        for trial_num in range(1, n_trials + 1):
            sample_params = {k: random.choice(space[k]) for k in keys}
            res = self.evaluate_params_walk_forward(
                X, y, forward_returns, sample_params
            )
            res.trial_id = trial_num
            trials.append(res)

        # Sort trials by OOS Sharpe ratio descending
        trials.sort(key=lambda t: (t.oos_sharpe, t.oos_auc), reverse=True)
        if trials:
            trials[0].is_optimal = True

        return trials

    def export_optimal_hyperparameters(
        self, trials: List[HyperparameterTrialResult], output_path: str
    ) -> Dict[str, Any]:
        """Export best performing hyperparameter configuration to JSON."""
        if not trials:
            raise ValueError("No trials to export.")

        best = trials[0]
        data = {
            "best_params": best.params,
            "best_oos_sharpe": best.oos_sharpe,
            "best_oos_auc": best.oos_auc,
            "best_oos_accuracy": best.oos_accuracy,
            "total_evaluated_trials": len(trials),
            "top_3_trials": [t.to_dict() for t in trials[:3]],
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return data
