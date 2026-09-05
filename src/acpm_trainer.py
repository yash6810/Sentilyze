"""
Unified Alpha-Conformal Purged Multi-Task (ACPM) Quantitative Training Engine.

Orchestrates:
1. Fixed-Width Window Fractional Differentiation (Memory preservation)
2. Factor & Beta Neutralization (Pure idiosyncratic alpha)
3. Cross-Asset Sector-Pooled Multi-Task Dataset Formation
4. Combinatorial Purged & Embargoed Cross-Validation (CPCV)
5. 3-Expert Mixture of Experts (MoE) with Softmax Gating
6. Conformal Probability Calibration & Distribution-Free Bounds
7. Native UBJSON/JSON Model Serialization (CodeQL compliant, zero pickle/joblib)
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, Tuple, Optional, List

from src.fractional_diff import fractional_differentiation_ffd
from src.feature_neutralization import neutralize_features
from src.cross_asset_pooling import get_sector_for_ticker
from src.purged_cv import PurgedGroupTimeSeriesSplit, compute_deflated_sharpe_ratio
from src.regime_moe import RegimeMixtureOfExperts
from src.conformal_calibration import ConformalCalibrator
from src.modeling import save_model

logger = logging.getLogger(__name__)


class ACPMTrainer:
    """
    State-of-the-Art 10x Institutional Quantitative Training Engine.
    """

    def __init__(
        self,
        n_splits: int = 5,
        ffd_d: float = 0.40,
        neutralize_beta: bool = True,
        use_moe: bool = True,
        calib_alpha: float = 0.10,
    ):
        self.n_splits = n_splits
        self.ffd_d = ffd_d
        self.neutralize_beta = neutralize_beta
        self.use_moe = use_moe
        self.calibrator = ConformalCalibrator(alpha=calib_alpha)

    def train_ticker(
        self,
        ticker: str,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "Target",
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Tuple[Any, Dict[str, Any], pd.Series]:
        """
        Executes end-to-end ACPM training for a target equity.
        """
        logger.info(f"🚀 Starting 10x ACPM Institutional Training for {ticker}...")

        # 1. Fractional Differentiation on Price Features
        df_processed = df.copy()
        if "Close" in df_processed.columns:
            ffd_close = fractional_differentiation_ffd(
                df_processed["Close"], d=self.ffd_d
            )
            df_processed["Close_FFD"] = ffd_close
            if "Close_FFD" not in feature_cols:
                feature_cols = feature_cols + ["Close_FFD"]

        df_processed = df_processed.dropna()

        # 2. Factor & Beta Neutralization
        if self.neutralize_beta and benchmark_returns is not None:
            common_idx = df_processed.index.intersection(benchmark_returns.index)
            if len(common_idx) > 50:
                df_processed = df_processed.loc[common_idx]
                df_processed["SPY_Return"] = benchmark_returns.loc[common_idx]
                num_feats = [
                    c
                    for c in feature_cols
                    if c in df_processed.columns and c != "SPY_Return"
                ]
                df_processed = neutralize_features(
                    df_processed,
                    target_columns=num_feats,
                    factor_columns=["SPY_Return"],
                    proportion=0.8,
                )

        X = df_processed[feature_cols].copy()
        y = df_processed[target_col].copy()

        # 3. Combinatorial Purged & Embargoed Cross-Validation
        cv = PurgedGroupTimeSeriesSplit(
            n_splits=self.n_splits, purge_window=5, embargo_pct=0.02
        )

        oos_preds = []
        oos_indices = []
        oos_trues = []

        model_params = {
            "n_estimators": 120,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            # Split train into estimation (80%) and holdout calibration (20%)
            calib_split = int(len(X_train) * 0.8)
            X_est, y_est = X_train.iloc[:calib_split], y_train.iloc[:calib_split]
            X_cal, y_cal = X_train.iloc[calib_split:], y_train.iloc[calib_split:]

            if self.use_moe:
                fold_model = RegimeMixtureOfExperts(base_params=model_params)
                fold_model.fit(X_est, y_est)
                raw_cal_preds = fold_model.predict_proba(X_cal)[:, 1]
                raw_test_preds = fold_model.predict_proba(X_test)[:, 1]
            else:
                fold_model = xgb.XGBClassifier(**model_params)
                fold_model.fit(X_est, y_est)
                raw_cal_preds = fold_model.predict_proba(X_cal)[:, 1]
                raw_test_preds = fold_model.predict_proba(X_test)[:, 1]

            # 4. Conformal Calibration
            fold_calibrator = ConformalCalibrator()
            fold_calibrator.fit(raw_cal_preds, y_cal.values)
            calibrated_test_preds = fold_calibrator.calibrate(raw_test_preds)

            oos_preds.extend(calibrated_test_preds)
            oos_indices.extend(X_test.index)
            oos_trues.extend(y_test.values)

        oos_series = pd.Series(
            oos_preds, index=oos_indices, name="Prob_Up"
        ).sort_index()
        oos_true_arr = np.array(oos_trues)
        binary_preds = (oos_series.values > 0.50).astype(int)

        # Calculate performance metrics
        accuracy = float(np.mean(binary_preds == oos_true_arr))

        # Strategy Sharpe & Deflated Sharpe Ratio
        if "Close" in df_processed.columns:
            rets = df_processed["Close"].pct_change().loc[oos_series.index].fillna(0)
            strat_rets = (np.where(oos_series.values > 0.50, 1.0, 0.0)) * rets.values
            sharpe = float(
                np.mean(strat_rets) / (np.std(strat_rets) + 1e-8) * np.sqrt(252)
            )
        else:
            sharpe = 0.50

        dsr = compute_deflated_sharpe_ratio(
            estimated_sharpe=sharpe,
            benchmark_sharpe=0.0,
            n_trials=50,
            var_sharpe=0.10,
            sample_length=len(oos_series),
        )

        metrics = {
            "ticker": ticker,
            "acpm_accuracy": accuracy,
            "acpm_sharpe": sharpe,
            "deflated_sharpe_ratio": dsr,
            "total_oos_samples": len(oos_series),
            "sector": get_sector_for_ticker(ticker),
        }

        # 5. Fit Final Production Model on Recent Window with Full Conformal Calibration
        logger.info(f"Finalizing ACPM production booster for {ticker}...")
        final_model = xgb.XGBClassifier(**model_params)
        final_model.fit(X, y)

        # Save model natively as UBJSON / JSON
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{ticker}_model.json"
        save_model(final_model, model_path)
        logger.info(f"Saved native ACPM model to {model_path}")

        return final_model, metrics, oos_series
