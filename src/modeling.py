import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.utils import get_logger, sanitize_filename, safe_path_join
from src.config import XGB_MODEL_PARAMS
from typing import Tuple, Dict, Any, List

logger = get_logger(__name__)


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    train_window: int = 500,
    test_window: int = 20,
    tune_hyperparameters: bool = False,
) -> Tuple[xgb.XGBClassifier, Dict[str, Any], pd.Series]:
    """
    Train the XGBoost model using Walk-Forward Optimization (WFO) alongside a Logistic Regression baseline.
    Strictly trains on past data (e.g., 500 days) to predict the immediate future (e.g., 20 days), rolling forward.

    Args:
        X (pd.DataFrame): The full features DataFrame (chronologically ordered).
        y (pd.Series): The full target Series.
        train_window (int): Number of days for the rolling training window.
        test_window (int): Number of days to predict iteratively.
        tune_hyperparameters (bool): If True, runs a randomized hyperparameter search.

    Returns:
        Tuple[xgb.XGBClassifier, Dict[str, Any], pd.Series]:
            - The final trained model (trained on all recent data for live use).
            - A dictionary of metrics across the entire WFO out-of-sample period (including baseline comparison).
            - A Series containing all out-of-sample predictions.
    """
    logger.info(
        f"Starting Walk-Forward Optimization (Train: {train_window}d, Test: {test_window}d)..."
    )

    # Store out-of-sample (OOS) predictions for XGBoost
    oos_predictions = []
    oos_true = []
    oos_indices = []

    # Store out-of-sample predictions for Logistic Regression baseline
    baseline_oos_predictions = []

    model_params = XGB_MODEL_PARAMS.copy()

    total_samples = len(X)

    if total_samples <= train_window:
        logger.error(
            f"Not enough data for WFO. Got {total_samples} rows, need > {train_window}"
        )
        raise ValueError("Insufficient data for requested training window.")

    # Rolling WFO Loop
    for start_idx in range(0, total_samples - train_window, test_window):
        end_train = start_idx + train_window
        end_test = min(end_train + test_window, total_samples)

        # Split Data
        X_train_fold = X.iloc[start_idx:end_train]
        y_train_fold = y.iloc[start_idx:end_train]
        X_test_fold = X.iloc[end_train:end_test]
        y_test_fold = y.iloc[end_train:end_test]

        # 1. Train Fold XGBoost Model
        fold_model = xgb.XGBClassifier(**model_params)
        fold_model.fit(X_train_fold, y_train_fold)
        fold_probs = fold_model.predict_proba(X_test_fold)[:, 1]

        # 2. Train Fold Baseline Logistic Regression Model (with scaling)
        try:
            baseline_model = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=500, random_state=42)
            )
            baseline_model.fit(X_train_fold, y_train_fold)
            baseline_probs = baseline_model.predict_proba(X_test_fold)[:, 1]
        except Exception:
            baseline_probs = np.full(len(y_test_fold), 0.5)

        oos_predictions.extend(fold_probs)
        baseline_oos_predictions.extend(baseline_probs)
        oos_true.extend(y_test_fold.values)
        oos_indices.extend(y_test_fold.index)

    final_oos_preds_series = pd.Series(oos_predictions, index=oos_indices)

    # Calculate XGBoost global metrics
    binary_preds = [1 if p > 0.5 else 0 for p in oos_predictions]
    accuracy = float(accuracy_score(oos_true, binary_preds))
    precision = float(precision_score(oos_true, binary_preds, zero_division=0))
    recall = float(recall_score(oos_true, binary_preds, zero_division=0))
    f1 = float(f1_score(oos_true, binary_preds, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(oos_true, oos_predictions))
    except Exception:
        roc_auc = 0.5

    report = classification_report(oos_true, binary_preds)

    # Calculate Logistic Regression baseline metrics
    baseline_binary = [1 if p > 0.5 else 0 for p in baseline_oos_predictions]
    baseline_accuracy = float(accuracy_score(oos_true, baseline_binary))
    try:
        baseline_roc_auc = float(roc_auc_score(oos_true, baseline_oos_predictions))
    except Exception:
        baseline_roc_auc = 0.5

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "baseline_logistic_accuracy": baseline_accuracy,
        "baseline_logistic_roc_auc": baseline_roc_auc,
        "classification_report": report,
        "best_params": model_params,
        "total_test_samples": len(oos_true),
    }

    logger.info(f"WFO complete across {len(oos_true)} out-of-sample days.")
    logger.info(f"XGBoost Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
    logger.info(
        f"Baseline Logistic Regression Accuracy: {baseline_accuracy:.4f}, ROC-AUC: {baseline_roc_auc:.4f}"
    )

    # Train final production model on the most recent data window
    logger.info("Training final production model on the most recent data window...")
    final_start = max(0, total_samples - train_window)
    X_final = X.iloc[final_start:]
    y_final = y.iloc[final_start:]

    final_model = xgb.XGBClassifier(**model_params)
    final_model.fit(X_final, y_final)

    return final_model, metrics, final_oos_preds_series


def save_model(model: xgb.XGBClassifier, filepath: str) -> None:
    """
    Save the trained model to a file using XGBoost's native format.
    This is safer than joblib/pickle and prevents arbitrary code execution.

    Args:
        model (xgb.XGBClassifier): The trained model to save.
        filepath (str): The path to save the model to.
    """
    dir_name = os.path.dirname(filepath) or "models"
    base_name = os.path.basename(filepath)
    if not base_name.endswith(".json"):
        base_name = base_name.replace(".joblib", ".json")
    clean_base = sanitize_filename(base_name)
    safe_fp = safe_path_join(dir_name, clean_base)

    logger.info(f"Saving model in native format to {safe_fp}...")
    os.makedirs(os.path.dirname(safe_fp), exist_ok=True)
    model.save_model(safe_fp)


def load_model(filepath: str) -> xgb.XGBClassifier:
    """
    Load a trained model from a file using XGBoost's native format.

    Args:
        filepath (str): The path to load the model from.

    Returns:
        xgb.XGBClassifier: The loaded model.
    """
    dir_name = os.path.dirname(filepath) or "models"
    base_name = os.path.basename(filepath)
    clean_base = sanitize_filename(base_name)
    safe_fp = safe_path_join(dir_name, clean_base)

    if not os.path.exists(safe_fp):
        json_base = clean_base.replace(".joblib", ".json")
        json_path = safe_path_join(dir_name, json_base)
        if os.path.exists(json_path):
            safe_fp = json_path

    logger.info(f"Loading model in safe native format from {safe_fp}...")
    model = xgb.XGBClassifier()
    model.load_model(safe_fp)
    return model


def get_prediction_on_latest_data(
    model: xgb.XGBClassifier, latest_data: pd.DataFrame, features: List[str]
) -> Tuple[Any, Any]:
    """
    Gets a prediction from the model for the latest available data point.

    Args:
        model (xgb.XGBClassifier): The trained model.
        latest_data (pd.DataFrame): The latest data to make a prediction on.
        features (List[str]): The list of features to use for the prediction.

    Returns:
        Tuple[Any, Any]: A tuple containing the prediction and the confidence score.
    """
    logger.info("Getting prediction for latest data...")
    prediction = model.predict(latest_data[features])
    confidence = model.predict_proba(latest_data[features])
    return prediction, confidence
