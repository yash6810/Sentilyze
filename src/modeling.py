import joblib
import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from src.utils import get_logger
from typing import Tuple, Dict, Any, List
import re

logger = get_logger(__name__)

def train_model(X: pd.DataFrame, y: pd.Series, train_window: int = 500, test_window: int = 20) -> Tuple[xgb.XGBClassifier, Dict[str, Any], pd.Series]:
    """
    Train the XGBoost model using Walk-Forward Optimization (WFO).
    This prevents look-ahead bias by strictly training on past data (e.g., 500 days)
    to predict the immediate future (e.g., 20 days), rolling forward iteratively.

    Args:
        X (pd.DataFrame): The full features DataFrame (chronologically ordered).
        y (pd.Series): The full target Series.
        train_window (int): Number of days for the rolling training window.
        test_window (int): Number of days to predict iteratively.

    Returns:
        Tuple[xgb.XGBClassifier, Dict[str, Any], pd.Series]: 
            - The final trained model (trained on all recent data for live use).
            - A dictionary of metrics across the entire WFO out-of-sample period.
            - A Series containing all out-of-sample predictions.
    """
    logger.info(f"Starting Walk-Forward Optimization (Train: {train_window}d, Test: {test_window}d)...")
    
    # Store out-of-sample (OOS) predictions
    oos_predictions = []
    oos_true = []
    oos_indices = []
    
    # Fixed parameters for speed and stability (from the paper)
    model_params = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 4,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    total_samples = len(X)
    
    if total_samples <= train_window:
        logger.error(f"Not enough data for WFO. Got {total_samples} rows, need > {train_window}")
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
        
        # Train fold model
        fold_model = xgb.XGBClassifier(**model_params)
        fold_model.fit(X_train_fold, y_train_fold)
        
        # Predict on out-of-sample fold using predict_proba to capture probabilities
        # We store the probability of the positive class (1)
        fold_probs = fold_model.predict_proba(X_test_fold)[:, 1]
        
        oos_predictions.extend(fold_probs)
        oos_true.extend(y_test_fold.values)
        oos_indices.extend(y_test_fold.index)
        
        logger.debug(f"WFO Fold Step: Trained on {len(X_train_fold)} rows, tested on {len(X_test_fold)} rows.")

    # Combine all OOS predictions into a Series
    # These are probabilities, not hard 0/1 classes, so the regime filter can use them
    final_oos_preds_series = pd.Series(oos_predictions, index=oos_indices)
    
    # Calculate global metrics (using 0.5 threshold as a baseline for accuracy)
    binary_preds = [1 if p > 0.5 else 0 for p in oos_predictions]
    accuracy = accuracy_score(oos_true, binary_preds)
    report = classification_report(oos_true, binary_preds)
    
    metrics = {
        "accuracy": accuracy, 
        "classification_report": report, 
        "best_params": model_params  # Hard-coded based on paper
    }
    
    logger.info(f"WFO complete across {len(oos_true)} out-of-sample days.")
    logger.info(f"OOS Accuracy: {accuracy:.4f}")
    
    # Train one final model on the MOST RECENT window to save for live future predictions
    logger.info("Training final production model on the most recent data window...")
    final_start = max(0, total_samples - train_window)
    X_final = X.iloc[final_start:]
    y_final = y.iloc[final_start:]
    
    final_model = xgb.XGBClassifier(**model_params)
    final_model.fit(X_final, y_final)

    return final_model, metrics, final_oos_preds_series


def _validate_model_filepath(filepath: str) -> str:
    """
    Validate and normalize the model file path to prevent loading
    arbitrary files based on untrusted input.

    Ensures that:
        - The resolved path is inside the expected ``models`` directory.
        - The file name matches the expected ``*_model.joblib`` pattern
          and contains only safe characters.

    Args:
        filepath (str): The original file path.

    Returns:
        str: A validated absolute path to the model file.

    Raises:
        ValueError: If the path is outside the ``models`` directory
            or does not conform to the expected naming pattern.
    """
    base_dir = os.path.abspath("models")
    abs_path = os.path.abspath(filepath)

    # Ensure the model file resides within the models directory
    if not (abs_path == base_dir or abs_path.startswith(base_dir + os.sep)):
        logger.error(f"Attempt to access model outside of models directory: {abs_path}")
        raise ValueError("Invalid model path.")

    filename = os.path.basename(abs_path)
    # Allow tickers comprised of safe characters only and enforce suffix
    if not re.fullmatch(r"[A-Za-z0-9_\-]+_model\.joblib", filename):
        logger.error(f"Invalid model filename: {filename}")
        raise ValueError("Invalid model filename.")

    return abs_path


def save_model(model: xgb.XGBClassifier, filepath: str) -> None:
    """
    Save the trained model to a file.

    Args:
        model (xgb.XGBClassifier): The trained model to save.
        filepath (str): The path to save the model to.
    """
    logger.info(f"Saving model to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: str) -> xgb.XGBClassifier:
    """
    Load a trained model from a file.

    Args:
        filepath (str): The path to load the model from.

    Returns:
        xgb.XGBClassifier: The loaded model.
    """
    safe_path = _validate_model_filepath(filepath)
    logger.info(f"Loading model from {safe_path}...")
    return joblib.load(safe_path)


def get_prediction_on_latest_data(model: xgb.XGBClassifier, latest_data: pd.DataFrame, features: List[str]) -> Tuple[Any, Any]:
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