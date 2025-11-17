import joblib
import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from src.utils import get_logger
from typing import Tuple, Dict, Any, List
from scipy.stats import randint, uniform

logger = get_logger(__name__)

def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[xgb.XGBClassifier, Dict[str, Any], pd.Series]:
    """
    Train the XGBoost model with hyperparameter tuning using RandomizedSearchCV.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.
        X_test (pd.DataFrame): The testing features.
        y_test (pd.Series): The testing target.

    Returns:
        Tuple[xgb.XGBClassifier, Dict[str, Any], pd.Series]: A tuple containing the trained model, a dictionary of metrics, and the predictions on the test set.
    """
    logger.info("Training XGBoost model with hyperparameter tuning using RandomizedSearchCV...")

    # Define the parameter distribution
    param_dist = {
        'n_estimators': randint(100, 500),
        'learning_rate': uniform(0.01, 0.1),
        'max_depth': randint(3, 10),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3),
    }

    # Use TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        param_distributions=param_dist,
        n_iter=50,  # Number of parameter settings that are sampled
        cv=tscv,
        n_jobs=-1,
        verbose=2,
        random_state=42,
        scoring='accuracy' # Use accuracy as the scoring metric
    )

    # Fit the random search to the data
    random_search.fit(X_train, y_train)

    # Get the best model
    model = random_search.best_estimator_

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    metrics = {"accuracy": accuracy, "classification_report": report, "best_params": random_search.best_params_}
    logger.info(f"Best parameters found: {random_search.best_params_}")
    logger.info(f"Model training complete. Accuracy: {accuracy:.4f}")
    return model, metrics, y_pred


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
    logger.info(f"Loading model from {filepath}...")
    return joblib.load(filepath)


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