import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.utils import get_logger
from typing import Tuple, Dict, Any, List

logger = get_logger(__name__)


def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[RandomForestClassifier, Dict[str, Any], pd.Series]:
    """
    Train the RandomForest model.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.
        X_test (pd.DataFrame): The testing features.
        y_test (pd.Series): The testing target.

    Returns:
        Tuple[RandomForestClassifier, Dict[str, Any], pd.Series]: A tuple containing the trained model, a dictionary of metrics, and the predictions on the test set.
    """
    logger.info("Training RandomForest model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    metrics = {"accuracy": accuracy, "classification_report": report}
    logger.info(f"Model training complete. Accuracy: {accuracy:.4f}")
    return model, metrics, y_pred


def save_model(model: RandomForestClassifier, filepath: str) -> None:
    """
    Save the trained model to a file.

    Args:
        model (RandomForestClassifier): The trained model to save.
        filepath (str): The path to save the model to.
    """
    logger.info(f"Saving model to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: str) -> RandomForestClassifier:
    """
    Load a trained model from a file.

    Args:
        filepath (str): The path to load the model from.

    Returns:
        RandomForestClassifier: The loaded model.
    """
    logger.info(f"Loading model from {filepath}...")
    return joblib.load(filepath)


def make_prediction(model: RandomForestClassifier, latest_data: pd.DataFrame, features: List[str]) -> Tuple[Any, Any]:
    """
    Make a prediction for the next day.

    Args:
        model (RandomForestClassifier): The trained model.
        latest_data (pd.DataFrame): The latest data to make a prediction on.
        features (List[str]): The list of features to use for the prediction.

    Returns:
        Tuple[Any, Any]: A tuple containing the prediction and the confidence score.
    """
    logger.info("Making prediction on latest data...")
    prediction = model.predict(latest_data[features])
    confidence = model.predict_proba(latest_data[features])
    return prediction, confidence
