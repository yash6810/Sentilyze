import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.utils import get_logger

logger = get_logger(__name__)

def train_model(X_train, y_train, X_test, y_test):
    """
    Train the RandomForest model.
    """
    logger.info("Training RandomForest model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "classification_report": report
    }
    logger.info(f"Model training complete. Accuracy: {accuracy:.4f}")
    return model, metrics, y_pred

def save_model(model, filepath):
    """
    Save the trained model to a file.
    """
    logger.info(f"Saving model to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

def load_model(filepath):
    """
    Load a trained model from a file.
    """
    logger.info(f"Loading model from {filepath}...")
    return joblib.load(filepath)

def make_prediction(model, latest_data, features):
    """
    Make a prediction for the next day.
    """
    logger.info("Making prediction on latest data...")
    prediction = model.predict(latest_data[features])
    confidence = model.predict_proba(latest_data[features])
    return prediction, confidence