import os
import pandas as pd
import numpy as np
import xgboost as xgb
from src.modeling import (
    train_model,
    save_model,
    load_model,
    get_prediction_on_latest_data,
)


# Create a fixture for sample data
def sample_data():
    """Generates sample data for testing. Needs at least 520 rows for WFO (train_window=500 + test_window=20)."""
    # Deterministic generation for rigorous testing
    X = pd.DataFrame(
        {"feature1": np.linspace(-1, 1, 550), "feature2": np.linspace(1, -1, 550)}
    )
    # Create a clear pattern: if feature1 > 0, target is 1
    y = pd.Series((X["feature1"] > 0).astype(int))
    return X, y


def test_train_model():
    """Tests the train_model function."""
    X, y = sample_data()
    # Use smaller windows to make the test run faster, or just let it use defaults since we have 550 rows
    model, metrics, oos_preds = train_model(X, y, train_window=500, test_window=20)

    assert isinstance(model, xgb.XGBClassifier)
    assert "accuracy" in metrics
    assert isinstance(metrics["accuracy"], float)
    assert len(oos_preds) > 0


def test_save_and_load_model(tmpdir):
    """Tests that a model can be saved and loaded correctly."""
    X, y = sample_data()
    model, _, _ = train_model(X, y)

    # Create a temporary file path
    filepath = os.path.join(str(tmpdir), "test_model.json")
    save_model(model, filepath)

    # Check if file exists
    assert os.path.exists(filepath)

    # Load the model
    loaded_model = load_model(filepath)
    assert isinstance(loaded_model, xgb.XGBClassifier)

    # Check if loaded model can predict
    X_test = X.tail(20)
    predictions = loaded_model.predict(X_test)
    assert len(predictions) == 20


def test_make_prediction():
    """Tests the get_prediction_on_latest_data function."""
    X, y = sample_data()
    model, _, _ = train_model(X, y)

    # Get a single sample for prediction
    latest_data = X.tail(1)
    features = ["feature1", "feature2"]

    prediction, confidence = get_prediction_on_latest_data(model, latest_data, features)

    assert prediction is not None
    assert confidence is not None
    assert len(prediction) == 1
    assert confidence.shape == (1, 2)  # (n_samples, n_classes)
