import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from src.modeling import train_model, save_model, load_model, make_prediction

# Create a fixture for sample data
def sample_data():
    """Generates sample data for testing."""
    X_train = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100)
    })
    y_train = pd.Series(np.random.randint(0, 2, 100))
    X_test = pd.DataFrame({
        'feature1': np.random.rand(20),
        'feature2': np.random.rand(20)
    })
    y_test = pd.Series(np.random.randint(0, 2, 20))
    return X_train, y_train, X_test, y_test

def test_train_model():
    """Tests the train_model function."""
    X_train, y_train, X_test, y_test = sample_data()
    model, metrics, y_pred = train_model(X_train, y_train, X_test, y_test)

    assert isinstance(model, xgb.XGBClassifier)
    assert 'accuracy' in metrics
    assert isinstance(metrics['accuracy'], float)
    assert len(y_pred) == len(y_test)

def test_save_and_load_model(tmpdir):
    """Tests that a model can be saved and loaded correctly."""
    X_train, y_train, X_test, y_test = sample_data()
    model, _, _ = train_model(X_train, y_train, X_test, y_test)

    # Create a temporary file path
    filepath = os.path.join(str(tmpdir), "test_model.joblib")
    save_model(model, filepath)

    # Check if file exists
    assert os.path.exists(filepath)

    # Load the model
    loaded_model = load_model(filepath)
    assert isinstance(loaded_model, xgb.XGBClassifier)

    # Check if loaded model can predict
    predictions = loaded_model.predict(X_test)
    assert len(predictions) == len(y_test)

def test_make_prediction():
    """Tests the make_prediction function."""
    X_train, y_train, X_test, y_test = sample_data()
    model, _, _ = train_model(X_train, y_train, X_test, y_test)

    # Get a single sample for prediction
    latest_data = X_test.head(1)
    features = ['feature1', 'feature2']

    prediction, confidence = make_prediction(model, latest_data, features)

    assert prediction is not None
    assert confidence is not None
    assert len(prediction) == 1
    assert confidence.shape == (1, 2) # (n_samples, n_classes)
