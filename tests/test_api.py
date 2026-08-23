import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from api import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "supported_tickers" in data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_predict_unsupported_ticker():
    response = client.get("/predict?ticker=NONEXISTENT")
    assert response.status_code == 404


@patch("api.preprocess_data")
@patch("api.load_model")
@patch("api.shap.TreeExplainer")
def test_predict_valid_ticker(mock_explainer_cls, mock_load_model, mock_preprocess):
    # Mock data
    dates = pd.date_range("2025-01-01", periods=5)
    from src.config import FEATURES
    mock_data = {feat: [1.0] * 5 for feat in FEATURES}
    features_df = pd.DataFrame(mock_data, index=dates)
    price_hist = pd.DataFrame(
        {"Close": [110.0] * 5, "rsi": [55.0] * 5, "sma200": [90.0] * 5},
        index=dates,
    )
    news_df = pd.DataFrame()

    mock_preprocess.return_value = (features_df, price_hist, news_df)

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.15, 0.85]])
    mock_load_model.return_value = mock_model

    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = np.array([[0.1] * len(FEATURES)])
    mock_explainer_cls.return_value = mock_explainer

    # Simulate existing model file
    with patch("os.path.exists", return_value=True):
        response = client.get("/predict?ticker=NVDA")

    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "NVDA"
    assert data["signal"] == "BUY"
    assert data["confidence"] == 0.85
    assert len(data["top_features"]) == 5
