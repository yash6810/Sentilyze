import pytest
import os
import torch
import numpy as np
import pandas as pd
from src.deep_learning_model import (
    DLinearTCNModel,
    create_sliding_window_tensors,
    train_dlinear_tcn_model,
    predict_momentum_probability,
    save_dlinear_model,
    load_dlinear_model,
)


def test_dlinear_tcn_forward_shape():
    """Verify that DLinear-TCN forward pass produces correct [batch, num_classes] output."""
    batch_size = 8
    seq_len = 10
    num_features = 25
    x = torch.randn(batch_size, seq_len, num_features)

    model = DLinearTCNModel(seq_len=seq_len, num_features=num_features)
    out = model(x)
    assert out.shape == (batch_size, 2)


def test_create_sliding_window_tensors():
    """Verify tensor generation from DataFrame."""
    n = 30
    cols = [f"feat_{i}" for i in range(5)]
    data = {c: np.random.randn(n) for c in cols}
    data["target"] = np.random.randint(0, 2, n)
    df = pd.DataFrame(data)

    X_t, y_t = create_sliding_window_tensors(df, cols, seq_len=10, target_col="target")
    assert X_t.shape == (21, 10, 5)
    assert y_t.shape == (21,)


def test_train_and_save_dlinear_model(tmp_path):
    """Verify CPU training loop and safe model checkpoint saving/loading."""
    X_train = torch.randn(64, 10, 25)
    y_train = torch.randint(0, 2, (64,))

    model, history = train_dlinear_tcn_model(X_train, y_train, epochs=3, batch_size=32)
    assert len(history["train_loss"]) == 3
    assert history["train_loss"][-1] > 0.0

    # Prediction
    pred = predict_momentum_probability(model, X_train[0])
    assert "bullish_probability" in pred
    assert "signal" in pred
    assert pred["signal"] in ["BUY", "SELL", "NEUTRAL"]

    # Save & Load
    save_file = str(tmp_path / "test_dlinear.pt")
    save_dlinear_model(model, save_file)
    assert os.path.exists(save_file)

    loaded_model = load_dlinear_model(save_file)
    loaded_pred = predict_momentum_probability(loaded_model, X_train[0])
    assert loaded_pred["bullish_probability"] == pred["bullish_probability"]
