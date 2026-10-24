"""Unit Tests for GPU-Accelerated Alpha Miner and Tensor Operators.

Verifies:
1. PyTorch tensor alpha operators (ts_rank, decay_linear, delta, cs_rank).
2. Market tensor loading and forward returns alignment.
3. Vectorized RankIC, ICIR, and Sharpe ratio evaluation.
4. Expression string evaluator.
5. Automated alpha search pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.gpu_alpha_miner import (
    DiscoveredAlpha,
    GPUAlphaMiner,
    TensorAlphaOperators,
)


@pytest.fixture
def sample_market_dict() -> dict[str, pd.DataFrame]:
    """Generate 60-day synthetic OHLCV data for 3 assets."""
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    tickers = ["NVDA", "AAPL", "MSFT"]
    data = {}

    for t in tickers:
        base = 100.0 + np.cumsum(np.random.randn(60) * 1.5)
        df = pd.DataFrame(
            {
                "open": base + np.random.randn(60) * 0.2,
                "high": base + 1.0,
                "low": base - 1.0,
                "close": base,
                "volume": np.random.randint(1000000, 5000000, size=60),
                "vwap": base + 0.1,
            },
            index=dates,
        )
        data[t] = df

    return data


def test_tensor_alpha_operators():
    """Verify PyTorch tensor operators on synthetic 2D tensor [T, N]."""
    x = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])

    # Delay
    delayed = TensorAlphaOperators.delay(x, 1)
    assert delayed[0, 0] == 0.0
    assert delayed[1, 0] == 1.0
    assert delayed[4, 1] == 40.0

    # Delta
    diff = TensorAlphaOperators.delta(x, 1)
    assert diff[1, 0] == 1.0
    assert diff[2, 1] == 10.0

    # Mean
    mean_val = TensorAlphaOperators.ts_mean(x, 3)
    assert mean_val[2, 0] == 2.0  # mean of [1, 2, 3] = 2

    # CS Rank
    cs = TensorAlphaOperators.cs_rank(x)
    assert cs.shape == x.shape
    assert cs[0, 0] < cs[0, 1]  # 1.0 < 10.0, so column 0 has lower rank


def test_market_tensor_loading(sample_market_dict: dict[str, pd.DataFrame]):
    """Verify market dictionary is converted into aligned GPU/CPU PyTorch tensors."""
    miner = GPUAlphaMiner(device="cpu")
    features, tickers, fwd_ret = miner.load_market_tensor(sample_market_dict)

    assert "close" in features
    assert "volume" in features
    assert tickers == ["AAPL", "MSFT", "NVDA"]
    assert features["close"].shape == (60, 3)
    assert fwd_ret.shape == (60, 3)


def test_expression_evaluation(sample_market_dict: dict[str, pd.DataFrame]):
    """Verify safe evaluation of mathematical formula strings."""
    miner = GPUAlphaMiner(device="cpu")
    features, _, _ = miner.load_market_tensor(sample_market_dict)

    expr = "-1 * ts_rank(delta(close, 5), 10)"
    alpha_tensor = miner.evaluate_expression_string(expr, features)

    assert isinstance(alpha_tensor, torch.Tensor)
    assert alpha_tensor.shape == (60, 3)
    assert not torch.all(alpha_tensor == 0.0)


def test_alpha_search_pipeline(sample_market_dict: dict[str, pd.DataFrame]):
    """Verify automated formula search outputs ranked DiscoveredAlpha objects."""
    miner = GPUAlphaMiner(device="cpu", min_ic=0.01, min_ic_ir=0.20)
    features, _, fwd_ret = miner.load_market_tensor(sample_market_dict)

    results = miner.run_alpha_search(features, fwd_ret, num_candidates=10, seed=42)

    assert len(results) >= 10
    assert all(isinstance(a, DiscoveredAlpha) for a in results)
    # Check sorted descending by absolute ICIR
    for i in range(len(results) - 1):
        assert abs(results[i].ic_ir) >= abs(results[i + 1].ic_ir)
