import pytest
import pandas as pd
import numpy as np
from src.continuous_learner import (
    enrich_features_with_alpha_interactions,
    execute_continuous_retrain_cycle,
)


def test_enrich_features_with_alpha_interactions():
    df = pd.DataFrame(
        {
            "rsi": [30.0, 45.0, 70.0],
            "mean_sentiment_score": [0.5, 0.2, -0.4],
            "return_5d": [0.02, -0.01, 0.05],
            "atr": [2.5, 2.8, 3.0],
            "ma7": [100.0, 102.0, 105.0],
            "ma21": [98.0, 100.0, 101.0],
        }
    )
    enriched = enrich_features_with_alpha_interactions(df)
    assert "rsi_sentiment_interaction" in enriched.columns
    assert "vol_adjusted_momentum" in enriched.columns
    assert "sentiment_delta_3d" in enriched.columns
    assert "ma_convergence_divergence" in enriched.columns
