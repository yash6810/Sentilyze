"""
Neural Mixture-of-Experts (MoE) Gating Network.
Fuses the Tri-Model Neural Brain:
  - Model 1: Fast FinBERT (Sentiment Polarity)
  - Model 2: Event & Emotion Companion Model (Catalyst Urgency & Type)
  - Model 3: DLinear-TCN (Price Structure & Momentum)
  - Microstructure Filter: Anchored VWAP & Dark Pool prints

Outputs calibrated directional signals (STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL),
conviction scores, model attributions, and microsecond-level latency tracking.
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import get_logger

logger = get_logger(__name__)


class TriBrainGatingLayer(nn.Module):
    """
    Neural Softmax Gating layer to dynamically allocate weights among the 3 models.
    Input features: [sentiment_score, confidence, event_bias, urgency, dlinear_prob, vwap_diff, dark_pool_ratio]
    """

    def __init__(self, input_dim: int = 7, num_experts: int = 3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, num_experts),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class NeuralGatingNetwork:
    """
    Production-grade Multi-Model Fusion Router.
    """

    def __init__(self):
        self.gating_net = TriBrainGatingLayer()
        self.gating_net.eval()

    def fuse(
        self,
        finbert_out: Dict[str, Any],
        event_out: Dict[str, Any],
        dlinear_out: Dict[str, Any],
        vwap_diff_pct: float = 0.0,
        dark_pool_ratio: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Fuses predictions from all three models into a unified decision with sub-millisecond latency.
        """
        t0 = time.perf_counter()

        # Extract normalized model features
        s_score = float(finbert_out.get("sentiment_score", 0.0))  # [-1, 1]
        s_conf = float(finbert_out.get("confidence", 0.5))  # [0, 1]
        e_bias = float(event_out.get("sentiment_bias", 0.0))  # [-1, 1]
        e_urgency = float(event_out.get("urgency_score", 0.2))  # [0, 1]
        d_prob = float(dlinear_out.get("bullish_probability", 0.5))  # [0, 1]
        d_score = (d_prob - 0.5) * 2.0  # [-1, 1]

        # Construct input tensor for Neural Gating
        features = np.array(
            [
                s_score,
                s_conf,
                e_bias,
                e_urgency,
                d_prob,
                vwap_diff_pct,
                dark_pool_ratio,
            ],
            dtype=np.float32,
        )
        t_feat = torch.tensor(features).unsqueeze(0)

        with torch.no_grad():
            weights = self.gating_net(t_feat).numpy()[0]

        w_finbert, w_event, w_dlinear = (
            float(weights[0]),
            float(weights[1]),
            float(weights[2]),
        )

        # Adjust weights dynamically if event is high-urgency material catalyst
        if event_out.get("is_material", False) and e_urgency >= 0.75:
            # Shift weight towards event + sentiment
            w_event = min(0.60, w_event + 0.20)
            w_finbert = min(0.35, w_finbert + 0.10)
            w_dlinear = max(0.05, 1.0 - (w_event + w_finbert))

        # Re-normalize weights to sum to 1.0
        total_w = w_finbert + w_event + w_dlinear
        w_finbert /= total_w
        w_event /= total_w
        w_dlinear /= total_w

        # Compute composite directional score in [-1.0, +1.0]
        raw_composite = (
            (w_finbert * s_score) + (w_event * e_bias) + (w_dlinear * d_score)
        )

        # Apply Institutional Gate (Anchored VWAP & Dark Pool verification)
        institutional_multiplier = 1.0
        veto_triggered = False

        if raw_composite > 0:
            # Long setup verification
            if vwap_diff_pct < -1.5 and dark_pool_ratio < 0.90:
                # Price below AVWAP with institutional selling -> Bull Trap Veto
                institutional_multiplier = 0.20
                veto_triggered = True
            elif vwap_diff_pct >= 0 and dark_pool_ratio >= 1.20:
                # Price holding above AVWAP with whale accumulation -> Conviction Boost
                institutional_multiplier = 1.25
        elif raw_composite < 0:
            # Short setup verification
            if vwap_diff_pct > 1.5 and dark_pool_ratio > 1.20:
                # Price above AVWAP with whale buying -> Bear Trap Veto
                institutional_multiplier = 0.20
                veto_triggered = True

        composite_score = np.clip(raw_composite * institutional_multiplier, -1.0, 1.0)
        conviction = abs(composite_score)

        # Directional Signal Mapping
        if composite_score >= 0.55:
            signal = "STRONG_BUY"
        elif composite_score >= 0.20:
            signal = "BUY"
        elif composite_score <= -0.55:
            signal = "STRONG_SELL"
        elif composite_score <= -0.20:
            signal = "SELL"
        else:
            signal = "NEUTRAL"

        t_elapsed = (time.perf_counter() - t0) * 1000.0

        return {
            "signal": signal,
            "composite_score": round(float(composite_score), 4),
            "conviction": round(float(conviction), 4),
            "veto_triggered": veto_triggered,
            "model_attributions": {
                "finbert_sentiment_pct": round(w_finbert * 100.0, 1),
                "event_companion_pct": round(w_event * 100.0, 1),
                "dlinear_tcn_pct": round(w_dlinear * 100.0, 1),
            },
            "expert_scores": {
                "finbert_polarity": s_score,
                "event_type": event_out.get("event_type", "UNKNOWN"),
                "event_urgency": e_urgency,
                "dlinear_momentum_prob": d_prob,
            },
            "latency_ms": round(t_elapsed, 4),
        }
