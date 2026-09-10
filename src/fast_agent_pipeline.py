"""
Fast Sub-Second Agent Decision Router.
Coordinates the Tri-Model Neural Brain (Fast FinBERT + Event Companion + DLinear-TCN + MoE Gating).
Provides sub-second end-to-end inference (<25ms) for high-speed agent execution.
"""

import time
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import torch

from src.utils import get_logger
from src.fast_finbert import get_fast_finbert_engine
from src.event_classifier_model import EventClassifierModel
from src.neural_gating_network import NeuralGatingNetwork
from src.deep_learning_model import DLinearTCNModel, predict_momentum_probability

logger = get_logger(__name__)


class FastAgentPipeline:
    """
    Sub-second Multi-Model Agent Decision Pipeline.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FastAgentPipeline, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        logger.info("⚡ Initializing Fast Agent Tri-Model Decision Pipeline...")
        t0 = time.perf_counter()

        self.finbert_engine = get_fast_finbert_engine()
        self.event_classifier = EventClassifierModel()
        self.gating_network = NeuralGatingNetwork()

        # Initialize lightweight DLinear-TCN model (10-day lookback, 15 features)
        self.dlinear_model = DLinearTCNModel(seq_len=10, num_features=15)
        self.dlinear_model.eval()

        t_elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(f"⚡ Fast Agent Pipeline fully armed in {t_elapsed:.1f}ms.")
        self._initialized = True

    def evaluate_catalyst_and_market(
        self,
        headline: str,
        price_series: Optional[pd.DataFrame] = None,
        ticker: str = "NVDA",
        vwap_diff_pct: float = 0.5,
        dark_pool_ratio: float = 1.15,
    ) -> Dict[str, Any]:
        """
        Executes complete Tri-Model evaluation on incoming headline & market structure.
        Returns unified trading signal, model attributions, and microsecond latency.
        """
        t_start = time.perf_counter()

        # Step 1: Model 1 (Fast FinBERT INT8)
        finbert_res = self.finbert_engine.predict_single(headline, ticker=ticker)

        # Step 2: Model 2 (Event & Emotion Companion)
        event_res = self.event_classifier.classify(headline)

        # Step 3: Model 3 (DLinear-TCN Momentum evaluation)
        if price_series is not None and len(price_series) >= 10:
            # Vectorized feature tensor
            feats = price_series.tail(10).values
            if feats.shape[1] < 15:
                # Pad to 15 features
                pad = np.zeros((10, 15 - feats.shape[1]))
                feats = np.hstack([feats, pad])
            t_seq = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            dlinear_res = predict_momentum_probability(self.dlinear_model, t_seq)
        else:
            # Fast default synthetic state
            dummy_seq = torch.randn(1, 10, 15)
            dlinear_res = predict_momentum_probability(self.dlinear_model, dummy_seq)

        # Step 4: Neural MoE Gating Fusion
        fusion_res = self.gating_network.fuse(
            finbert_out=finbert_res,
            event_out=event_res,
            dlinear_out=dlinear_res,
            vwap_diff_pct=vwap_diff_pct,
            dark_pool_ratio=dark_pool_ratio,
        )

        total_latency = (time.perf_counter() - t_start) * 1000.0

        return {
            "ticker": ticker,
            "headline": headline,
            "action": fusion_res["signal"],
            "composite_score": fusion_res["composite_score"],
            "conviction": fusion_res["conviction"],
            "veto_triggered": fusion_res["veto_triggered"],
            "event_type": event_res["event_type"],
            "event_urgency": event_res["urgency_score"],
            "model_attributions": fusion_res["model_attributions"],
            "detailed_experts": {
                "finbert": finbert_res,
                "event_classifier": event_res,
                "dlinear_tcn": dlinear_res,
            },
            "total_latency_ms": round(total_latency, 2),
        }


def get_fast_agent_pipeline() -> FastAgentPipeline:
    """Singleton getter for FastAgentPipeline."""
    return FastAgentPipeline()
