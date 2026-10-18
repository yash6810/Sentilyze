"""
Train Native XGBoost JSON Models for Crypto Majors (BTC-USD, ETH-USD, SOL-USD).
==============================================================================
Institutional Cross-Validation Pipeline:
- Combinatorial Purged & Embargoed Cross-Validation (CPCV, López de Prado 2018).
- Conformal probability calibration.
- Native XGBoost JSON format (zero pickle/joblib).
- Saves directly to models/{ticker}_model.json.
"""

import os
import sys
from typing import List

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import get_logger
from src.continuous_learner import execute_continuous_retrain_cycle

logger = get_logger("train_crypto_models")

CRYPTO_MAJORS: List[str] = ["BTC-USD", "ETH-USD", "SOL-USD"]


def main():
    logger.info(
        "🚀 [CRYPTO MODEL TRAINING] Starting training pipeline for crypto majors..."
    )
    results = {}

    for ticker in CRYPTO_MAJORS:
        logger.info(f"⏳ Training native model for {ticker}...")
        try:
            res = execute_continuous_retrain_cycle(
                ticker=ticker,
                tune_hyperparameters=False,
                force_deploy=True,
            )
            results[ticker] = res
            status = res.get("deployment_status", res.get("status", "SUCCESS"))
            acc = res.get("metrics", {}).get("accuracy", 0.0)
            auc = res.get("metrics", {}).get("roc_auc", 0.0)
            logger.info(
                f"✅ [{ticker}] Status: {status} | Accuracy: {acc:.2%} | AUC: {auc:.3f} | Model Saved to models/{ticker}_model.json"
            )
        except Exception as e:
            logger.error(f"❌ Failed to train model for {ticker}: {e}")
            results[ticker] = {"status": "FAILED", "error": str(e)}

    logger.info("🏆 [CRYPTO MODEL TRAINING COMPLETE] Summary:")
    for ticker, res in results.items():
        logger.info(
            f" - {ticker}: {res.get('deployment_status', res.get('status'))} "
            f"(Acc: {res.get('metrics', {}).get('accuracy', 0.0):.2%})"
        )


if __name__ == "__main__":
    main()
