import os
import shap
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from src.preprocessing import preprocess_data
from src.modeling import load_model, get_prediction_on_latest_data
from src.config import FEATURES
from src.utils import get_logger, sanitize_filename, safe_path_join

logger = get_logger(__name__)

app = FastAPI(
    title="Sentilyze Inference API",
    description="Production REST microservice for AI-powered next-day stock momentum predictions combining NLP sentiment and technical indicators.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_TICKERS = ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "TSLA", "AMZN"]


class FeatureContribution(BaseModel):
    feature: str = Field(..., description="Feature name")
    value: float = Field(..., description="Feature input value")
    shap_value: float = Field(..., description="SHAP attribution impact on prediction")


class PredictionResponse(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    signal: str = Field(..., description="Trading signal: BUY, SELL, or HOLD")
    prediction: int = Field(
        ..., description="Binary directional prediction: 1 (Up) or 0 (Down)"
    )
    confidence: float = Field(
        ..., description="Predicted probability of positive momentum"
    )
    rsi: float = Field(..., description="Current 14-period RSI indicator")
    trend: str = Field(
        ..., description="Macro trend relative to 200-day SMA: Bullish or Bearish"
    )
    top_features: List[FeatureContribution] = Field(
        ..., description="Top contributing features by SHAP attribution magnitude"
    )
    metadata: Dict[str, Any] = Field(
        ..., description="Execution metadata and pipeline information"
    )


@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Sentilyze Inference API",
        "version": "1.0.0",
        "supported_tickers": SUPPORTED_TICKERS,
        "docs": "/docs",
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    return {"status": "healthy", "service": "sentilyze-api"}


@app.get("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(
    ticker: str = Query(..., description="Stock ticker symbol (e.g. NVDA, AAPL, MSFT)"),
    use_cache: bool = Query(
        True, description="Whether to favor cached data for fast response"
    ),
):
    """
    Fetches the latest market and sentiment data, computes technical indicators,
    runs the XGBoost specialist model, and computes SHAP feature attributions.
    """
    ticker_upper = sanitize_filename(ticker.strip().upper())
    model_path = safe_path_join("models", f"{ticker_upper}_model.json")

    if not os.path.exists(model_path) and not os.path.exists(
        model_path.replace(".json", ".joblib")
    ):
        raise HTTPException(
            status_code=404,
            detail=f"Trained model for ticker '{ticker_upper}' not found. Supported tickers: {SUPPORTED_TICKERS}",
        )

    try:
        # 1. Preprocess latest data
        features_df, price_hist, news_df = preprocess_data(
            ticker_upper, use_cache=use_cache
        )
        if features_df.empty:
            raise HTTPException(
                status_code=500,
                detail=f"Insufficient data to generate prediction for {ticker_upper}",
            )

        # 2. Load model and run inference
        model = load_model(model_path)
        latest_features = features_df.iloc[-1:][FEATURES]

        pred, conf = get_prediction_on_latest_data(model, latest_features, FEATURES)
        confidence = float(conf[0][1])
        rsi = float(price_hist["rsi"].iloc[-1])
        sma200 = float(price_hist["sma200"].iloc[-1])
        latest_close = float(price_hist["Close"].iloc[-1])
        trend = "Bullish" if latest_close > sma200 else "Bearish"

        # 3. Regime Filter Logic
        if confidence > 0.80 and rsi < 70:
            signal = "BUY"
            final_pred = 1
        elif confidence > 0.50 and rsi < 70:
            signal = "BUY"
            final_pred = 1
        elif confidence <= 0.50:
            signal = "SELL"
            final_pred = 0
        else:
            signal = "HOLD"
            final_pred = 0

        # 4. SHAP Feature Attribution
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(latest_features)

        if isinstance(shap_values, list):
            shap_vec = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif len(shap_values.shape) == 2:
            shap_vec = shap_values[0]
        else:
            shap_vec = shap_values

        contributions = []
        for feat_name, shap_val in zip(FEATURES, shap_vec):
            feat_val = float(latest_features[feat_name].iloc[0])
            contributions.append(
                FeatureContribution(
                    feature=feat_name,
                    value=round(feat_val, 4),
                    shap_value=round(float(shap_val), 4),
                )
            )

        # Sort top 5 features by absolute SHAP impact
        top_features = sorted(
            contributions, key=lambda c: abs(c.shap_value), reverse=True
        )[:5]

        return PredictionResponse(
            ticker=ticker_upper,
            signal=signal,
            prediction=final_pred,
            confidence=round(confidence, 4),
            rsi=round(rsi, 2),
            trend=trend,
            top_features=top_features,
            metadata={
                "model_type": "XGBClassifier (Walk-Forward Optimized)",
                "sentiment_model": "FinBERT fine-tuned",
                "cached": use_cache,
                "data_points": len(features_df),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error for {ticker_upper}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Prediction pipeline failure: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
