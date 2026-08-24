# src/config.py

# List of features to be used for model training and prediction
FEATURES = [
    "ma7",
    "ma21",
    "sma200",
    "rsi",
    "macd",
    "bollinger_upper",
    "bollinger_lower",
    "stochastic_oscillator",
    "atr",
    "return_1d",
    "return_5d",
    "return_21d",
    "volatility_21d",
    "price_to_sma200",
    "rsi_slope",
    "ma_spread",
    "volume_ratio",
    "atr_ratio",
    "vix_close",
    "vix_ma5",
    "vix_change_1d",
    "mean_sentiment_score",
    "positive",
    "negative",
    "neutral",
]

# Hyperparameters for the XGBoost Classifier (regularized for low-noise time series)
XGB_MODEL_PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.03,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "eval_metric": "logloss",
}
