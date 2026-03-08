# src/config.py

# List of features to be used for model training and prediction
FEATURES = [
    'ma7',
    'ma21',
    'sma200',
    'rsi',
    'macd',
    'bollinger_upper',
    'bollinger_lower',
    'stochastic_oscillator',
    'atr',
    'vix_close',
    'vix_ma5',
    'mean_sentiment_score',
    'positive',
    'negative',
    'neutral'
]

# Hyperparameters for the XGBoost Classifier
XGB_MODEL_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 4,
    'random_state': 42,
    'eval_metric': 'logloss'
}
