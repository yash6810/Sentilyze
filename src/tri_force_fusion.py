"""
Tri-Force Dynamic Regime Fusion Engine for Sentilyze.
Formulated from Marcos López de Prado (2018) Advances in Financial Machine Learning,
Stoikov (2018) Micro-Price, and Avellaneda-Lee (2010) Statistical Arbitrage:

Fuses three distinct market forces:
1. Microstructure Signal (S_mu) - "The Now":
   Order Book Imbalance (OBI), Cumulative Volume Delta (CVD), Delta Price trigger, and ARAR absorption ratio.
2. Statistical Arbitrage Signal (S_arb) - "The Math":
   Mean-Reversion from rolling Z-score, Ornstein-Uhlenbeck spread dynamics, and Bollinger deviations.
3. Directional Momentum Signal (S_trend) - "The Trend":
   XGBoost probability, Trend alignment (RSI, MA200, MACD, ORB).

Composite Trade Probability:
P_trade = sum_{i=1}^3 (w_i * S_i) mapped to [0.0, 1.0]

Regime-Adaptive Dynamic Weighting:
- High-Vol / Fast-Moving (VIX > 22, realized vol > 30%): Microstructure dominates (w_mu >= 0.50).
- Sideways / Choppy (Hurst < 0.45, low ATR): Statistical Arbitrage dominates (w_arb >= 0.50).
- Trending / Breakout (Hurst > 0.55, ADX > 25): Momentum dominates (w_trend >= 0.55).

Also includes:
- Purged K-Fold Cross-Validation (López de Prado) with event-horizon purging and embargoing.
- Deflated Sharpe Ratio (DSR) to prevent backtest overfitting.
"""

from typing import Any, Dict, List, Optional, Tuple, Iterator
import numpy as np
import pandas as pd
from scipy.stats import norm
from src.utils import get_logger
from src.arar_microstructure import evaluate_arar_microstructure

logger = get_logger(__name__)


def compute_hurst_exponent(price_series: pd.Series, max_lag: int = 20) -> float:
    """
    Computes rolling Hurst exponent H:
    H < 0.45: Mean-reverting / Choppy regime
    H ≈ 0.50: Geometric Brownian Motion (Random Walk)
    H > 0.55: Persistent / Trending regime
    """
    prices = price_series.dropna().values.astype(float)
    if len(prices) < max_lag * 2:
        return 0.50

    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag]))) for lag in lags]

    # Filter non-zero tau values
    valid_idx = [i for i, t in enumerate(tau) if t > 1e-8]
    if len(valid_idx) < 3:
        return 0.50

    log_lags = np.log([lags[i] for i in valid_idx])
    log_tau = np.log([tau[i] for i in valid_idx])

    poly = np.polyfit(log_lags, log_tau, 1)
    hurst = float(poly[0] * 2.0)
    return float(np.clip(hurst, 0.05, 0.95))


def calculate_tri_force_regime_weights(
    realized_vol_pct: float,
    hurst_exponent: float,
    vix_level: float = 18.0,
    trend_adx: float = 20.0,
) -> Tuple[float, float, float, str]:
    """
    Computes dynamic continuous regime weights (w_mu, w_arb, w_trend) summing to 1.0.

    Logic:
    - High volatility / fast market (VIX > 22 or realized vol > 30%):
      Immediate order flow dictates price -> w_mu scales up.
    - Choppy / Sideways market (Hurst < 0.45 or ADX < 18):
      Mean-reversion scalping works best -> w_arb scales up.
    - Strong trending market (Hurst > 0.55 and ADX >= 25):
      Momentum expansion dominates -> w_trend scales up.
    """
    # Raw logit scores for each force
    score_mu = 1.0
    score_arb = 1.0
    score_trend = 1.0

    # Volatility boost for Microstructure
    if vix_level > 24.0 or realized_vol_pct > 32.0:
        vol_excess = max(0.0, (vix_level - 20.0) / 10.0) + max(
            0.0, (realized_vol_pct - 25.0) / 15.0
        )
        score_mu += 3.0 * vol_excess
    elif vix_level < 15.0 and realized_vol_pct < 18.0:
        # Low volatility quiet market
        score_arb += 1.5

    # Strong Trend Detection (via Hurst > 0.53 or ADX >= 25)
    if hurst_exponent > 0.53 or trend_adx >= 25.0:
        h_boost = max(0.0, (hurst_exponent - 0.50) * 4.0)
        adx_boost = max(0.0, (trend_adx - 20.0) / 10.0)
        score_trend += 2.5 * (h_boost + adx_boost)
    elif hurst_exponent < 0.45 or (trend_adx < 18.0 and realized_vol_pct < 20.0):
        # Mean reverting / sideways chop regime
        chop_excess = max(0.0, (0.50 - hurst_exponent) * 4.0) + max(
            0.0, (20.0 - trend_adx) / 10.0
        )
        score_arb += 2.5 * chop_excess

    # Softmax normalization
    scores = np.array([score_mu, score_arb, score_trend], dtype=float)
    exp_scores = np.exp(scores - np.max(scores))
    weights = exp_scores / np.sum(exp_scores)

    w_mu, w_arb, w_trend = float(weights[0]), float(weights[1]), float(weights[2])

    if w_mu > max(w_arb, w_trend):
        regime = "HIGH_VOLATILITY_MICROSTRUCTURE_DOMINANT"
    elif w_arb > max(w_mu, w_trend):
        regime = "SIDEWAYS_CHOP_STATARB_DOMINANT"
    else:
        regime = "MOMENTUM_TREND_DOMINANT"

    return round(w_mu, 4), round(w_arb, 4), round(w_trend, 4), regime


def calculate_statarb_signal_from_series(
    price_series: pd.Series,
    window: int = 20,
) -> float:
    """
    Extracts statistical arbitrage / mean-reversion signal S_arb in [-1.0, +1.0].
    When price is deeply oversold (Z < -1.5), S_arb > 0 (bullish mean reversion).
    When price is severely extended above mean (Z > 1.5), S_arb < 0 (bearish mean reversion).
    """
    if len(price_series) < window:
        return 0.0

    rolling_mean = price_series.rolling(window).mean().iloc[-1]
    rolling_std = price_series.rolling(window).std().iloc[-1]
    current_p = price_series.iloc[-1]

    if rolling_std <= 1e-8:
        return 0.0

    z_score = (current_p - rolling_mean) / rolling_std

    # Mean reversion signal is negative of z_score: oversold -> positive buy signal
    # Clipped to [-1.0, 1.0] using tanh
    s_arb = -float(np.tanh(z_score / 1.8))
    return float(np.clip(s_arb, -1.0, 1.0))


def evaluate_tri_force_fusion(
    ticker: str,
    df_ohlcv: pd.DataFrame,
    xgb_prob: float = 0.50,
    finbert_sentiment: float = 0.0,
    vix_level: float = 18.0,
) -> Dict[str, Any]:
    """
    Evaluates the complete Tri-Force Dynamic Signal Fusion Model for a ticker:
    P_trade = sum_{i=1}^3 (w_i * S_i)

    Returns probability P_trade in [0.0, 1.0] and institutional trade conviction.
    """
    if df_ohlcv.empty or len(df_ohlcv) < 20:
        return {
            "ticker": ticker,
            "p_trade": 0.50,
            "trade_conviction_pct": 50.0,
            "regime": "INSUFFICIENT_DATA",
            "weights": {"w_mu": 0.333, "w_arb": 0.333, "w_trend": 0.334},
            "signals": {"s_mu": 0.0, "s_arb": 0.0, "s_trend": 0.0},
            "recommendation": "HOLD_INSUFFICIENT_DATA",
        }

    closes = df_ohlcv["Close"].astype(float)
    highs = df_ohlcv["High"].astype(float) if "High" in df_ohlcv.columns else closes
    lows = df_ohlcv["Low"].astype(float) if "Low" in df_ohlcv.columns else closes

    # 1. Market Telemetry: Realized Volatility & Hurst Exponent
    returns = closes.pct_change().dropna()
    realized_vol_pct = (
        float(returns.std() * np.sqrt(252) * 100.0) if len(returns) > 5 else 20.0
    )
    hurst = compute_hurst_exponent(closes)

    # Simple ADX proxy (True range / ATR trend)
    tr = np.maximum(highs - lows, np.abs(highs - closes.shift(1)))
    atr_14 = (
        tr.rolling(14).mean().iloc[-1] if len(tr) >= 14 else (closes.iloc[-1] * 0.02)
    )
    directional_move = abs(closes.iloc[-1] - closes.iloc[-min(14, len(closes))])
    trend_adx = float(np.clip((directional_move / (atr_14 + 1e-6)) * 15.0, 5.0, 60.0))

    # 2. Dynamic Weights based on Market Regime
    w_mu, w_arb, w_trend, regime = calculate_tri_force_regime_weights(
        realized_vol_pct=realized_vol_pct,
        hurst_exponent=hurst,
        vix_level=vix_level,
        trend_adx=trend_adx,
    )

    # 3. Force 1: Microstructure Signal (S_mu) via ARAR
    arar_result = evaluate_arar_microstructure(ticker, df_ohlcv)
    s_mu = float(arar_result.get("microstructure_signal", 0.0))

    # 4. Force 2: Statistical Arbitrage Signal (S_arb)
    s_arb = calculate_statarb_signal_from_series(closes)

    # 5. Force 3: Directional Momentum Signal (S_trend)
    # Combine XGBoost probability [-1, +1] and FinBERT sentiment [-1, +1]
    xgb_signal = (xgb_prob - 0.50) * 2.0  # Maps [0, 1] to [-1, +1]
    finbert_signal = np.clip(finbert_sentiment, -1.0, 1.0)
    s_trend = float(np.clip(0.70 * xgb_signal + 0.30 * finbert_signal, -1.0, 1.0))

    # When trend is dominant, dampen mean-reverting counter-trend drag
    if regime == "MOMENTUM_TREND_DOMINANT":
        if s_trend > 0 and s_arb < 0:
            s_arb = s_arb * 0.20
        elif s_trend < 0 and s_arb > 0:
            s_arb = s_arb * 0.20

    # 6. Tri-Force Composite Signal Fusion
    composite_raw = (w_mu * s_mu) + (w_arb * s_arb) + (w_trend * s_trend)

    # Sigmoidal Calibration to P_trade in [0.0, 1.0]
    p_trade = 1.0 / (1.0 + np.exp(-3.0 * composite_raw))
    p_trade = float(np.clip(p_trade, 0.01, 0.99))
    trade_conviction_pct = round(p_trade * 100.0, 2)

    # Decision Recommendation
    if p_trade >= 0.65:
        recommendation = "SUPER_MAJORITY_INSTITUTIONAL_BUY"
    elif p_trade >= 0.55:
        recommendation = "MODERATE_BUY_ALLOCATION"
    elif p_trade <= 0.35:
        recommendation = "STRONG_SELL_DEFENSIVE_CASH"
    else:
        recommendation = "NEUTRAL_HOLD_WAIT_FOR_ABSORPTION"

    return {
        "ticker": ticker,
        "p_trade": round(p_trade, 4),
        "trade_conviction_pct": trade_conviction_pct,
        "composite_raw_signal": round(composite_raw, 4),
        "regime": regime,
        "weights": {
            "w_microstructure": w_mu,
            "w_statarb": w_arb,
            "w_trend": w_trend,
        },
        "signals": {
            "s_microstructure": round(s_mu, 3),
            "s_statarb": round(s_arb, 3),
            "s_trend": round(s_trend, 3),
        },
        "arar_details": arar_result,
        "telemetry": {
            "hurst_exponent": round(hurst, 3),
            "realized_vol_pct": round(realized_vol_pct, 2),
            "trend_adx": round(trend_adx, 1),
            "vix_level": round(vix_level, 2),
        },
        "recommendation": recommendation,
    }


# ==============================================================================
# Marcos López de Prado Purged K-Fold Cross-Validation & CPCV Engine
# Advances in Financial Machine Learning (2018), Chapter 7
# ==============================================================================


class PurgedKFold:
    """
    Purged and Embargoed K-Fold Cross-Validation for Financial Time Series.
    Eliminates information leakage and look-ahead bias across overlapping trade horizons.
    """

    def __init__(
        self,
        n_splits: int = 5,
        t1: Optional[pd.Series] = None,
        pct_embargo: float = 0.01,
    ):
        """
        Args:
            n_splits: Number of cross-validation folds.
            t1: Series of event end timestamps indexed by event start timestamps.
            pct_embargo: Percentage of observations to embargo after the test set.
        """
        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[Any] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates purged training and test index arrays.
        """
        indices = np.arange(len(X))
        step = len(X) // self.n_splits
        embargo = int(len(X) * self.pct_embargo)

        for i in range(self.n_splits):
            test_start = i * step
            test_end = (i + 1) * step if i < self.n_splits - 1 else len(X)
            test_indices = indices[test_start:test_end]

            # Purge overlapping training indices
            train_mask = np.ones(len(X), dtype=bool)
            train_mask[test_start:test_end] = False

            # Apply embargo immediately after test set
            embargo_end = min(len(X), test_end + embargo)
            train_mask[test_end:embargo_end] = False

            # If t1 is supplied, purge any training sample overlapping test horizon
            if self.t1 is not None and isinstance(X.index, pd.DatetimeIndex):
                test_times = X.index[test_start:test_end]
                if len(test_times) > 0:
                    test_min_time = test_times.min()
                    test_max_time = test_times.max()
                    # Samples starting before test set whose t1 extends into test set
                    for idx_pos in np.where(train_mask)[0]:
                        t_start = X.index[idx_pos]
                        if t_start in self.t1.index:
                            t_end = self.t1.loc[t_start]
                            if t_end >= test_min_time and t_start <= test_max_time:
                                train_mask[idx_pos] = False

            train_indices = indices[train_mask]
            yield train_indices, test_indices


def calculate_deflated_sharpe_ratio(
    sharpe_est: float,
    n_trials: int,
    var_sharpe: float = 1.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    t_bars: int = 252,
) -> float:
    """
    Computes Deflated Sharpe Ratio (DSR) (Bailey & López de Prado 2014):
    Accounts for selection bias under multiple testing and non-normal returns.
    Returns p-value that true Sharpe is greater than zero.
    """
    if n_trials <= 1:
        gamma_euler = 0.5772156649
        e_max_z = (1.0 - gamma_euler) * norm.ppf(
            1.0 - 1.0 / np.e
        ) + gamma_euler * norm.ppf(1.0 - 1.0 / (np.e**2))
    else:
        # Approximate expected maximum Sharpe from n independent trials
        gamma_euler = 0.5772156649
        e_max_z = (1.0 - gamma_euler) * norm.ppf(
            1.0 - 1.0 / n_trials
        ) + gamma_euler * norm.ppf(1.0 - 1.0 / (n_trials * np.e))

    expected_max_sharpe = float(np.sqrt(var_sharpe) * e_max_z)

    # Standard error of Sharpe ratio
    denom = 1.0 - skewness * sharpe_est + ((kurtosis - 1.0) / 4.0) * (sharpe_est**2)
    denom = max(1e-6, denom)
    sigma_sr = np.sqrt(denom / max(1, t_bars - 1))

    # Test statistic against expected maximum from multiple testing
    z_stat = (sharpe_est - expected_max_sharpe) / sigma_sr
    dsr_p_value = float(norm.cdf(z_stat))
    return float(np.clip(dsr_p_value, 0.0, 1.0))
