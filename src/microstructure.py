"""
Institutional Market Microstructure, Order Flow Toxicity & Liquidity Shield for Sentilyze.

Grounding:
1. Easley, Lopez de Prado, O'Hara (2012) - Flow Toxicity and Liquidity in a High-Frequency World (VPIN)
2. Corwin & Schultz (2012) - A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices
3. Roll (1984) - A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market
4. Amihud (2002) - Illiquidity and Stock Returns: Cross-Section and Time-Series Effects
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm
from src.utils import get_logger
from src.data_ingestion import get_price_history

logger = get_logger(__name__)


def compute_vpin(
    df: pd.DataFrame,
    bucket_volume: Optional[float] = None,
    n_buckets: int = 50,
) -> Dict[str, Any]:
    """
    Computes Volume-Synchronized Probability of Toxicity (VPIN) using Bulk Volume Classification (BVC).

    VPIN = sum(|V_tau^B - V_tau^S|) / (N * V)
    where:
    - Delta P = P_t - P_{t-1}
    - Z = Delta P / sigma(Delta P)
    - V_B = V * Phi(Z)
    - V_S = V * (1 - Phi(Z))
    """
    if (
        df.empty
        or len(df) < 15
        or "Close" not in df.columns
        or "Volume" not in df.columns
    ):
        return {
            "vpin": 0.35,
            "toxicity_regime": "NORMAL_FLOW",
            "is_toxic": False,
            "buyer_volume_ratio": 0.50,
            "warning": "Insufficient price/volume data.",
        }

    closes = df["Close"].astype(float).values
    volumes = df["Volume"].astype(float).values

    # Avoid zero volumes
    volumes = np.where(volumes <= 0, 1.0, volumes)

    # 1. Price changes and standard deviation
    delta_p = np.diff(closes)
    sigma_dp = np.std(delta_p) + 1e-9
    z_scores = delta_p / sigma_dp

    # 2. Bulk Volume Classification (BVC)
    phi_z = norm.cdf(z_scores)
    step_vols = volumes[1:]
    buy_vols = step_vols * phi_z
    sell_vols = step_vols * (1.0 - phi_z)

    # 3. Synchronized volume bucketing
    total_vol = np.sum(step_vols)
    if bucket_volume is None or bucket_volume <= 0:
        bucket_volume = max(total_vol / float(n_buckets), 1000.0)

    bucket_imbalances = []
    curr_buy = 0.0
    curr_sell = 0.0
    curr_vol = 0.0

    for b, s, v in zip(buy_vols, sell_vols, step_vols):
        curr_buy += b
        curr_sell += s
        curr_vol += v
        if curr_vol >= bucket_volume:
            bucket_imbalances.append(abs(curr_buy - curr_sell))
            curr_buy = 0.0
            curr_sell = 0.0
            curr_vol = 0.0

    if not bucket_imbalances:
        # Fallback if fewer than 1 bucket filled
        vpin_val = float(
            np.sum(np.abs(buy_vols - sell_vols)) / (np.sum(step_vols) + 1e-9)
        )
    else:
        vpin_val = float(np.mean(bucket_imbalances) / bucket_volume)

    vpin_val = float(np.clip(vpin_val, 0.01, 0.99))

    # Toxicity Regime Thresholds (Paper 2012)
    # < 0.40: Low Toxicity / Clean Flow
    # 0.40 - 0.65: Moderate Flow
    # > 0.65: Severe Adverse Selection / Toxic Institutional Flow
    if vpin_val >= 0.65:
        regime = "HIGH_TOXICITY"
        is_toxic = True
    elif vpin_val >= 0.42:
        regime = "MODERATE_TOXICITY"
        is_toxic = False
    else:
        regime = "CLEAN_FLOW"
        is_toxic = False

    tot_buy = float(np.sum(buy_vols))
    tot_step = float(np.sum(step_vols))
    buyer_ratio = float(tot_buy / (tot_step + 1e-9))

    return {
        "vpin": round(vpin_val, 4),
        "toxicity_regime": regime,
        "is_toxic": is_toxic,
        "buyer_volume_ratio": round(buyer_ratio, 4),
        "seller_volume_ratio": round(1.0 - buyer_ratio, 4),
        "bucket_count": len(bucket_imbalances),
    }


def compute_corwin_schultz_spread(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Computes Corwin-Schultz (2012) High-Low Effective Bid-Ask Spread Estimator.

    alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2)))
    S_CS = 2 * (exp(alpha) - 1) / (1 + exp(alpha))
    """
    if df.empty or len(df) < 5 or "High" not in df.columns or "Low" not in df.columns:
        return pd.Series(dtype=float)

    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    # 1. 1-day high/low log ratio squared
    hl_ratio_sq = (np.log(high / (low + 1e-9))) ** 2
    beta = hl_ratio_sq + hl_ratio_sq.shift(1)

    # 2. 2-day high/low log ratio squared
    h2 = pd.concat([high, high.shift(1)], axis=1).max(axis=1)
    l2 = pd.concat([low, low.shift(1)], axis=1).min(axis=1)
    gamma = (np.log(h2 / (l2 + 1e-9))) ** 2

    # 3. Alpha calculation with numerical stability
    k = 3.0 - 2.0 * np.sqrt(2.0)
    sqrt_beta = np.sqrt(beta)
    alpha = (np.sqrt(2.0 * beta) - sqrt_beta) / k - np.sqrt(gamma / k)

    # Handle negative alpha or invalid
    alpha = alpha.fillna(0.0)
    alpha = alpha.clip(lower=-5.0, upper=2.0)

    # 4. Spread formulation
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    spread = spread.clip(lower=0.0, upper=0.25)  # Cap spread at 25% max

    if window > 1:
        spread = spread.rolling(window, min_periods=3).median()

    return spread.fillna(0.001)


def compute_roll_spread(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Computes Roll (1984) Effective Spread Estimator from serial price autocovariance:
    S_Roll = 2 * sqrt(-min(0, Cov(Delta P_t, Delta P_{t-1})))
    """
    if df.empty or len(df) < 5 or "Close" not in df.columns:
        return pd.Series(dtype=float)

    closes = df["Close"].astype(float)
    dp = closes.diff()
    dp_lag = dp.shift(1)

    # Rolling covariance between Delta P_t and Delta P_{t-1}
    cov_series = dp.rolling(window, min_periods=5).cov(dp_lag)

    # When cov < 0 (bid-ask bounce exists), spread = 2 * sqrt(-cov)
    # Relative to price
    neg_cov = np.where(cov_series < 0, -cov_series, 0.0)
    dollar_spread = 2.0 * np.sqrt(neg_cov)
    rel_spread = dollar_spread / (closes + 1e-9)

    return pd.Series(rel_spread, index=df.index).fillna(0.0005)


def compute_amihud_illiquidity(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Computes Amihud (2002) Illiquidity Ratio:
    ILLIQ_t = |R_t| / (Price_t * Volume_t) * 1e6
    """
    if (
        df.empty
        or len(df) < 5
        or "Close" not in df.columns
        or "Volume" not in df.columns
    ):
        return pd.Series(dtype=float)

    closes = df["Close"].astype(float)
    volumes = df["Volume"].astype(float)
    returns = closes.pct_change().abs()

    dollar_vol = closes * volumes
    dollar_vol = np.where(dollar_vol <= 0, 1.0, dollar_vol)

    illiq = (returns / dollar_vol) * 1e6
    if window > 1:
        illiq = illiq.rolling(window, min_periods=3).median()

    return illiq.fillna(0.0)


def evaluate_microstructure_health(
    ticker: str,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Full-spectrum institutional market microstructure audit for a given asset.
    Returns composite liquidity score, VPIN toxicity, Corwin-Schultz spread (bps), and execution gates.
    """
    if df is None or df.empty:
        try:
            df = get_price_history(ticker, period="6mo", use_cache=True)
        except Exception as e:
            logger.debug(
                f"Failed to fetch price history for microstructure {ticker}: {e}"
            )
            df = pd.DataFrame()

    if df.empty or len(df) < 15:
        return {
            "ticker": ticker,
            "status": "INSUFFICIENT_DATA",
            "microstructure_score": 50.0,
            "vpin": 0.35,
            "vpin_regime": "NORMAL_FLOW",
            "is_toxic_flow": False,
            "corwin_schultz_spread_bps": 5.0,
            "roll_spread_bps": 4.0,
            "amihud_illiquidity": 0.01,
            "execution_recommendation": "STANDARD_EXECUTION",
            "action_penalty_multiplier": 1.0,
        }

    # 1. VPIN Order Flow Toxicity
    vpin_data = compute_vpin(df)
    vpin_val = vpin_data["vpin"]
    is_toxic = vpin_data["is_toxic"]
    vpin_regime = vpin_data["toxicity_regime"]

    # 2. Corwin-Schultz Spread
    cs_series = compute_corwin_schultz_spread(df)
    cs_spread = (
        float(cs_series.dropna().iloc[-1]) if not cs_series.dropna().empty else 0.001
    )
    cs_bps = round(cs_spread * 10000.0, 1)

    # 3. Roll Spread
    roll_series = compute_roll_spread(df)
    roll_spread = (
        float(roll_series.dropna().iloc[-1])
        if not roll_series.dropna().empty
        else 0.0005
    )
    roll_bps = round(roll_spread * 10000.0, 1)

    # 4. Amihud Illiquidity
    amihud_series = compute_amihud_illiquidity(df)
    amihud_val = (
        float(amihud_series.dropna().iloc[-1])
        if not amihud_series.dropna().empty
        else 0.0
    )

    # 5. Composite Microstructure Score (0 to 100)
    # Higher = better execution quality, lower spreads, cleaner order flow
    score = 100.0
    # Penalty for high VPIN (> 0.50)
    if vpin_val > 0.50:
        score -= (vpin_val - 0.50) * 80.0
    # Penalty for wide spreads (> 15 bps)
    if cs_bps > 15.0:
        score -= min((cs_bps - 15.0) * 1.5, 30.0)
    # Penalty for Amihud illiquidity
    if amihud_val > 0.10:
        score -= min(amihud_val * 50.0, 20.0)

    composite_score = round(max(min(score, 100.0), 10.0), 1)

    # 6. Execution Recommendations & Sizing Multipliers
    if is_toxic or composite_score < 40.0:
        rec = "TOXIC_FLOW_CAUTION_EXECUTION_PENALTY"
        penalty = 0.50  # 50% Kelly position haircut
    elif composite_score < 65.0 or cs_bps > 25.0:
        rec = "ELEVATED_SPREAD_SCALE_IN_SLOWLY"
        penalty = 0.75
    else:
        rec = "EXECUTION_OPTIMAL"
        penalty = 1.0

    return {
        "ticker": ticker,
        "status": "AUDIT_COMPLETE",
        "microstructure_score": composite_score,
        "vpin": vpin_val,
        "vpin_regime": vpin_regime,
        "is_toxic_flow": is_toxic,
        "buyer_ratio": vpin_data.get("buyer_volume_ratio", 0.5),
        "corwin_schultz_spread_bps": cs_bps,
        "roll_spread_bps": roll_bps,
        "amihud_illiquidity": round(amihud_val, 5),
        "execution_recommendation": rec,
        "action_penalty_multiplier": penalty,
    }
