"""
Volume-Synchronized Probability of Toxicity (VPIN) Microstructure Engine for Sentilyze.

Implementation of Easley, Lopez de Prado & O'Hara (2012):
"The Volume Clock: Resulting Implications for Systemic Risk and Trading".

Institutional Capabilities:
1. Equal-Volume Bucket Partitioning: Transforms calendar-time price/volume into volume-time clock.
2. Bulk Volume Classification (BVC): Decomposes trade volume into buyer-initiated vs seller-initiated volume
   using the cumulative normal distribution of standardized price changes:
   V_tau^B = V * Phi(Delta P_tau / sigma_DeltaP)
   V_tau^S = V * (1 - Phi(Delta P_tau / sigma_DeltaP))
3. Vectorized VPIN Metric:
   VPIN = sum_{tau=1}^N |V_tau^B - V_tau^S| / (N * V)
4. Toxic Flow & Institutional Dumping Detection:
   Signals the Chief Risk Officer (CRO) to veto entries when toxic informed traders dominate flow.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from src.utils import get_logger
from src.data_ingestion import get_price_history

logger = get_logger(__name__)


def calculate_vpin(
    df: pd.DataFrame,
    num_buckets: int = 50,
    rolling_window: int = 20,
) -> Dict[str, Any]:
    """
    Computes Volume-Synchronized Probability of Toxicity (VPIN) from price and volume data.

    Args:
        df: DataFrame with 'Close' and 'Volume' columns.
        num_buckets: Total number of volume buckets to partition into (default: 50).
        rolling_window: Number of volume buckets used for rolling VPIN (default: 20).

    Returns:
        Dict containing current VPIN score, toxicity regime, CDF distribution, and CRO veto flag.
    """
    if (
        df is None
        or len(df) < 15
        or "Close" not in df.columns
        or "Volume" not in df.columns
    ):
        return {
            "vpin": 0.35,
            "toxicity_regime": "INSUFFICIENT_DATA",
            "is_toxic_dumping": False,
            "cro_veto_recommended": False,
            "confidence": 0.5,
        }

    prices = df["Close"].values.astype(float)
    volumes = df["Volume"].values.astype(float)

    # Avoid zero-volume flatlines
    valid_mask = volumes > 0
    if np.sum(valid_mask) < 10:
        return {
            "vpin": 0.35,
            "toxicity_regime": "ILLIQUID_FLATLINE",
            "is_toxic_dumping": False,
            "cro_veto_recommended": True,
            "confidence": 0.4,
        }

    prices = prices[valid_mask]
    volumes = volumes[valid_mask]

    total_volume = np.sum(volumes)
    bucket_size = total_volume / max(num_buckets, 5)

    if bucket_size <= 0:
        return {
            "vpin": 0.35,
            "toxicity_regime": "ZERO_VOLUME",
            "is_toxic_dumping": False,
            "cro_veto_recommended": False,
            "confidence": 0.5,
        }

    # Partition into volume buckets using cumulative volume
    cum_vol = np.cumsum(volumes)
    bucket_edges = np.arange(bucket_size, total_volume + bucket_size, bucket_size)

    # Compute price changes between volume bucket boundaries
    bucket_indices = np.searchsorted(cum_vol, bucket_edges)
    bucket_indices = np.clip(bucket_indices, 0, len(prices) - 1)

    bucket_prices = prices[bucket_indices]
    price_deltas = np.diff(bucket_prices)

    if len(price_deltas) < 5:
        return {
            "vpin": 0.35,
            "toxicity_regime": "LOW_SAMPLE_COUNT",
            "is_toxic_dumping": False,
            "cro_veto_recommended": False,
            "confidence": 0.5,
        }

    # Standardize price changes
    sigma_dp = np.std(price_deltas)
    if sigma_dp <= 1e-8:
        # No volatility
        return {
            "vpin": 0.20,
            "toxicity_regime": "ZERO_VOLATILITY",
            "is_toxic_dumping": False,
            "cro_veto_recommended": False,
            "confidence": 0.6,
        }

    std_deltas = price_deltas / sigma_dp

    # Bulk Volume Classification (BVC)
    # Buy volume fraction: Phi(std_delta)
    # Imbalance = |2 * Phi(std_delta) - 1| * bucket_size
    phi_values = stats.norm.cdf(std_deltas)
    imbalances = np.abs(2.0 * phi_values - 1.0) * bucket_size

    # Rolling VPIN calculation over the last N buckets
    window = min(rolling_window, len(imbalances))
    recent_imbalances = imbalances[-window:]
    vpin = float(np.sum(recent_imbalances) / (window * bucket_size))
    vpin = round(float(np.clip(vpin, 0.0, 1.0)), 4)

    # Classify Toxicity Regime
    if vpin >= 0.78:
        toxicity_regime = "TOXIC_INSTITUTIONAL_DUMPING"
        is_toxic = True
        veto = True
    elif vpin >= 0.65:
        toxicity_regime = "ELEVATED_INVENTORY_RISK"
        is_toxic = False
        veto = False
    elif vpin >= 0.40:
        toxicity_regime = "BALANCED_ORDER_FLOW"
        is_toxic = False
        veto = False
    else:
        toxicity_regime = "LOW_TOXICITY_BENIGN"
        is_toxic = False
        veto = False

    return {
        "vpin": vpin,
        "toxicity_regime": toxicity_regime,
        "is_toxic_dumping": is_toxic,
        "cro_veto_recommended": veto,
        "bucket_size": round(float(bucket_size), 2),
        "total_buckets": len(imbalances),
        "confidence": 0.85,
    }


def evaluate_ticker_toxicity(
    ticker: str, period: str = "5d", interval: str = "5m"
) -> Dict[str, Any]:
    """
    Pulls price history and evaluates institutional order flow toxicity for a given ticker.
    First attempts to pull intraday 5-min bars (Easley, Lopez de Prado & O'Hara 2012)
    to compute micro-bucket VPIN accurately. If unavailable or insufficient,
    gracefully falls back to daily bars.

    Args:
        ticker: Stock symbol (e.g. 'NVDA', 'PGR', 'FANG').
        period: Historical lookback period for intraday data (default '5d').
        interval: Bar interval for micro-buckets (default '5m').

    Returns:
        Structured VPIN toxicity report.
    """
    # 1. Primary Attempt: 5-minute Intraday Bars for micro-bucket BVC (Easley et al. 2012)
    try:
        import yfinance as yf

        t_obj = yf.Ticker(ticker)
        df_intraday = t_obj.history(period=period, interval=interval)
        if (
            df_intraday is not None
            and isinstance(df_intraday, pd.DataFrame)
            and len(df_intraday) >= 30
            and "Close" in df_intraday.columns
            and "Volume" in df_intraday.columns
        ):
            res = calculate_vpin(df_intraday, num_buckets=50, rolling_window=20)
            res["ticker"] = ticker
            res["data_source"] = "5m_intraday"
            return res
    except Exception as e:
        logger.debug(
            f"Could not fetch 5m intraday for {ticker} ({e}), falling back to daily bars."
        )

    # 2. Secondary Fallback: Daily Bars
    try:
        daily_period = "3mo" if period in ["5d", "1d"] else period
        df = get_price_history(ticker, period=daily_period, use_cache=True)
        if df is None or df.empty:
            return {
                "ticker": ticker,
                "vpin": 0.35,
                "toxicity_regime": "NO_PRICE_DATA",
                "is_toxic_dumping": False,
                "cro_veto_recommended": False,
                "data_source": "none",
            }
        res = calculate_vpin(df, num_buckets=30, rolling_window=10)
        res["ticker"] = ticker
        res["data_source"] = "daily_fallback"
        return res
    except Exception as e:
        logger.error(f"Error evaluating VPIN toxicity for {ticker}: {e}")
        return {
            "ticker": ticker,
            "vpin": 0.35,
            "toxicity_regime": "ERROR_FALLBACK",
            "is_toxic_dumping": False,
            "cro_veto_recommended": False,
            "data_source": "error_fallback",
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="VPIN Market Microstructure Toxicity Engine"
    )
    parser.add_argument(
        "--ticker", type=str, default="FANG", help="Stock ticker to evaluate"
    )
    args = parser.parse_args()

    report = evaluate_ticker_toxicity(args.ticker)
    print(f"=== VPIN MICROSTRUCTURE TOXICITY REPORT: {args.ticker} ===")
    print(f"VPIN Score:          {report['vpin']:.4f}")
    print(f"Toxicity Regime:     {report['toxicity_regime']}")
    print(f"Toxic Dumping Flag:  {report['is_toxic_dumping']}")
    print(f"CRO Veto Advised:    {report['cro_veto_recommended']}")
