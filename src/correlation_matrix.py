import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from src.config import DATA_DIR
from src.data_ingestion import get_price_history
from src.utils import get_logger

logger = get_logger(__name__)

UNIVERSE_TICKERS = [
    "NVDA", "AAPL", "MSFT", "GOOGL", "META", "TSLA", "AMZN",
    "AVGO", "AMD", "PLTR", "LLY", "QQQ", "SPY", "JPM", "COST", "NFLX", "TSM"
]


def compute_cross_asset_correlation(
    tickers: List[str] = UNIVERSE_TICKERS,
    window_days: int = 90,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Computes cross-asset returns correlation matrix and identifies optimal hedge pairs.

    Args:
        tickers (List[str]): List of asset tickers.
        window_days (int): Lookback rolling window in trading days (default: 90).

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Correlation matrix DataFrame and Hedge Pair analytics.
    """
    close_prices = {}

    for ticker in tickers:
        cache_path = os.path.join(DATA_DIR, f"{ticker}_price_history.csv")
        if os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
                if not df.empty and "Close" in df.columns:
                    close_prices[ticker] = df["Close"]
            except Exception as e:
                logger.warning(f"Error reading cache for {ticker}: {e}")

    if len(close_prices) < 2:
        # Fallback to fetching live
        for ticker in tickers[:7]:
            df = get_price_history(ticker, period="1y")
            if not df.empty and "Close" in df.columns:
                close_prices[ticker] = df["Close"]

    if not close_prices:
        return pd.DataFrame(), {}

    df_prices = pd.DataFrame(close_prices).dropna(how="all").ffill().dropna()
    df_recent = df_prices.tail(window_days)
    df_returns = df_recent.pct_change().dropna()

    corr_matrix = df_returns.corr().round(2)

    # Identify optimal hedge pairs (Lowest / Most Negative correlation)
    hedge_pairs = []
    seen_pairs = set()

    for t1 in corr_matrix.columns:
        for t2 in corr_matrix.columns:
            if t1 != t2:
                pair_key = tuple(sorted([t1, t2]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    c_val = float(corr_matrix.loc[t1, t2])
                    hedge_pairs.append({
                        "Asset A": t1,
                        "Asset B": t2,
                        "Correlation": c_val,
                        "Hedge Quality": "🟢 Excellent Hedge" if c_val < 0.2 else ("🟡 Moderate Hedge" if c_val < 0.5 else "🔴 High Correlation (No Hedge)")
                    })

    hedge_pairs = sorted(hedge_pairs, key=lambda x: x["Correlation"])

    # Macro Regime Analysis
    spy_corr = corr_matrix["SPY"].drop("SPY", errors="ignore").mean() if "SPY" in corr_matrix.columns else 0.5
    regime = "🔥 RISK-ON / HIGH BETA CLUSTER" if spy_corr > 0.65 else ("🛡️ DIVERSIFIED / SECTOR ROTATION" if spy_corr > 0.40 else "⚠️ DE-CORRELATED / REGIME SHIFT")

    analytics = {
        "lookback_days": window_days,
        "assets_count": len(corr_matrix.columns),
        "avg_market_correlation": round(float(spy_corr), 2),
        "macro_regime": regime,
        "top_hedge_pairs": pd.DataFrame(hedge_pairs[:10]),
    }

    return corr_matrix, analytics


def compute_correlation_matrix(
    tickers: List[str] = UNIVERSE_TICKERS,
    window_days: int = 90,
) -> Dict[str, Any]:
    """Convenience wrapper returning correlation matrix and analytics dictionary."""
    matrix, analysis = compute_cross_asset_correlation(tickers=tickers, window_days=window_days)
    return {"matrix": matrix, "analysis": analysis}

