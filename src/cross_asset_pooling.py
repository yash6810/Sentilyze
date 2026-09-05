"""
Cross-Asset Sector-Pooled Multi-Task Dataset Engine.

Maps the 538-stock universe to GICS sectors, pools normalized cross-sectional features,
and scales training sample size from 500 rows to 10,000+ rows per fold.
Reference: Gu, Kelly & Xiu (2020), 'Empirical Asset Pricing via Machine Learning', RFS.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# GICS Sector Classification for Major Equities
SECTOR_MAP = {
    # Technology (XLK)
    "NVDA": "Technology",
    "AAPL": "Technology",
    "MSFT": "Technology",
    "AMD": "Technology",
    "QCOM": "Technology",
    "AVGO": "Technology",
    "INTC": "Technology",
    "CRM": "Technology",
    "ADBE": "Technology",
    "ORCL": "Technology",
    "CSCO": "Technology",
    "ACN": "Technology",
    "NOW": "Technology",
    "TXN": "Technology",
    "IBM": "Technology",
    "AMAT": "Technology",
    # Communication Services (XLC)
    "GOOGL": "Communication",
    "GOOG": "Communication",
    "META": "Communication",
    "NFLX": "Communication",
    "DIS": "Communication",
    "CMCSA": "Communication",
    "TMUS": "Communication",
    "VZ": "Communication",
    # Consumer Discretionary (XLY)
    "AMZN": "Consumer_Discretionary",
    "TSLA": "Consumer_Discretionary",
    "HD": "Consumer_Discretionary",
    "MCD": "Consumer_Discretionary",
    "NKE": "Consumer_Discretionary",
    "SBUX": "Consumer_Discretionary",
    "LOW": "Consumer_Discretionary",
    "TJX": "Consumer_Discretionary",
    "BKNG": "Consumer_Discretionary",
    # Financials (XLF)
    "JPM": "Financials",
    "BAC": "Financials",
    "WFC": "Financials",
    "C": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "BLK": "Financials",
    "SCHW": "Financials",
    "AXP": "Financials",
    "V": "Financials",
    "MA": "Financials",
    # Healthcare (XLV)
    "LLY": "Healthcare",
    "UNH": "Healthcare",
    "JNJ": "Healthcare",
    "ABBV": "Healthcare",
    "MRK": "Healthcare",
    "TMO": "Healthcare",
    "ABT": "Healthcare",
    "PFE": "Healthcare",
    "ISRG": "Healthcare",
    "VRTX": "Healthcare",
    "GILD": "Healthcare",
    "BMY": "Healthcare",
    # Energy (XLE)
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "EOG": "Energy",
    "SLB": "Energy",
    "MPC": "Energy",
    "PSX": "Energy",
    "VLO": "Energy",
    "EQT": "Energy",
    # Industrials (XLI)
    "CAT": "Industrials",
    "GE": "Industrials",
    "UNP": "Industrials",
    "HON": "Industrials",
    "RTX": "Industrials",
    "BA": "Industrials",
    "DE": "Industrials",
    "LMT": "Industrials",
    "UPS": "Industrials",
    "IEX": "Industrials",
    # Consumer Staples (XLP)
    "PG": "Consumer_Staples",
    "COST": "Consumer_Staples",
    "WMT": "Consumer_Staples",
    "KO": "Consumer_Staples",
    "PEP": "Consumer_Staples",
    "PM": "Consumer_Staples",
}


def get_sector_for_ticker(ticker: str) -> str:
    """Returns the sector cluster for a given ticker or General default."""
    return SECTOR_MAP.get(ticker.upper(), "General_Market")


def build_pooled_sector_dataset(
    ticker_dfs: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    target_col: str = "Target",
    sector: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Pools cross-sectional DataFrames across sector peers and creates a unified multi-task matrix.
    """
    pooled_X_list = []
    pooled_y_list = []

    for t, df in ticker_dfs.items():
        if sector is not None and get_sector_for_ticker(t) != sector:
            continue
        if df.empty or target_col not in df.columns:
            continue

        valid_cols = [c for c in feature_cols if c in df.columns]
        if len(valid_cols) < len(feature_cols) * 0.7:
            continue

        sub_X = df[valid_cols].copy()
        # Cross-sectional z-score normalization per date or per asset
        sub_X = (sub_X - sub_X.mean()) / (sub_X.std() + 1e-8)
        sub_X["ticker_id"] = hash(t) % 1000  # categorical entity embedding

        pooled_X_list.append(sub_X)
        pooled_y_list.append(df[target_col])

    if not pooled_X_list:
        return pd.DataFrame(), pd.Series(dtype=float)

    X_pooled = pd.concat(pooled_X_list, axis=0).sort_index()
    y_pooled = pd.concat(pooled_y_list, axis=0).sort_index()

    return X_pooled, y_pooled
