"""
Daniel & Moskowitz (JFE 2016) Volatility-Scaled Multi-Horizon Momentum Engine.
=============================================================================
Eliminates momentum crash risk by scaling exposure inversely with conditional
realized volatility (RiskMetrics EWMA lambda=0.94) across 3 orthogonal horizons:
- Fast Momentum: 21 trading days (1 month, earnings/catalyst drift)
- Medium Momentum: 63 trading days (1 quarter, institutional rebalancing)
- Slow Momentum: 252 trading days skipping the most recent 21 days (structural trend)
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd
import yfinance as yf
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class VolScaledMomentumResult:
    ticker: str
    target_exposure: float  # Range [-2.0, +2.0] leverage units
    composite_zscore: float
    annualized_realized_vol: float
    vol_scaling_multiplier: float
    momentum_regime: str
    momentum_breakdown: Dict[str, float]


class VolScaledMomentumEngine:
    """
    Implements Daniel & Moskowitz (2016) Volatility-Scaled Multi-Horizon Momentum.
    """

    def __init__(
        self,
        target_vol: float = 0.12,
        max_leverage: float = 2.0,
        ewma_lambda: float = 0.94,
    ):
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.decay_lambda = ewma_lambda

    def compute_conditional_volatility(self, returns: pd.Series) -> pd.Series:
        """Computes RiskMetrics EWMA realized volatility with zero lookahead bias."""
        clean_ret = returns.dropna()
        if len(clean_ret) < 5:
            return pd.Series([0.20] * len(returns), index=returns.index)

        var_t = float(clean_ret.iloc[:20].var()) if len(clean_ret) >= 20 else 0.04
        variances = []
        for r in clean_ret:
            var_t = self.decay_lambda * var_t + (1.0 - self.decay_lambda) * (r**2)
            variances.append(var_t)

        ewma_daily = np.sqrt(variances)
        ewma_annual = pd.Series(ewma_daily * np.sqrt(252), index=clean_ret.index).clip(
            lower=0.04, upper=1.50
        )
        return ewma_annual.reindex(returns.index).ffill().fillna(0.20)

    def calculate_signal(
        self,
        ticker: str,
        df_history: Optional[pd.DataFrame] = None,
        lookback_days: int = 450,
    ) -> VolScaledMomentumResult:
        """
        Calculates the composite volatility-scaled momentum position for a given ticker.
        """
        if df_history is not None and not df_history.empty:
            df = df_history.copy()
        else:
            df = yf.download(
                ticker, period=f"{lookback_days}d", progress=False, auto_adjust=True
            )

        if df.empty or len(df) < 260:
            logger.debug(
                f"Insufficient history for {ticker}; using default neutral momentum."
            )
            return VolScaledMomentumResult(
                ticker=ticker,
                target_exposure=0.0,
                composite_zscore=0.0,
                annualized_realized_vol=0.20,
                vol_scaling_multiplier=1.0,
                momentum_regime="INSUFFICIENT_DATA",
                momentum_breakdown={"21d": 0.0, "63d": 0.0, "252d_skip": 0.0},
            )

        close = df["Close"].astype(float)
        daily_ret = close.pct_change()

        # Multi-Horizon Return Metrics
        ret_21 = close.pct_change(21)
        ret_63 = close.pct_change(63)
        # 12M minus 1M skip: close[t-21] / close[t-252] - 1.0
        ret_252_21 = close.shift(21) / (close.shift(252) + 1e-9) - 1.0

        sig_df = pd.DataFrame(index=close.index)
        sig_df["RET_21"] = ret_21
        sig_df["RET_63"] = ret_63
        sig_df["RET_252_21"] = ret_252_21

        # Estimate Conditional Realized Volatility
        ewma_vol = self.compute_conditional_volatility(daily_ret)
        sig_df["EWMA_VOL"] = ewma_vol
        ann_vol = ewma_vol.clip(lower=0.05)

        # Standardize horizon returns by expected horizon volatility (Daniel & Moskowitz 2016)
        sig_df["RET_21_Z"] = sig_df["RET_21"] / (ann_vol * np.sqrt(21.0 / 252.0) + 1e-6)
        sig_df["RET_63_Z"] = sig_df["RET_63"] / (ann_vol * np.sqrt(63.0 / 252.0) + 1e-6)
        sig_df["RET_252_21_Z"] = sig_df["RET_252_21"] / (
            ann_vol * np.sqrt(231.0 / 252.0) + 1e-6
        )

        # Composite Z-Score (adaptive weights based on availability)
        z21 = sig_df["RET_21_Z"].fillna(0.0)
        z63 = sig_df["RET_63_Z"].fillna(0.0)
        z252 = sig_df["RET_252_21_Z"].fillna(0.0)

        sig_df["COMPOSITE_Z"] = 0.35 * z21 + 0.35 * z63 + 0.30 * z252

        # Volatility Scaling Factor
        vol_scalar = np.clip(
            self.target_vol / (sig_df["EWMA_VOL"] + 1e-6),
            0.20,
            self.max_leverage,
        )
        sig_df["VOL_SCALAR"] = vol_scalar

        # Final Bounded Exposure via Hyperbolic Tangent
        sig_df["TARGET_POSITION"] = sig_df["VOL_SCALAR"] * np.tanh(
            sig_df["COMPOSITE_Z"]
        )

        valid_rows = sig_df.dropna(subset=["TARGET_POSITION", "COMPOSITE_Z"])
        if valid_rows.empty:
            valid_rows = sig_df.bfill().ffill().fillna(0.0)

        latest = valid_rows.iloc[-1]
        pos = float(latest["TARGET_POSITION"])
        comp_z = float(latest["COMPOSITE_Z"])
        realized_vol = float(latest["EWMA_VOL"])
        scalar = float(latest["VOL_SCALAR"])

        if pos > 0.5:
            regime = "STRONG_BULLISH_MOMENTUM"
        elif pos > 0.15:
            regime = "MODERATE_BULLISH_MOMENTUM"
        elif pos < -0.5:
            regime = "DEFENSIVE_SHORT_MOMENTUM"
        elif pos < -0.15:
            regime = "MODERATE_BEARISH_MOMENTUM"
        else:
            regime = "NEUTRAL_TRANSITION"

        return VolScaledMomentumResult(
            ticker=ticker,
            target_exposure=round(pos, 2),
            composite_zscore=round(comp_z, 2),
            annualized_realized_vol=round(realized_vol, 3),
            vol_scaling_multiplier=round(scalar, 2),
            momentum_regime=regime,
            momentum_breakdown={
                "21d": round(float(latest["RET_21"]) * 100, 2),
                "63d": round(float(latest["RET_63"]) * 100, 2),
                "252d_skip": round(float(latest["RET_252_21"]) * 100, 2),
            },
        )
