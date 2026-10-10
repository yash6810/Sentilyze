"""
Hurst Exponent & Regime-Adaptive Execution Policy Engine for Sentilyze.

Grounded in Mandelbrot (1971), Peters (1994) Fractal Market Analysis,
and institutional options market maker gamma microstructure (SpotGamma / SqueezeMetrics):
1. Vectorized Rescaled Range (R/S) Hurst Exponent Estimation:
   - H > 0.55: Persistent / Trending Regime (Momentum continuation & 0.5N Pyramiding)
   - 0.45 <= H <= 0.55: Random Walk / Brownian Motion
   - H < 0.45: Anti-persistent / Mean-Reverting Regime (Choppy range, tight harvest)
2. Unified Adaptive Execution Selector:
   Synthesizes Hurst Exponent + Options Gamma Exposure (GEX Flip Line, Call/Put Walls)
   + Vaulted Genetic Strategy DNA into an optimal execution directive.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

from src.utils import get_logger
from src.data_ingestion import get_price_history

try:
    from src.options_gex import compute_gamma_exposure_profile
except ImportError:
    compute_gamma_exposure_profile = None

try:
    from src.strategy_incubator import load_strategy_vault
except ImportError:
    load_strategy_vault = None

logger = get_logger(__name__)


def calculate_hurst_exponent(
    series: Union[np.ndarray, pd.Series, List[float]],
    min_window: int = 10,
    max_window: Optional[int] = None,
) -> float:
    """
    Computes the Hurst exponent (H) using vectorized Rescaled Range (R/S) analysis.

    Args:
        series: Price or log-return time series.
        min_window: Minimum sub-period window size (default: 10).
        max_window: Maximum sub-period window size (default: len // 2).

    Returns:
        float: Hurst exponent H clamped in [0.05, 0.95].
               H > 0.55 -> Persistent / Trending
               H < 0.45 -> Anti-persistent / Mean-Reverting
               ~0.50   -> Random Walk
    """
    arr = np.asarray(series, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)

    if n < 30:
        return 0.50  # Neutral baseline for insufficient data

    # Use log returns if price series provided (positive non-stationary prices)
    if np.all(arr > 0) and not np.allclose(arr, arr[0]):
        # Calculate daily log returns
        returns = np.diff(np.log(arr))
    else:
        returns = arr

    n_ret = len(returns)
    if n_ret < 20 or np.allclose(returns, returns[0]):
        return 0.50

    if max_window is None:
        max_window = max(min_window + 5, n_ret // 2)

    max_window = min(max_window, n_ret // 2)
    if min_window >= max_window:
        return 0.50

    # Generate log-spaced window sizes
    window_sizes = np.unique(
        np.logspace(np.log10(min_window), np.log10(max_window), num=12, dtype=int)
    )
    window_sizes = [w for w in window_sizes if w >= min_window and w <= max_window]

    if len(window_sizes) < 3:
        return 0.50

    rs_values = []
    valid_windows = []

    for w in window_sizes:
        num_splits = n_ret // w
        if num_splits < 1:
            continue

        split_rs = []
        for i in range(num_splits):
            segment = returns[i * w : (i + 1) * w]
            mean_val = np.mean(segment)
            std_val = np.std(segment, ddof=1)

            if std_val < 1e-9:
                continue

            # Cumulative deviations from mean
            cum_dev = np.cumsum(segment - mean_val)
            r_range = np.max(cum_dev) - np.min(cum_dev)
            if r_range > 0:
                split_rs.append(r_range / std_val)

        if split_rs:
            rs_values.append(np.mean(split_rs))
            valid_windows.append(w)

    if len(valid_windows) < 3:
        return 0.50

    # Log-log regression: log(R/S) = H * log(tau) + c
    log_w = np.log(valid_windows)
    log_rs = np.log(rs_values)

    try:
        poly = np.polyfit(log_w, log_rs, 1)
        hurst = float(poly[0])
    except Exception:
        hurst = 0.50

    return float(np.clip(hurst, 0.05, 0.95))


def get_ticker_hurst_regime(
    ticker: str,
    df: Optional[pd.DataFrame] = None,
    period: str = "1y",
) -> Dict[str, Any]:
    """
    Evaluates rolling and aggregate Hurst exponent regimes for a specific ticker.

    Returns:
        Dict with current Hurst value, regime label, and quantitative interpretation.
    """
    if df is None or df.empty:
        df = get_price_history(ticker, period=period, use_cache=True)

    if df.empty or len(df) < 30:
        return {
            "ticker": ticker,
            "hurst_exponent": 0.50,
            "regime": "RANDOM_WALK_NEUTRAL",
            "regime_description": "Insufficient price data; defaulting to neutral Brownian motion.",
            "is_trending": False,
            "is_mean_reverting": False,
        }

    close_series = df["Close"].dropna().values
    hurst_full = calculate_hurst_exponent(close_series)

    # 30-day fast rolling Hurst proxy
    recent_series = close_series[-45:] if len(close_series) >= 45 else close_series
    hurst_recent = calculate_hurst_exponent(recent_series)

    # Blended Hurst (60% fast recent, 40% full history)
    blended_h = round(0.60 * hurst_recent + 0.40 * hurst_full, 3)

    if blended_h >= 0.55:
        regime = "TRENDING_PERSISTENT"
        desc = (
            f"Strong momentum persistence (H={blended_h:.3f}). Price trends exhibit positive memory; "
            "optimal for 0.5N Turtle Pyramiding and trailing stop rides."
        )
        is_trending = True
        is_mean_reverting = False
    elif blended_h <= 0.45:
        regime = "MEAN_REVERTING_CHOP"
        desc = (
            f"Anti-persistent mean-reversion (H={blended_h:.3f}). Breakouts decay into chop; "
            "optimal for sniper entries at support and rapid profit harvests."
        )
        is_trending = False
        is_mean_reverting = True
    else:
        regime = "RANDOM_WALK_NEUTRAL"
        desc = (
            f"Geometric Brownian motion (H={blended_h:.3f}). Market is balanced with no dominant "
            "trend persistence; require strict confluence before trading."
        )
        is_trending = False
        is_mean_reverting = False

    return {
        "ticker": ticker,
        "hurst_exponent": blended_h,
        "hurst_full": round(hurst_full, 3),
        "hurst_recent": round(hurst_recent, 3),
        "regime": regime,
        "regime_description": desc,
        "is_trending": is_trending,
        "is_mean_reverting": is_mean_reverting,
        "sample_bars": len(close_series),
    }


def get_adaptive_execution_policy(
    ticker: str,
    spot_price: Optional[float] = None,
    df_history: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Unified Adaptive Execution Selector.
    Synthesizes:
    1. Hurst Exponent Regime (Trending vs Choppy)
    2. Options Market Maker Gamma Exposure (GEX Flip Line, Call Wall, Put Wall)
    3. Vaulted Genetic Strategy Genome (RSI, ATR multiples, Sentiment Gates)

    Outputs an authoritative execution policy governing pyramiding, profit targets,
    and trailing stops for the autonomous trading engine.
    """
    # 1. Evaluate Hurst Regime
    hurst_info = get_ticker_hurst_regime(ticker, df=df_history)
    hurst_val = float(hurst_info.get("hurst_exponent", 0.50))
    is_trending_h = bool(hurst_info.get("is_trending", False))
    is_choppy_h = bool(hurst_info.get("is_mean_reverting", False))

    # 2. Evaluate Options GEX Regime
    gex_data = {}
    total_net_gex = 0.0
    gamma_flip = spot_price or 100.0
    call_wall = (spot_price or 100.0) * 1.05
    put_wall = (spot_price or 100.0) * 0.95
    has_real_options = False

    if compute_gamma_exposure_profile is not None:
        try:
            gex_data = compute_gamma_exposure_profile(ticker, spot_price=spot_price)
            if gex_data.get("status") != "NO_OPTIONS_DATA":
                total_net_gex = float(gex_data.get("total_net_gex", 0.0))
                gamma_flip = float(gex_data.get("gamma_flip", spot_price or 100.0))
                call_wall = float(
                    gex_data.get("call_wall", (spot_price or 100.0) * 1.05)
                )
                put_wall = float(gex_data.get("put_wall", (spot_price or 100.0) * 0.95))
                has_real_options = bool(gex_data.get("is_real_data", False))
        except Exception as e:
            logger.debug(f"Options GEX profile notice for {ticker}: {e}")

    cur_spot = float(spot_price or gex_data.get("spot_price", 100.0))

    # 3. Load Vaulted Genetic Strategy Genome
    genetic_genome = {
        "tp_atr_multiple": 2.50,
        "sl_atr_multiple": 1.25,
        "sentiment_gate": 0.20,
        "rsi_period": 14,
    }
    if load_strategy_vault is not None:
        try:
            vault = load_strategy_vault()
            ticker_vaulted = [
                item
                for item in vault
                if item.get("ticker", "").upper() == ticker.upper()
            ]
            if ticker_vaulted:
                # Pick highest fitness score
                ticker_vaulted.sort(
                    key=lambda x: x.get("fitness_score", 0), reverse=True
                )
                genetic_genome = ticker_vaulted[0].get("genome", genetic_genome)
        except Exception as e:
            logger.debug(f"Genetic vault lookup notice for {ticker}: {e}")

    base_tp_atr = float(genetic_genome.get("tp_atr_multiple", 2.50))
    base_sl_atr = float(genetic_genome.get("sl_atr_multiple", 1.25))

    # 4. Synthesize Execution Mode
    # Negative dealer gamma (GEX < -5.0) or persistent trend (H > 0.55) accelerates momentum
    is_negative_gex = total_net_gex < -5.0
    is_positive_gex = total_net_gex > 5.0

    if is_trending_h or is_negative_gex:
        execution_mode = "RUNNER_PYRAMID"
        pyramiding_enabled = True
        tp_atr_multiple = max(3.0, base_tp_atr * 1.25)
        sl_atr_multiple = max(1.25, base_sl_atr)
        fast_harvest_target_pct = None
        strategy_directive = (
            f"🚀 [TREND RUNNER MODE] Persistence H={hurst_val:.2f} | Net GEX={total_net_gex:+.1f}M. "
            "Enable 0.5N Turtle Pyramiding (+1N, +2N additions). Ratchet stop to Breakeven at +1N. "
            "Let runners ride with Chandelier trailing stop (P_max - 2.5N ATR)."
        )
    elif is_choppy_h and is_positive_gex:
        execution_mode = "MEAN_REVERSION_SNIPE"
        pyramiding_enabled = False
        tp_atr_multiple = min(1.75, base_tp_atr * 0.75)
        sl_atr_multiple = min(1.0, base_sl_atr * 0.85)
        # Target Call Wall or +2.0% quick cash lock
        call_wall_gain_pct = (
            round((call_wall - cur_spot) / cur_spot * 100.0, 2) if cur_spot > 0 else 2.0
        )
        fast_harvest_target_pct = max(1.20, min(3.50, call_wall_gain_pct))
        strategy_directive = (
            f"🎯 [SNIPER HARVEST MODE] Anti-persistent H={hurst_val:.2f} | Positive GEX={total_net_gex:+.1f}M (Volatility Suppressed). "
            f"Pyramiding DISABLED. Enforce fast cash harvest at Call Wall resistance (+{fast_harvest_target_pct:.1f}%). "
            "Move to 100% cash immediately after target hit."
        )
    else:
        execution_mode = "HYBRID_BALANCED"
        pyramiding_enabled = True
        tp_atr_multiple = base_tp_atr
        sl_atr_multiple = base_sl_atr
        fast_harvest_target_pct = 2.50
        strategy_directive = (
            f"⚖️ [HYBRID BALANCED MODE] H={hurst_val:.2f} | Net GEX={total_net_gex:+.1f}M. "
            "Standard progressive profit locks (+3% -> +1% locked, +5% -> +2.5% locked). "
            "Allow single pyramid unit at +1N ATR on confirmed Volume PoC migration."
        )

    return {
        "ticker": ticker,
        "execution_mode": execution_mode,
        "pyramiding_enabled": pyramiding_enabled,
        "tp_atr_multiple": round(tp_atr_multiple, 2),
        "sl_atr_multiple": round(sl_atr_multiple, 2),
        "fast_harvest_target_pct": fast_harvest_target_pct,
        "strategy_directive": strategy_directive,
        "microstructure_signals": {
            "hurst_exponent": hurst_val,
            "hurst_regime": hurst_info.get("regime", "RANDOM_WALK_NEUTRAL"),
            "net_gex_millions": round(total_net_gex, 2),
            "gamma_flip_level": round(gamma_flip, 2),
            "call_wall_resistance": round(call_wall, 2),
            "put_wall_support": round(put_wall, 2),
            "has_real_options_data": has_real_options,
        },
        "genetic_genome_source": genetic_genome,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
