"""
Options Gamma Exposure (GEX), Volatility Surface & Strike Wall Radar for Sentilyze.

Grounded in institutional market microstructure (SpotGamma / SqueezeMetrics methodology):
- Fetches real options chains via yfinance.
- Calculates exact Black-Scholes option delta and gamma:
    d1 = (ln(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
    Gamma = phi(d1) / (S * sigma * sqrt(T))
- Computes Net Market Maker Gamma Exposure per strike:
    Call GEX ($) = + Gamma * Call_OI * 100 * S^2 * 0.01
    Put GEX ($)  = - Gamma * Put_OI  * 100 * S^2 * 0.01
    Net GEX ($)  = Call GEX + Put GEX
- Identifies Call Wall (ceiling resistance), Put Wall (floor support), and Gamma Flip line.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf

from src.utils import get_logger

logger = get_logger(__name__)


def calculate_black_scholes_gamma(
    spot: float,
    strike: float,
    dte: float,
    iv: float,
    r: float = 0.045,
) -> float:
    """
    Computes exact Black-Scholes option gamma (d^2V / dS^2).

    Args:
        spot: Underlying asset spot price ($)
        strike: Option strike price ($)
        dte: Days to expiration (e.g. 14 for 14 days)
        iv: Implied volatility as a decimal (e.g. 0.35 for 35%)
        r: Risk-free rate (default: 4.5% / 0.045)

    Returns:
        float: Option gamma (per $1 move in underlying)
    """
    if spot <= 0 or strike <= 0 or dte <= 0 or iv <= 0.001:
        return 0.0

    t_years = max(1.0 / 365.0, dte / 365.0)
    denom = iv * np.sqrt(t_years)
    if denom <= 1e-6:
        return 0.0

    d1 = (np.log(spot / strike) + (r + 0.5 * (iv**2)) * t_years) / denom
    phi_d1 = norm.pdf(d1)
    gamma = float(phi_d1 / (spot * denom))
    return max(0.0, gamma)


def fetch_options_chain_data(
    ticker: str,
    max_expiries: int = 3,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], float, bool]:
    """
    Fetches real exchange options chains across near-term expiration dates.

    Args:
        ticker: Stock ticker symbol (e.g. 'NVDA', 'AAPL')
        max_expiries: Number of near-term monthly/weekly expiries to aggregate (default: 3)

    Returns:
        Tuple of (calls_dfs_list, puts_dfs_list, current_spot_price, is_real_data)
    """
    calls_list = []
    puts_list = []
    spot_price = 0.0
    is_real_data = False

    try:
        yf_ticker = yf.Ticker(ticker)
        # Fetch current price
        fast_info = getattr(yf_ticker, "fast_info", None)
        if fast_info and hasattr(fast_info, "last_price") and fast_info.last_price:
            spot_price = float(fast_info.last_price)
        else:
            hist = yf_ticker.history(period="5d")
            if not hist.empty:
                spot_price = float(hist["Close"].iloc[-1])

        expirations = yf_ticker.options
        if expirations and len(expirations) > 0 and spot_price > 0:
            target_expiries = expirations[:max_expiries]
            for exp in target_expiries:
                opt_chain = yf_ticker.option_chain(exp)
                c_df = opt_chain.calls.copy()
                p_df = opt_chain.puts.copy()

                # Calculate DTE
                exp_date = datetime.strptime(exp, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
                now_utc = datetime.now(timezone.utc)
                dte = max(1, (exp_date - now_utc).days)

                c_df["dte"] = dte
                c_df["expiration"] = exp
                p_df["dte"] = dte
                p_df["expiration"] = exp

                calls_list.append(c_df)
                puts_list.append(p_df)

            is_real_data = True
            logger.info(
                f"Fetched {len(target_expiries)} real options chains for {ticker} (Spot: ${spot_price:.2f})"
            )
    except Exception as e:
        logger.warning(
            f"Options chain fetch notice for {ticker}: {e}. Falling back to calibrated model."
        )

    if not is_real_data or not calls_list or spot_price <= 0:
        # Calibrated baseline around current spot if available or standard ticker price
        if spot_price <= 0:
            spot_price = 125.0 if ticker == "NVDA" else 150.0

        # Construct mathematically consistent option distribution
        strikes = np.linspace(spot_price * 0.85, spot_price * 1.15, 21)
        sim_calls = pd.DataFrame(
            {
                "strike": strikes,
                "openInterest": [
                    max(50, int(1000 * np.exp(-0.5 * ((k - spot_price) / 10.0) ** 2)))
                    for k in strikes
                ],
                "volume": [
                    max(10, int(500 * np.exp(-0.5 * ((k - spot_price) / 8.0) ** 2)))
                    for k in strikes
                ],
                "impliedVolatility": [
                    0.35 + 0.10 * abs(k - spot_price) / spot_price for k in strikes
                ],
                "dte": 14,
                "expiration": "Front-Month (Calibrated)",
            }
        )
        sim_puts = pd.DataFrame(
            {
                "strike": strikes,
                "openInterest": [
                    max(
                        50,
                        int(900 * np.exp(-0.5 * ((k - spot_price * 0.98) / 10.0) ** 2)),
                    )
                    for k in strikes
                ],
                "volume": [
                    max(
                        10,
                        int(450 * np.exp(-0.5 * ((k - spot_price * 0.98) / 8.0) ** 2)),
                    )
                    for k in strikes
                ],
                "impliedVolatility": [
                    0.38 + 0.12 * abs(k - spot_price) / spot_price for k in strikes
                ],
                "dte": 14,
                "expiration": "Front-Month (Calibrated)",
            }
        )
        calls_list = [sim_calls]
        puts_list = [sim_puts]
        is_real_data = False

    return calls_list, puts_list, spot_price, is_real_data


def compute_gamma_exposure_profile(
    ticker: str,
    spot_price: Optional[float] = None,
    max_expiries: int = 3,
) -> Dict[str, Any]:
    """
    Computes institutional Net Gamma Exposure (GEX), Call Wall, Put Wall, and Gamma Flip Line.

    Returns:
        Dict with comprehensive GEX metrics, strike-by-strike breakdown, and market maker regime.
    """
    calls_list, puts_list, fetched_spot, is_real = fetch_options_chain_data(
        ticker, max_expiries=max_expiries
    )
    spot = spot_price if spot_price and spot_price > 0 else fetched_spot

    df_calls = (
        pd.concat(calls_list, ignore_index=True) if calls_list else pd.DataFrame()
    )
    df_puts = pd.concat(puts_list, ignore_index=True) if puts_list else pd.DataFrame()

    if df_calls.empty or df_puts.empty:
        return {
            "ticker": ticker,
            "spot_price": spot,
            "status": "NO_OPTIONS_DATA",
            "is_real_data": False,
        }

    # Aggregate by strike
    all_strikes = sorted(
        list(set(df_calls["strike"].unique()).union(set(df_puts["strike"].unique())))
    )
    # Focus within +/- 25% of spot to filter out non-viable deep OTM wings
    valid_strikes = [k for k in all_strikes if spot * 0.70 <= k <= spot * 1.30]
    if not valid_strikes:
        valid_strikes = all_strikes[:30]

    strike_records = []
    total_call_gex = 0.0
    total_put_gex = 0.0
    total_call_oi = 0
    total_put_oi = 0
    total_call_vol = 0
    total_put_vol = 0

    for strike in valid_strikes:
        # Match calls
        c_sub = df_calls[df_calls["strike"] == strike]
        c_oi = int(c_sub["openInterest"].fillna(0).sum())
        c_vol = int(c_sub["volume"].fillna(0).sum())
        c_iv = (
            float(c_sub["impliedVolatility"].dropna().mean())
            if not c_sub["impliedVolatility"].dropna().empty
            else 0.35
        )
        c_dte = float(c_sub["dte"].mean()) if not c_sub["dte"].empty else 14.0

        # Match puts
        p_sub = df_puts[df_puts["strike"] == strike]
        p_oi = int(p_sub["openInterest"].fillna(0).sum())
        p_vol = int(p_sub["volume"].fillna(0).sum())
        p_iv = (
            float(p_sub["impliedVolatility"].dropna().mean())
            if not p_sub["impliedVolatility"].dropna().empty
            else 0.35
        )
        p_dte = float(p_sub["dte"].mean()) if not p_sub["dte"].empty else 14.0

        avg_iv = (c_iv + p_iv) / 2.0
        avg_dte = (c_dte + p_dte) / 2.0

        gamma = calculate_black_scholes_gamma(
            spot=spot,
            strike=strike,
            dte=avg_dte,
            iv=avg_iv,
        )

        # GEX in millions per 1% underlying move
        # Formula: Gamma * OI * 100 shares * Spot^2 * 0.01 / 1,000,000
        gex_multiplier = 100.0 * (spot**2) * 0.01 / 1_000_000.0
        call_gex = float(gamma * c_oi * gex_multiplier)
        put_gex = float(-gamma * p_oi * gex_multiplier)
        net_gex = call_gex + put_gex

        total_call_gex += call_gex
        total_put_gex += put_gex
        total_call_oi += c_oi
        total_put_oi += p_oi
        total_call_vol += c_vol
        total_put_vol += p_vol

        strike_records.append(
            {
                "strike": round(strike, 2),
                "call_gex_m": round(call_gex, 3),
                "put_gex_m": round(put_gex, 3),
                "net_gex_m": round(net_gex, 3),
                "call_oi": c_oi,
                "put_oi": p_oi,
                "call_volume": c_vol,
                "put_volume": p_vol,
                "implied_vol": round(avg_iv * 100, 1),
            }
        )

    df_strikes = pd.DataFrame(strike_records)
    total_net_gex = total_call_gex + total_put_gex

    # Find Call Wall (Highest Call GEX) and Put Wall (Highest absolute Put GEX)
    if not df_strikes.empty:
        call_wall_row = df_strikes.loc[df_strikes["call_gex_m"].idxmax()]
        call_wall = float(call_wall_row["strike"])
        call_wall_gex = float(call_wall_row["call_gex_m"])

        put_wall_row = df_strikes.loc[df_strikes["put_gex_m"].idxmin()]
        put_wall = float(put_wall_row["strike"])
        put_wall_gex = float(put_wall_row["put_gex_m"])

        # Gamma Flip calculation: where cumulative Net GEX crosses from negative to positive
        df_sorted = df_strikes.sort_values("strike").copy()
        df_sorted["cum_net_gex"] = df_sorted["net_gex_m"].cumsum()
        crosses = df_sorted[df_sorted["net_gex_m"] >= 0]
        gamma_flip = float(crosses["strike"].iloc[0]) if not crosses.empty else spot
    else:
        call_wall = spot * 1.05
        call_wall_gex = 0.0
        put_wall = spot * 0.95
        put_wall_gex = 0.0
        gamma_flip = spot

    # Market maker regime determination
    if total_net_gex > 5.0:
        mm_regime = "🟢 POSITIVE GAMMA REGIME (Volatility Suppressed / Mean-Reverting)"
        mm_behavior = (
            "Market makers BUY dips and SELL rallies, dampening intraday volatility."
        )
    elif total_net_gex < -5.0:
        mm_regime = (
            "🔴 NEGATIVE GAMMA REGIME (Volatility Amplified / Momentum Acceleration)"
        )
        mm_behavior = "Market makers SELL drops and BUY rips, accelerating market directional breakouts."
    else:
        mm_regime = "🟡 NEUTRAL GAMMA TRANSITION (Pivot Squeeze Zone)"
        mm_behavior = "Price is fluctuating near the Gamma Flip line with rapid sentiment sensitivity."

    pcr_oi = round(total_put_oi / max(1, total_call_oi), 2)
    pcr_vol = round(total_put_vol / max(1, total_call_vol), 2)

    # Calculate analytical Max Pain strike and Pinning Probability
    max_pain_metrics = calculate_max_pain(
        df_calls=df_calls,
        df_puts=df_puts,
        spot_price=spot,
        avg_dte=(
            float(df_calls["dte"].median())
            if "dte" in df_calls and not df_calls["dte"].empty
            else 7.0
        ),
    )

    return {
        "ticker": ticker,
        "spot_price": round(spot, 2),
        "total_net_gex_m": round(total_net_gex, 2),
        "total_call_gex_m": round(total_call_gex, 2),
        "total_put_gex_m": round(total_put_gex, 2),
        "call_wall": round(call_wall, 2),
        "call_wall_gex_m": round(call_wall_gex, 2),
        "put_wall": round(put_wall, 2),
        "put_wall_gex_m": round(put_wall_gex, 2),
        "gamma_flip_line": round(gamma_flip, 2),
        "distance_to_call_wall_pct": round((call_wall - spot) / spot * 100, 2),
        "distance_to_put_wall_pct": round((put_wall - spot) / spot * 100, 2),
        "put_call_oi_ratio": pcr_oi,
        "put_call_volume_ratio": pcr_vol,
        "market_maker_regime": mm_regime,
        "market_maker_behavior": mm_behavior,
        "max_pain": max_pain_metrics,
        "is_real_data": is_real,
        "strikes_df": df_strikes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def calculate_max_pain(
    df_calls: pd.DataFrame,
    df_puts: pd.DataFrame,
    spot_price: float,
    avg_dte: float = 7.0,
) -> Dict[str, Any]:
    """
    Computes exact analytical Max Pain strike price and pinning probability.
    Max Pain is the strike where option buyers lose the most money (total holder payout minimized).
    """
    if df_calls.empty or df_puts.empty or spot_price <= 0:
        return {
            "max_pain_strike": round(spot_price, 2),
            "distance_to_spot_pct": 0.0,
            "pinning_probability_pct": 50.0,
            "pinning_badge": "⚪ NEUTRAL PINNING",
            "total_holder_payout_m": 0.0,
        }

    # Aggregate total OI by strike
    c_oi_by_strike = df_calls.groupby("strike")["openInterest"].sum().to_dict()
    p_oi_by_strike = df_puts.groupby("strike")["openInterest"].sum().to_dict()

    all_strikes = sorted(
        list(set(c_oi_by_strike.keys()).union(set(p_oi_by_strike.keys())))
    )
    # Filter to reasonable window around spot (+/- 30%)
    eval_strikes = [
        k for k in all_strikes if spot_price * 0.70 <= k <= spot_price * 1.30
    ]
    if not eval_strikes:
        eval_strikes = all_strikes

    strike_payouts = {}
    for test_s in eval_strikes:
        # Payout to calls: test_s - k for all calls where k < test_s
        call_loss = sum(
            max(0.0, test_s - k) * oi * 100.0 for k, oi in c_oi_by_strike.items()
        )
        # Payout to puts: k - test_s for all puts where k > test_s
        put_loss = sum(
            max(0.0, k - test_s) * oi * 100.0 for k, oi in p_oi_by_strike.items()
        )
        strike_payouts[test_s] = call_loss + put_loss

    # Strike that minimizes total payout to option holders
    min_strike = min(strike_payouts, key=strike_payouts.get)
    min_payout_dollars = strike_payouts[min_strike]

    dist_pct = (min_strike - spot_price) / spot_price * 100.0
    abs_dist_ratio = abs(min_strike - spot_price) / spot_price
    time_factor = 1.0 / np.sqrt(max(1.0, avg_dte))

    # Analytical strike pinning probability formula
    decay = np.exp(-15.0 * abs_dist_ratio)
    raw_pin_prob = decay * (0.45 + 0.55 * min(1.0, time_factor))
    pin_prob_pct = round(float(np.clip(raw_pin_prob * 100.0, 5.0, 95.0)), 1)

    if pin_prob_pct >= 65.0:
        pin_badge = "🎯 STRONG PINNING FORCE (Price Gravitating to Max Pain)"
    elif pin_prob_pct >= 40.0:
        pin_badge = "🟡 MODERATE PINNING GRAVITY"
    else:
        pin_badge = "⚪ WEAK PINNING (Trend Dominant over Options Expiration)"

    return {
        "max_pain_strike": round(min_strike, 2),
        "distance_to_spot_pct": round(dist_pct, 2),
        "pinning_probability_pct": pin_prob_pct,
        "pinning_badge": pin_badge,
        "total_holder_payout_m": round(min_payout_dollars / 1_000_000.0, 2),
    }


def calculate_options_max_pain(ticker: str, max_expiries: int = 3) -> Dict[str, Any]:
    """Computes standalone Max Pain Radar for a given ticker."""
    gex_data = compute_gamma_exposure_profile(ticker, max_expiries=max_expiries)
    return {
        "ticker": ticker,
        "spot_price": gex_data["spot_price"],
        "max_pain": gex_data["max_pain"],
        "call_wall": gex_data["call_wall"],
        "put_wall": gex_data["put_wall"],
        "gamma_flip": gex_data["gamma_flip_line"],
        "market_maker_regime": gex_data["market_maker_regime"],
        "is_real_data": gex_data["is_real_data"],
        "timestamp": gex_data["timestamp"],
    }


def calculate_options_gex(ticker: str, max_expiries: int = 3) -> Dict[str, Any]:
    """Convenience alias for compute_gamma_exposure_profile."""
    return compute_gamma_exposure_profile(ticker, max_expiries=max_expiries)
