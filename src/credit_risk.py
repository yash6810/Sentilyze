"""
Synthetic CDS & Merton Structural Credit Risk Model (Sprint 2, Module 2.2 / Idea 28)

Inverts the Merton (1974) structural credit risk model:
Treats company equity as a European call option on firm assets V with strike equal to debt D:
E = V * Phi(d1) - D * exp(-r * T) * Phi(d2)
sigma_E * E = Phi(d1) * sigma_V * V

Computes:
1. Implied Firm Asset Value (V) & Asset Volatility (sigma_V)
2. Distance to Default (DD) and 1-Year Default Probability (DP)
3. Synthetic CDS Spread (basis points)
4. Equity-Credit Decoupling Sentinel (detects when equity rallies while balance sheet credit deteriorates)
"""

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import root

from src.data_ingestion import get_price_history
from src.agent_committee import fetch_financial_statements

logger = logging.getLogger("Sentilyze.CreditRisk")
logging.basicConfig(level=logging.INFO)


def solve_merton_model(
    equity_val: float,
    equity_vol: float,
    total_debt: float,
    risk_free_rate: float = 0.045,
    T: float = 1.0,
) -> Dict[str, Any]:
    """
    Solves non-linear 2-equation Merton system for (V, sigma_V).

    Equations:
    1) E - V*Phi(d1) + D*exp(-rT)*Phi(d2) = 0
    2) sigma_E * E - Phi(d1) * sigma_V * V = 0
    """
    E = max(float(equity_val), 1.0)
    sigma_E = max(float(equity_vol), 0.05)
    D = max(float(total_debt), 1.0)
    r = float(risk_free_rate)
    t_sqrt = np.sqrt(T)

    # Initial guess: V0 = E + D, sigma_V0 = sigma_E * (E / (E + D))
    v_init = E + D
    sig_v_init = max(sigma_E * (E / v_init), 0.02)

    def equations(vars):
        v, sig_v = vars[0], max(vars[1], 1e-4)
        if v <= 0 or sig_v <= 0:
            return [1e6, 1e6]

        d1 = (np.log(v / D) + (r + 0.5 * sig_v**2) * T) / (sig_v * t_sqrt)
        d2 = d1 - sig_v * t_sqrt

        phi_d1 = norm.cdf(d1)
        phi_d2 = norm.cdf(d2)

        eq1 = v * phi_d1 - D * np.exp(-r * T) * phi_d2 - E
        eq2 = phi_d1 * sig_v * v - sigma_E * E
        return [eq1, eq2]

    sol = root(equations, [v_init, sig_v_init], method="hybr")

    if sol.success and sol.x[0] > 0 and sol.x[1] > 0:
        v_sol, sig_v_sol = float(sol.x[0]), float(sol.x[1])
    else:
        # Fallback approximation
        v_sol = v_init
        sig_v_sol = sig_v_init

    # Compute Distance to Default (DD) and Default Probability (DP)
    d1 = (np.log(v_sol / D) + (r + 0.5 * sig_v_sol**2) * T) / (sig_v_sol * t_sqrt)
    d2 = d1 - sig_v_sol * t_sqrt
    dd = float(d2)
    default_prob = float(norm.cdf(-dd))

    # Synthetic CDS spread in basis points (ISDA standard: CDS = DP * (1 - Recovery) * 10,000)
    # Senior unsecured corporate debt standard recovery rate R = 0.40 (LGD = 0.60)
    recovery_rate = 0.40
    lgd = 1.0 - recovery_rate
    liquidity_premium_bps = 25.0  # Base market CDS liquidity premium
    cds_spread_bps = (default_prob * lgd * 10000.0) + liquidity_premium_bps
    cds_spread_bps = max(25.0, min(10000.0, round(cds_spread_bps, 1)))

    return {
        "implied_asset_value": round(v_sol, 2),
        "implied_asset_volatility": round(sig_v_sol, 4),
        "distance_to_default": round(dd, 2),
        "default_probability_1y_pct": round(default_prob * 100.0, 3),
        "synthetic_cds_spread_bps": cds_spread_bps,
        "equity_to_debt_ratio": round(E / D, 2),
    }


def audit_equity_credit_decoupling(
    ticker: str,
    risk_free_rate: float = 0.045,
    lookback_days: int = 60,
) -> Dict[str, Any]:
    """
    Evaluates corporate structural credit risk and flags equity-credit divergence.
    """
    # 1. Fetch balance sheet for debt & market cap
    fin_data = fetch_financial_statements(ticker)
    market_cap = float(fin_data.get("market_cap", 0.0))
    if market_cap <= 0:
        quote_p = float(fin_data.get("spot_price", 100.0))
        market_cap = quote_p * 1e8  # 100M shares fallback

    bs = fin_data.get("balance_sheet", {})
    total_debt = float(
        bs.get("Total Debt", bs.get("Long Term Debt", market_cap * 0.35))
    )
    if total_debt <= 0:
        total_debt = market_cap * 0.20  # conservative baseline

    # 2. Fetch price series for annualized equity volatility & momentum
    try:
        df = get_price_history(ticker, period="6mo", use_cache=True)
    except Exception:
        df = pd.DataFrame()

    if not df.empty and len(df) >= 30:
        rets = df["Close"].pct_change().dropna()
        equity_vol = float(rets.std() * np.sqrt(252))
        ret_30d = float(df["Close"].pct_change(min(30, len(df) - 1)).iloc[-1] * 100.0)
    else:
        equity_vol = 0.30
        ret_30d = 0.0

    # 3. Solve Merton structural credit model
    merton_res = solve_merton_model(
        equity_val=market_cap,
        equity_vol=equity_vol,
        total_debt=total_debt,
        risk_free_rate=risk_free_rate,
        T=1.0,
    )

    dd = merton_res["distance_to_default"]
    cds_bps = merton_res["synthetic_cds_spread_bps"]

    # 4. Decoupling & Credit Sentinel logic
    is_decoupling_detected = False
    credit_tier = "INVESTMENT_GRADE"

    if dd > 4.5 and cds_bps < 75.0:
        credit_tier = "AAA_FORTRESS_BALANCE_SHEET"
        verdict = "FORTRESS_CREDIT_QUALITY"
    elif dd >= 3.0:
        credit_tier = "INVESTMENT_GRADE"
        verdict = "HEALTHY_SOLVENCY"
    elif dd >= 1.8:
        credit_tier = "MODERATE_LEVERAGE"
        verdict = "ACCEPTABLE_CREDIT_RISK"
    else:
        credit_tier = "HIGH_DISTRESS_SPECULATIVE"
        verdict = "CRITICAL_DEFAULT_RISK"

    # Flag Divergence: Equity rising while credit is distressed
    if ret_30d > 4.0 and (cds_bps > 250.0 or dd < 2.0):
        is_decoupling_detected = True
        verdict = "EQUITY_CREDIT_DIVERGENCE_BEARISH_TRAP"

    return {
        "status": "SUCCESS",
        "ticker": ticker.upper(),
        "credit_tier": credit_tier,
        "distance_to_default": dd,
        "default_probability_1y_pct": merton_res["default_probability_1y_pct"],
        "synthetic_cds_spread_bps": cds_bps,
        "annualized_equity_volatility": round(equity_vol * 100.0, 1),
        "recent_equity_return_30d_pct": round(ret_30d, 2),
        "is_decoupling_divergence": is_decoupling_detected,
        "credit_health_verdict": verdict,
        "merton_details": merton_res,
    }


def get_synthetic_cds_spread(ticker: str) -> float:
    """Helper returning synthetic CDS spread in basis points."""
    res = audit_equity_credit_decoupling(ticker)
    return float(res.get("synthetic_cds_spread_bps", 100.0))
