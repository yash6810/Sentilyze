"""
Unit tests for Synthetic CDS Merton Credit Model (Sprint 2, Module 2.2)
"""

import pytest
import numpy as np
from src.credit_risk import (
    solve_merton_model,
    audit_equity_credit_decoupling,
    get_synthetic_cds_spread,
)


def test_solve_merton_model_healthy_firm():
    # Fortress firm: Equity $100B, Debt $10B, Vol 20%, r = 4%
    res = solve_merton_model(
        equity_val=1e11,
        equity_vol=0.20,
        total_debt=1e10,
        risk_free_rate=0.04,
        T=1.0,
    )
    assert res["implied_asset_value"] > 1e11
    assert res["distance_to_default"] > 4.0
    assert res["default_probability_1y_pct"] < 0.1
    assert res["synthetic_cds_spread_bps"] < 80.0


def test_solve_merton_model_distressed_firm():
    # Distressed firm: Equity $5B, Debt $50B, Vol 60%, r = 4%
    res = solve_merton_model(
        equity_val=5e9,
        equity_vol=0.60,
        total_debt=5e10,
        risk_free_rate=0.04,
        T=1.0,
    )
    assert res["distance_to_default"] < 2.0
    assert res["default_probability_1y_pct"] > 2.0
    assert res["synthetic_cds_spread_bps"] > 200.0


def test_audit_equity_credit_decoupling_smoke():
    res = audit_equity_credit_decoupling("NVDA")
    assert res["status"] == "SUCCESS"
    assert "distance_to_default" in res
    assert "synthetic_cds_spread_bps" in res
    assert "credit_tier" in res
    assert "is_decoupling_divergence" in res


def test_get_synthetic_cds_spread():
    spread = get_synthetic_cds_spread("NVDA")
    assert isinstance(spread, float)
    assert spread > 0.0
