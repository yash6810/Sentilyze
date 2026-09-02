"""
Unit tests for Papers 15-24: Lightweight Safety Algorithms.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cusum_detector import CUSUMDetector
from src.ewma_monitor import EWMACorrelationMonitor
from src.grossman_zhou import grossman_zhou_allocation
from src.page_hinkley import PageHinkleyDetector
from src.hmm_regime import GaussianHMMRegimeDetector
from src.cppi_insurance import calculate_cppi_allocation, run_cppi_backtest
from src.adwin_detector import ADWINDetector
from src.risk_constrained_kelly import risk_constrained_kelly_allocation
from src.cdar_optimizer import calculate_cdar, optimize_cdar_portfolio
from src.dcc_correlation import DCCCorrelation


# ── Paper 16: CUSUM ──────────────────────────────────────────────


class TestCUSUM:
    def test_no_alarm_on_stable_stream(self):
        det = CUSUMDetector(threshold_h=5.0, drift_k=0.5)
        np.random.seed(42)
        stable = np.random.normal(0, 0.01, 100)
        results = det.update_batch(stable)
        alarms = [r for r in results if r["alarm"]]
        assert len(alarms) == 0

    def test_alarm_on_mean_shift(self):
        det = CUSUMDetector(threshold_h=3.0, drift_k=0.5, target_mean=0.0)
        # Stable then sudden shift
        stream = np.concatenate([np.zeros(50), np.ones(50) * 2.0])
        results = det.update_batch(stream)
        alarms = [r for r in results if r["alarm"]]
        assert len(alarms) > 0
        assert alarms[0]["direction"] == "UP"

    def test_reset_clears_state(self):
        det = CUSUMDetector()
        det.update(5.0)
        det.reset()
        assert det.s_pos == 0.0
        assert det.s_neg == 0.0
        assert det.n_observations == 0


# ── Paper 17: EWMA ──────────────────────────────────────────────


class TestEWMA:
    def test_initialization(self):
        np.random.seed(42)
        returns_df = pd.DataFrame(
            np.random.normal(0, 0.01, (50, 3)),
            columns=["A", "B", "C"],
        )
        monitor = EWMACorrelationMonitor(decay_lambda=0.94)
        monitor.initialize(returns_df)
        assert monitor.initialized is True
        assert len(monitor.tickers) == 3

    def test_update_returns_correlation(self):
        np.random.seed(42)
        returns_df = pd.DataFrame(
            np.random.normal(0, 0.01, (50, 3)),
            columns=["A", "B", "C"],
        )
        monitor = EWMACorrelationMonitor(decay_lambda=0.94)
        monitor.initialize(returns_df)
        result = monitor.update(np.array([0.01, -0.005, 0.002]))
        assert "avg_pairwise_correlation" in result
        assert "correlation_breakdown_alert" in result

    def test_high_correlation_triggers_alert(self):
        # All assets move identically -> correlation = 1.0
        monitor = EWMACorrelationMonitor(correlation_alert_threshold=0.5)
        seed = pd.DataFrame(
            np.column_stack([np.random.normal(0, 0.01, 30)] * 3),
            columns=["A", "B", "C"],
        )
        monitor.initialize(seed)
        # Feed identical returns
        for _ in range(100):
            r = np.random.normal(0.01, 0.001)
            result = monitor.update(np.array([r, r, r]))
        assert result["correlation_breakdown_alert"] is True


# ── Paper 18: Grossman-Zhou ──────────────────────────────────────


class TestGrossmanZhou:
    def test_at_peak_full_allocation(self):
        result = grossman_zhou_allocation(
            current_wealth=100000.0,
            running_max_wealth=100000.0,
            max_drawdown_tolerance=0.15,
        )
        assert result["risky_weight"] > 0.0
        assert result["at_floor"] is False

    def test_at_floor_zero_allocation(self):
        result = grossman_zhou_allocation(
            current_wealth=85000.0,
            running_max_wealth=100000.0,
            max_drawdown_tolerance=0.15,
        )
        assert result["risky_weight"] == 0.0
        assert result["at_floor"] is True

    def test_partial_drawdown(self):
        result = grossman_zhou_allocation(
            current_wealth=92000.0,
            running_max_wealth=100000.0,
            max_drawdown_tolerance=0.15,
        )
        assert 0.0 < result["risky_weight"] < 1.0
        assert result["floor"] == 85000.0


# ── Paper 22: Page-Hinkley ───────────────────────────────────────


class TestPageHinkley:
    def test_no_drift_on_stable(self):
        det = PageHinkleyDetector(threshold_lambda=50.0)
        np.random.seed(42)
        stable = np.random.normal(0, 0.01, 200)
        results = det.update_batch(stable)
        drifts = [r for r in results if r["drift_detected"]]
        assert len(drifts) == 0

    def test_detects_mean_shift(self):
        det = PageHinkleyDetector(threshold_lambda=20.0, min_magnitude_delta=0.001)
        stream = np.concatenate([np.zeros(100), np.ones(100) * 0.5])
        results = det.update_batch(stream)
        drifts = [r for r in results if r["drift_detected"]]
        assert len(drifts) > 0

    def test_reset(self):
        det = PageHinkleyDetector()
        det.update(1.0)
        det.reset()
        assert det.n == 0


# ── Paper 15: HMM ───────────────────────────────────────────────


class TestHMM:
    def test_classify_returns_regime(self):
        det = GaussianHMMRegimeDetector(n_states=3)
        result = det.update(0.001)
        assert result["regime"] in ["Bull", "Normal", "Crisis"]
        assert 0.0 <= result["crisis_probability"] <= 1.0

    def test_crisis_on_large_negative(self):
        det = GaussianHMMRegimeDetector(n_states=3)
        # Feed large negative returns
        for _ in range(30):
            result = det.update(-0.03)
        assert result["regime"] == "Crisis"
        assert result["is_crisis"] is True

    def test_bull_on_positive_returns(self):
        det = GaussianHMMRegimeDetector(n_states=3)
        for _ in range(30):
            result = det.update(0.005)
        assert result["regime"] == "Bull"

    def test_classify_series(self):
        det = GaussianHMMRegimeDetector()
        np.random.seed(42)
        rets = np.random.normal(0, 0.01, 100)
        df = det.classify_series(rets)
        assert len(df) == 100
        assert "regime" in df.columns


# ── Paper 20: CPPI ──────────────────────────────────────────────


class TestCPPI:
    def test_allocation_above_floor(self):
        result = calculate_cppi_allocation(
            portfolio_value=100000.0,
            floor_value=80000.0,
            multiplier=3.0,
        )
        assert result["risky_weight"] > 0.0
        assert result["cushion"] == 20000.0

    def test_allocation_at_floor(self):
        result = calculate_cppi_allocation(
            portfolio_value=80000.0,
            floor_value=80000.0,
            multiplier=3.0,
        )
        assert result["risky_weight"] == 0.0
        assert result["at_floor"] is True

    def test_backtest_floor_respected(self):
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.015, 252)
        result = run_cppi_backtest(returns, floor_pct=0.85)
        assert result["final_value"] >= 85000.0 * 0.99
        assert not result["floor_breached"]


# ── Paper 21: ADWIN ─────────────────────────────────────────────


class TestADWIN:
    def test_no_drift_on_stable(self):
        det = ADWINDetector(confidence_delta=0.002)
        np.random.seed(42)
        stable = np.random.normal(0, 0.01, 200)
        results = det.update_batch(stable)
        drifts = [r for r in results if r["drift_detected"]]
        # Stable data should produce few or no drifts
        assert len(drifts) < 5

    def test_detects_shift(self):
        det = ADWINDetector(confidence_delta=0.01)
        np.random.seed(42)
        stream = np.concatenate(
            [
                np.random.normal(0, 0.01, 200),
                np.random.normal(1.0, 0.01, 200),
            ]
        )
        results = det.update_batch(stream)
        drifts = [r for r in results if r["drift_detected"]]
        assert len(drifts) > 0


# ── Paper 23: Risk-Constrained Kelly ─────────────────────────────


class TestRiskKelly:
    def test_solver_converges(self):
        np.random.seed(42)
        d = 5
        mu = np.random.uniform(0.05, 0.15, d)
        cov = np.eye(d) * 0.04
        result = risk_constrained_kelly_allocation(mu, cov)
        assert result["solver_converged"] is True
        assert result["log_growth_rate"] > 0.0

    def test_weights_sum_leq_1(self):
        np.random.seed(42)
        d = 5
        mu = np.random.uniform(0.05, 0.15, d)
        cov = np.eye(d) * 0.04
        result = risk_constrained_kelly_allocation(mu, cov, max_leverage=1.0)
        assert result["weights"].sum() <= 1.01

    def test_variance_within_budget(self):
        np.random.seed(42)
        d = 3
        mu = np.array([0.10, 0.08, 0.12])
        cov = np.eye(3) * 0.04
        result = risk_constrained_kelly_allocation(mu, cov)
        assert result["variance_used_pct"] <= 100.5


# ── Paper 19: CDaR ──────────────────────────────────────────────


class TestCDaR:
    def test_cdar_positive(self):
        np.random.seed(42)
        rets = np.random.normal(0, 0.01, 252)
        cdar = calculate_cdar(rets, alpha=0.05)
        assert cdar >= 0.0

    def test_optimize_weights_sum_to_1(self):
        np.random.seed(42)
        df = pd.DataFrame(
            np.random.normal(0.0003, 0.01, (252, 4)),
            columns=["A", "B", "C", "D"],
        )
        result = optimize_cdar_portfolio(df)
        assert abs(result["weights"].sum() - 1.0) < 0.01


# ── Paper 24: DCC ───────────────────────────────────────────────


class TestDCC:
    def test_fit_returns_correlation(self):
        np.random.seed(42)
        df = pd.DataFrame(
            np.random.normal(0, 0.01, (200, 3)),
            columns=["X", "Y", "Z"],
        )
        model = DCCCorrelation()
        result = model.fit(df)
        assert "final_avg_pairwise_correlation" in result
        assert result["n_observations"] == 200

    def test_correlated_assets_high_dcc(self):
        np.random.seed(42)
        base = np.random.normal(0, 0.01, 200)
        df = pd.DataFrame(
            {
                "A": base + np.random.normal(0, 0.001, 200),
                "B": base + np.random.normal(0, 0.001, 200),
            }
        )
        model = DCCCorrelation()
        result = model.fit(df)
        assert result["final_avg_pairwise_correlation"] > 0.5
