"""
Unit tests for Cloud Multi-Agent Swarm Market Simulator.
Verifies agent committee order book matching, deterministic fallback,
quantitative strategy benchmarks, and strict portfolio preservation.
"""

import os
import json
import pytest
import numpy as np

from src.cloud_market_simulator import (
    CloudMarketSimulator,
    fetch_ticker_snapshot,
    run_cloud_market_simulation,
    SP100_TICKERS,
    SCENARIOS,
    RESULTS_SIM_FILE,
)


def test_simulator_initialization():
    sim = CloudMarketSimulator()
    assert sim.model == "gemini-3.6-flash"
    assert len(SP100_TICKERS) >= 20
    assert "Normal Drift" in SCENARIOS
    assert "Earnings Beat Surprise" in SCENARIOS


def test_fetch_ticker_snapshot():
    snapshot = fetch_ticker_snapshot("SPY")
    assert "ticker" in snapshot
    assert snapshot["ticker"] == "SPY"
    assert snapshot["current_price"] > 0
    assert snapshot["atr_14"] > 0
    assert 0 <= snapshot["rsi_14"] <= 100


def test_deterministic_simulation_fallback():
    sim = CloudMarketSimulator(api_key="")  # Force deterministic engine
    snapshot = {
        "ticker": "AAPL",
        "current_price": 200.0,
        "atr_14": 3.5,
        "rsi_14": 55.0,
        "sma_20": 198.0,
        "sma_50": 195.0,
        "recent_trend": "bullish",
    }
    res = sim._deterministic_simulation_fallback(
        ticker="AAPL",
        snapshot=snapshot,
        scenario_name="Normal Drift",
        scenario=SCENARIOS["Normal Drift"],
        rounds=4,
        seed=42,
    )

    assert "price_path" in res
    assert len(res["price_path"]) == 5  # Initial price + 4 rounds
    assert res["price_path"][0] == 200.0
    assert len(res["rounds"]) == 4
    assert res["execution_mode"] == "deterministic_matching_engine"

    # Verify all 5 agent personas are present in round 1
    r1 = res["rounds"][0]
    agents = r1["agents"]
    assert "Institutional Smart Money" in agents
    assert "Market Maker / HFT" in agents
    assert "Retail Momentum / FOMO" in agents
    assert "Contrarian Short-Seller" in agents
    assert "Chief Risk Officer & Arbitrator" in agents


def test_academic_strategies_evaluation():
    sim = CloudMarketSimulator()
    snapshot = {"current_price": 100.0, "atr_14": 2.0}
    price_path = [100.0, 101.5, 102.8, 104.0, 103.5]
    benchmarks = sim._evaluate_academic_strategies(price_path, snapshot)

    assert "Buy_and_Hold_SPY" in benchmarks
    assert "Hazan_Online_Newton_Step" in benchmarks
    assert "Boyd_Stanford_Convex_SOCP" in benchmarks
    assert "Triple_Barrier_ATR" in benchmarks
    assert "Fractional_Kelly_Growth" in benchmarks
    assert "Sentilyze_Unified_Master" in benchmarks

    # Total return of benchmark: (103.5 - 100) / 100 = 3.5%
    assert benchmarks["Buy_and_Hold_SPY"]["return_pct"] == 3.5
    assert benchmarks["Buy_and_Hold_SPY"]["alpha_vs_benchmark_pct"] == 0.0


def test_end_to_end_simulation_execution():
    res = run_cloud_market_simulation(
        ticker="SPY",
        scenario_name="Earnings Beat Surprise",
        rounds=3,
    )
    assert res["ticker"] == "SPY"
    assert len(res["price_path"]) >= 4
    assert len(res["rounds_detail"]) == 3
    assert res["latency_ms"] > 0
    assert os.path.exists(RESULTS_SIM_FILE)


def test_strict_portfolio_preservation():
    """
    STRICT PORTFOLIO PRESERVATION MANDATE:
    Simulations must NEVER touch results/paper_portfolio.json.
    Cash must remain strictly $159,199.62.
    """
    portfolio_file = os.path.join("results", "paper_portfolio.json")
    assert os.path.exists(portfolio_file)

    with open(portfolio_file, "r") as f:
        port = json.load(f)

    assert port["cash"] == pytest.approx(159199.62, abs=1.0)
    assert port.get("realized_pnl", 0) == pytest.approx(59199.62, abs=1.0)
