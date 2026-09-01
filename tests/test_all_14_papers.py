import pytest
import numpy as np
import pandas as pd

# Paper 1: ONS
from src.online_newton_step import OnlineNewtonStepOptimizer

# Paper 2: Boyd Convex SOCP
from src.convex_optimizer import PolyTimeConvexOptimizer

# Paper 3: Almgren-Chriss Optimal Execution
from src.almgren_chriss_execution import calculate_almgren_chriss_trajectory

# Paper 4: Moment-SOS Higher Order Portfolio
from src.moment_sos_portfolio import optimize_higher_order_moments

# Paper 5: Bellman-Ford FX Arbitrage
from src.fx_arbitrage_graph import detect_negative_cycle_arbitrage

# Paper 6: CPH Multi-Agent Committee
from src.agent_committee import ChiefRiskOfficerAgent, TechnicalAlphaAgent

# Paper 7: QuantAgents Autonomous Trading
from src.autonomous_trader import AutonomousTradingEngine

# Paper 8: HedgeAgents Balanced Hedging
from src.hedge_agents import compute_balanced_hedge_allocation

# Paper 9: When Agents Trade Daily Scanner
from src.daily_scanner import run_daily_market_scan

# Paper 10: Deflated Sharpe Ratio
from src.triple_barrier import calculate_deflated_sharpe_ratio

# Paper 11: Triple-Barrier Method
from src.triple_barrier import apply_triple_barrier_labeling

# Paper 12: Hierarchical Risk Parity
from src.portfolio import calculate_hrp_weights

# Paper 13: GNN Supply Chain Contagion
from src.gnn_supply_chain import analyze_supply_chain_spillover

# Paper 14: Fractional Kelly Capital Growth
from src.agent_committee import compute_fractional_kelly_sizing


def test_paper1_online_newton_step():
    ons = OnlineNewtonStepOptimizer(num_assets=2)
    w = ons.step(np.array([1.02, 0.98]))
    assert len(w) == 2
    assert w.sum() == pytest.approx(1.0)


def test_paper2_boyd_convex_optimizer():
    opt = PolyTimeConvexOptimizer()
    alphas = pd.Series([0.10, 0.05], index=["NVDA", "AAPL"])
    cov = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.02]], index=["NVDA", "AAPL"], columns=["NVDA", "AAPL"]
    )
    res = opt.optimize_allocation(alphas, cov)
    assert res["solver_success"] is True


def test_paper3_almgren_chriss_execution():
    res = calculate_almgren_chriss_trajectory(
        total_shares=10000.0, total_time_intervals=5
    )
    assert len(res["holdings_trajectory"]) == 6
    assert len(res["trade_sizes"]) == 5
    assert sum(res["trade_sizes"]) == pytest.approx(10000.0, abs=1.0)


def test_paper4_moment_sos_portfolio():
    rets = pd.DataFrame(
        {"NVDA": [0.01, -0.02, 0.03, 0.01], "AAPL": [0.005, 0.002, -0.01, 0.004]}
    )
    res = optimize_higher_order_moments(rets)
    assert res["solver_success"] is True
    assert res["weights"].sum() == pytest.approx(1.0)


def test_paper5_fx_arbitrage_graph():
    # Triangular pair: USD -> EUR (0.90), EUR -> GBP (0.85), GBP -> USD (1.35)
    # Product: 0.90 * 0.85 * 1.35 = 1.03275 (> 1 => 3.27% arbitrage)
    pairs = [("USD", "EUR", 0.90), ("EUR", "GBP", 0.85), ("GBP", "USD", 1.35)]
    res = detect_negative_cycle_arbitrage(pairs)
    assert res["has_arbitrage_opportunity"] is True
    assert res["implied_risk_free_profit_pct"] > 0.0


def test_paper6_cph_multi_agent_committee():
    cro = ChiefRiskOfficerAgent()
    assert hasattr(cro, "evaluate_and_sign_off")


def test_paper7_quant_agents_trader():
    trader = AutonomousTradingEngine()
    assert hasattr(trader, "broker")


def test_paper8_hedge_agents():
    port_rets = pd.Series([0.02, -0.01, 0.03, 0.01])
    bench_rets = pd.Series([0.01, -0.005, 0.015, 0.005])
    res = compute_balanced_hedge_allocation(port_rets, bench_rets)
    assert "current_portfolio_beta" in res
    assert res["current_portfolio_beta"] > 0.0


def test_paper9_when_agents_trade_scanner(monkeypatch):
    # Fast test verifying scanner logic with mock ticker
    monkeypatch.setattr("os.path.exists", lambda path: False)
    signals = run_daily_market_scan()
    assert isinstance(signals, list)


def test_paper10_deflated_sharpe_ratio():
    rets = pd.Series([0.01, 0.02, -0.005, 0.015, 0.01] * 10)
    res = calculate_deflated_sharpe_ratio(rets, num_trials=10)
    assert res["annualized_sharpe"] > 0.0


def test_paper11_triple_barrier_method():
    df = pd.DataFrame(
        {
            "High": [105.0, 110.0, 115.0, 120.0, 125.0],
            "Low": [95.0, 100.0, 105.0, 110.0, 115.0],
            "Close": [100.0, 105.0, 110.0, 115.0, 120.0],
        }
    )
    res = apply_triple_barrier_labeling(df, max_holding_days=2)
    assert "target_barrier" in res.columns


def test_paper12_hierarchical_risk_parity():
    rets = pd.DataFrame({"A": [0.01, 0.02, -0.01], "B": [0.005, 0.01, -0.005]})
    w = calculate_hrp_weights(rets)
    assert w.sum() == pytest.approx(1.0)


def test_paper13_gnn_supply_chain():
    res = analyze_supply_chain_spillover("NVDA", shock_pct=-5.0)
    assert res["total_impacted_nodes"] > 0


def test_paper14_fractional_kelly_capital_growth():
    res = compute_fractional_kelly_sizing(win_rate=0.55, payoff_ratio=1.75)
    assert isinstance(res, dict)
    assert res["fractional_kelly_pct"] > 0.0
    assert res["fractional_kelly_pct"] <= 15.0
