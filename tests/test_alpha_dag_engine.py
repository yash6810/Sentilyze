import pytest
from src.alpha_dag_engine import QuantAlphaDAGEngine, AlphaDAGStageContract


def test_alpha_dag_engine_gen1_execution():
    engine = QuantAlphaDAGEngine(
        tickers=["NVDA", "AAPL"], lookback_period="6mo", mode="gen1_naive"
    )
    results = engine.run_dag()

    assert "individual_alphas" in results
    assert "composite_book" in results
    assert "buy_and_hold_benchmark" in results
    assert "lineage_graph" in results
    assert len(results["individual_alphas"]) == 4


def test_alpha_dag_engine_gen2_adaptive_execution():
    engine = QuantAlphaDAGEngine(
        tickers=["NVDA", "AAPL", "MSFT"], lookback_period="6mo", mode="gen2_adaptive"
    )
    results = engine.run_dag()

    assert results["mode"] == "gen2_adaptive"
    assert "dynamic_alpha_allocations" in results
    alloc = results["dynamic_alpha_allocations"]
    assert sum(alloc.values()) == pytest.approx(1.0, rel=1e-2)

    comp = results["composite_book"]
    assert "sharpe_ratio" in comp
    assert "total_return_pct" in comp
    assert len(results["lineage_graph"]) >= 5
