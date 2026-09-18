from unittest.mock import patch

from experiments.full_market_experiment import (
    run_duckdb_benchmark,
    run_security_master_experiment,
    run_sec_crawler_experiment,
    run_sandbox_funnel_experiment,
)
from src.duckdb_engine import DuckDBMarketEngine


def test_run_duckdb_benchmark_in_memory(tmp_path):
    db_path = str(tmp_path / "exp_bench.duckdb")
    engine = DuckDBMarketEngine(db_path=db_path)

    res = run_duckdb_benchmark(engine)
    assert isinstance(res, dict)
    assert "total_bars_scanned" in res
    assert "duckdb_momentum_scan_sec" in res
    assert "vectorized_speedup_factor" in res
    engine.close()


def test_run_security_master_experiment(tmp_path):
    with patch(
        "src.security_master.SECURITY_MASTER_CACHE", str(tmp_path / "sec_cache.csv")
    ):
        res = run_security_master_experiment()
        assert isinstance(res, dict)
        assert "total_exchange_symbols" in res
        assert "tradable_clean_universe" in res
        assert "penny_stock_rejection_rate" in res


def test_run_sec_crawler_experiment():
    with patch(
        "src.sec_crawler.SECCatalystCrawler.load_sec_cik_map",
        return_value={"NVDA": "0001045810"},
    ):
        res = run_sec_crawler_experiment()
        assert isinstance(res, dict)
        assert "total_ciks_mapped" in res
        assert "catalyst_sentiment_tests" in res
        assert len(res["catalyst_sentiment_tests"]) == 3


def test_run_sandbox_funnel_experiment_isolation(tmp_path):
    db_path = str(tmp_path / "sandbox.duckdb")
    engine = DuckDBMarketEngine(db_path=db_path)

    with patch(
        "experiments.full_market_experiment.SANDBOX_DIR",
        str(tmp_path / "sandbox_results"),
    ):
        res = run_sandbox_funnel_experiment(engine)
        assert isinstance(res, dict)
        assert "stage1_screened_candidates" in res
        assert "production_portfolio_preserved" in res
        assert res["production_portfolio_preserved"] is True
    engine.close()
