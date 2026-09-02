import pytest
from src.strategy_incubator import (
    StrategyGenome,
    evaluate_3zone_robustness,
    breed_strategy_generation,
    load_strategy_vault,
)


def test_strategy_genome_creation_and_mutation():
    g = StrategyGenome.random()
    assert g.rsi_period > 0
    assert g.tp_atr_multiple > 0
    assert g.sl_atr_multiple > 0

    mutated = g.mutate()
    assert mutated.genome_id != g.genome_id or mutated.rsi_period is not None


def test_evaluate_3zone_robustness():
    g = StrategyGenome.random()
    res = evaluate_3zone_robustness(g, ticker="NVDA")

    assert "fitness_score" in res
    assert 0.0 <= res["fitness_score"] <= 100.0
    assert "train_return_pct" in res
    assert "oos_return_pct" in res
    assert "mc_survival_rate_pct" in res
    assert isinstance(res["is_vaulted"], bool)


def test_breed_strategy_generation_fast():
    res = breed_strategy_generation(ticker="NVDA", population_size=6, generations=2)
    assert res["status"] == "INCUBATION_SUCCESS"
    assert len(res["generation_history"]) == 2
    assert "best_strategy" in res
