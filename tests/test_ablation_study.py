import pytest
from src.ablation_study import run_committee_ablation_backtest


def test_committee_ablation_study_execution():
    res = run_committee_ablation_backtest(ticker="AAPL", lookback_days=150)

    assert res["ticker"] == "AAPL"
    assert "configurations" in res
    cfgs = res["configurations"]

    assert "full_committee" in cfgs
    assert "minus_forensic" in cfgs
    assert "minus_sentiment" in cfgs
    assert "minus_cro" in cfgs
    assert "technical_only" in cfgs

    for key, c in cfgs.items():
        assert "total_return_pct" in c
        assert "sharpe_ratio" in c
        assert "win_rate_pct" in c
        assert "max_drawdown_pct" in c

    assert "sensitivity_attribution" in res
    assert "summary" in res["sensitivity_attribution"]
