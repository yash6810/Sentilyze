import pytest
import os
import json
from src.paper_broker import PaperBroker, reconcile_split_if_needed
from src.autonomous_trader import FormalLogicZ3RuleGate
from src.compound_engine import evaluate_pyramiding_step
from src.agent_committee import ChiefRiskOfficerAgent, execute_committee_order


@pytest.fixture
def temp_portfolio_file(tmp_path):
    portfolio_path = tmp_path / "test_hardening_portfolio.json"
    return str(portfolio_path)


def test_crypto_fractional_scale_out(temp_portfolio_file):
    """Verify fractional cryptocurrency lots (e.g. 0.04 BTC) are halved cleanly without integer truncation."""
    broker = PaperBroker(portfolio_path=temp_portfolio_file, initial_cash=100000.0)

    # Directly seed an open crypto position
    broker.state["open_positions"]["BTC-USD"] = {
        "ticker": "BTC-USD",
        "shares": 0.04,
        "entry_price": 60000.0,
        "current_price": 60000.0,
        "entry_date": "2026-10-16",
        "tp1_target": 63000.0,
        "tp2_target": 68000.0,
        "sl_target": 57000.0,
        "scaled_out": False,
    }
    broker.state["cash"] = 50000.0
    broker._save()

    # Price reaches TP1 ($63,500 >= $63,000)
    signals = [
        {
            "ticker": "BTC-USD",
            "signal": "HOLD",
            "confidence": 0.60,
            "current_price": 63500.0,
        }
    ]
    actions = broker.execute_daily_signals(signals)

    assert len(actions["take_profits_tp1"]) == 1
    tp1 = actions["take_profits_tp1"][0]
    assert tp1["shares"] == 0.02
    assert broker.state["open_positions"]["BTC-USD"]["shares"] == 0.02
    assert broker.state["open_positions"]["BTC-USD"]["scaled_out"] is True
    # Verify proceeds: 0.02 * 63500 = $1,270.00 minus 3.5 bps friction ($0.4445)
    gross_proceeds = 0.02 * 63500.0
    friction = gross_proceeds * 0.00035
    assert broker.state["cash"] == pytest.approx(
        50000.0 + gross_proceeds - friction, rel=1e-3
    )
    assert broker.state["realized_pnl"] == pytest.approx(
        0.02 * 3500.0 - friction, rel=1e-3
    )


def test_single_whole_share_scale_out_closes_cleanly(temp_portfolio_file):
    """Verify that a 1-share whole position reaching TP1 exits 100% cleanly without leaving 0 shares behind."""
    broker = PaperBroker(portfolio_path=temp_portfolio_file, initial_cash=100000.0)

    broker.state["open_positions"]["NVDA"] = {
        "ticker": "NVDA",
        "shares": 1,
        "entry_price": 120.0,
        "current_price": 120.0,
        "entry_date": "2026-10-16",
        "tp1_target": 130.0,
        "tp2_target": 145.0,
        "sl_target": 114.0,
        "scaled_out": False,
    }
    broker.state["cash"] = 99880.0
    broker._save()

    signals = [
        {
            "ticker": "NVDA",
            "signal": "HOLD",
            "confidence": 0.60,
            "current_price": 132.0,
        }
    ]
    actions = broker.execute_daily_signals(signals)

    assert len(actions["take_profits_tp1"]) == 1
    assert "NVDA" not in broker.state["open_positions"]
    assert broker.state["winning_trades"] == 1
    gross_pnl = 1 * (132.0 - 120.0)
    friction = 132.0 * 0.00035
    assert broker.state["realized_pnl"] == pytest.approx(gross_pnl - friction, rel=1e-3)


def test_stock_split_reconciliation_forward_and_reverse():
    """Verify 10:1 forward split and 1:5 reverse split reconcile bases and stops seamlessly."""
    # 1. 10:1 Forward Split (e.g. NVDA pre-split $1,200 drops to $120)
    pos_forward = {
        "ticker": "NVDA",
        "shares": 10,
        "entry_price": 1200.0,
        "sl_target": 1140.0,
        "tp1_target": 1300.0,
        "tp2_target": 1450.0,
    }
    spot_forward = 120.0
    reconciled = reconcile_split_if_needed(pos_forward, "NVDA", spot_forward)
    assert reconciled is True
    assert pos_forward["shares"] == 100
    assert pos_forward["entry_price"] == 120.0
    assert pos_forward["sl_target"] == 114.0
    assert pos_forward["tp1_target"] == 130.0
    assert pos_forward["tp2_target"] == 145.0

    # 2. 1:5 Reverse Split (e.g. $10 surges to $50)
    pos_reverse = {
        "ticker": "ACME",
        "shares": 50,
        "entry_price": 10.0,
        "sl_target": 9.0,
        "tp1_target": 12.0,
    }
    spot_reverse = 50.0
    reconciled_rev = reconcile_split_if_needed(pos_reverse, "ACME", spot_reverse)
    assert reconciled_rev is True
    assert pos_reverse["shares"] == 10
    assert pos_reverse["entry_price"] == 50.0
    assert pos_reverse["sl_target"] == 45.0
    assert pos_reverse["tp1_target"] == 60.0


def test_pyramid_unit3_stop_above_blended_entry(temp_portfolio_file):
    """Verify that adding Unit 3 clamps the unified trailing stop strictly >= blended cost basis."""
    broker = PaperBroker(portfolio_path=temp_portfolio_file, initial_cash=100000.0)

    # Buy initial unit (100 shares @ $100)
    broker.execute_daily_signals(
        [
            {
                "ticker": "CAT",
                "signal": "BUY",
                "confidence": 0.80,
                "current_price": 100.0,
                "take_profit": 120.0,
                "stop_loss": 96.0,
            }
        ]
    )
    pos = broker.state["open_positions"]["CAT"]
    pos["shares"] = 100
    pos["entry_price"] = 100.0
    pos["sl_target"] = 96.0
    pos["pyramid_units_filled"] = 1
    broker._save()

    # Step 1: Unit 2 added @ $104
    eval_u1 = evaluate_pyramiding_step(current_price=104.0, position_state=pos, atr=3.0)
    assert eval_u1["action"] == "ADD_UNIT_1"
    res_u2 = broker.execute_pyramid_unit(
        ticker="CAT",
        unit_num=2,
        shares=100,
        price=104.0,
        new_sl=eval_u1["new_sl"],
        reason="ADD_UNIT_2",
    )
    assert res_u2["success"] is True
    assert res_u2["blended_entry"] == 102.0  # (100*100 + 100*104)/200

    # Step 2: Unit 3 added @ $108 (crossed +2.0 ATR = $106)
    pos = broker.state["open_positions"]["CAT"]
    pos["blended_entry"] = 102.0
    eval_u2 = evaluate_pyramiding_step(current_price=108.0, position_state=pos, atr=3.0)
    assert eval_u2["action"] == "ADD_UNIT_2"
    res_u3 = broker.execute_pyramid_unit(
        ticker="CAT",
        unit_num=3,
        shares=100,
        price=108.0,
        new_sl=eval_u2["new_sl"],
        reason="ADD_UNIT_3",
    )
    assert res_u3["success"] is True
    blended_final = res_u3["blended_entry"]  # (200*102 + 100*108)/300 = 104.0
    assert blended_final == 104.0
    # Crucial check: SL must be >= blended_final * 1.002 ($104.21)
    assert res_u3["new_sl"] >= round(blended_final * 1.002, 2)
    assert broker.state["open_positions"]["CAT"]["sl_target"] >= 104.20


def test_cro_double_crash_veto_enforced(monkeypatch):
    """Verify that when double crash exhaustion is detected, the CRO issues a hard VETO."""
    cro = ChiefRiskOfficerAgent()
    mock_reports = [
        {
            "agent_name": "Technical Momentum Specialist",
            "vote": "BUY",
            "conviction_score": 75.0,
        },
        {
            "agent_name": "Sentiment & Alternative Data Specialist",
            "vote": "BUY",
            "conviction_score": 70.0,
        },
    ]

    class MockMiner:
        def analyze_market_cashflow_and_shifts(self, ticker):
            return {
                "market_shift_verdict": "DOUBLE_CRASH_EXHAUSTION",
                "metrics": {"double_crash_probability_pct": 85.0},
            }

    monkeypatch.setattr("src.reddit_comment_miner.RedditCommentMiner", MockMiner)

    # Deliberate with mock double crash exhaustion trigger
    res = cro.evaluate_and_sign_off(
        ticker="SPY",
        spot_price=500.0,
        agent_reports=mock_reports,
        vix_level=16.0,
    )

    assert res["action_code"] == "VETO"
    assert "DOUBLE CRASH EXHAUSTION" in res["final_resolution"]
    assert res["approved_leverage"] == 0.0


def test_single_ticket_concentration_cap_enforced(temp_portfolio_file):
    """Verify execute_committee_order never allocates more than 10% of total portfolio equity to a single ticket."""
    broker = PaperBroker(portfolio_path=temp_portfolio_file, initial_cash=100000.0)

    # Deliberation requesting an excessive 25% Kelly allocation
    delib = {
        "action": "BUY",
        "action_code": "BUY",
        "conviction_score": 85.0,
        "kelly_allocation_pct": 25.0,  # Aggressive sizing request
        "spot_price": 100.0,
        "tp1_target": 110.0,
        "tp2_target": 120.0,
        "sl_target": 95.0,
        "cro_signoff": {"approved": True},
    }

    res = execute_committee_order(
        ticker="AAPL",
        deliberation=delib,
        broker=broker,
    )

    assert res["success"] is True
    # Equity is $100,000. Maximum allowed ticket is 10% = $10,000.
    assert res["cost"] <= 10000.0
    assert res["shares"] <= 100


def test_formal_logic_gate_allows_fractional_crypto():
    """Verify that FormalLogicZ3RuleGate allows fractional crypto lots."""
    gate = FormalLogicZ3RuleGate(min_cash_buffer=1000.0)
    portfolio = {"cash": 50000.0, "total_equity": 50000.0}
    delib = {"tp1_target": 65000.0, "tp2_target": 70000.0, "sl_target": 58000.0}

    # BTC-USD with 0.05 fractional shares ($3,000 cost = 6% < 25% max position cap)
    res = gate.verify_trade(
        ticker="BTC-USD",
        spot_price=60000.0,
        shares=0.05,
        deliberation=delib,
        portfolio_state=portfolio,
    )
    assert res["is_valid"] is True
    assert res["status"] == "SATISFIED"
    assert len(res["violations"]) == 0


def test_post_mortem_learner_nan_inf_immunity(tmp_path):
    """Verify TradePostMortemLearner never diverges or returns NaN/Inf even with corrupted weights/data."""
    from src.post_mortem_learner import TradePostMortemLearner, DEFAULT_WEIGHTS
    import numpy as np

    weights_file = tmp_path / "corrupted_weights.json"
    # Seed file with NaN / Inf
    with open(weights_file, "w") as f:
        json.dump(
            {"Technical Momentum": float("nan"), "FinBERT Sentiment": float("inf")}, f
        )

    learner = TradePostMortemLearner(weights_path=str(weights_file))
    loaded = learner.load_current_weights()
    # Must fallback cleanly to DEFAULT_WEIGHTS
    assert loaded == DEFAULT_WEIGHTS
    assert sum(loaded.values()) == pytest.approx(1.0, rel=1e-3)

    # Compute deltas with NaN and Inf trade returns
    corrupted_trades = [
        {"return_pct": float("nan"), "quality": "WINNER", "pnl": float("nan")},
        {"return_pct": float("inf"), "quality": "SEVERE_LOSS", "pnl": 100.0},
        {"return_pct": -2.5, "quality": "CONTROLLED_LOSS", "pnl": -50.0},
    ]
    deltas = learner.compute_agent_performance_deltas(corrupted_trades)
    for agent, delta in deltas.items():
        assert not np.isnan(delta)
        assert not np.isinf(delta)

    # Update weights bayesian with extreme/invalid deltas
    extreme_deltas = {k: 9999.0 for k in DEFAULT_WEIGHTS}
    extreme_deltas["Technical Momentum"] = float("nan")
    updated = learner.update_weights_bayesian(DEFAULT_WEIGHTS, extreme_deltas)

    assert sum(updated.values()) == pytest.approx(1.0, rel=1e-3)
    for w in updated.values():
        assert 0.10 <= w <= 0.35
        assert not np.isnan(w)
        assert not np.isinf(w)


def test_vpin_toxicity_intraday_fallback(monkeypatch):
    """Verify evaluate_ticker_toxicity falls back gracefully between 5m intraday and daily data."""
    from src.vpin_toxicity import evaluate_ticker_toxicity
    import pandas as pd

    # Mock yfinance intraday to raise an exception, testing the daily fallback route
    class MockTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        def history(self, *args, **kwargs):
            raise ConnectionError("Simulated intraday network failure")

    monkeypatch.setattr("yfinance.Ticker", MockTicker)

    # Mock get_price_history to return synthetic daily bars
    dates = pd.date_range("2026-01-01", periods=60)
    mock_df = pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(60)],
            "High": [102.0 + i for i in range(60)],
            "Low": [98.0 + i for i in range(60)],
            "Close": [101.0 + i for i in range(60)],
            "Volume": [1000000 + i * 1000 for i in range(60)],
        },
        index=dates,
    )
    monkeypatch.setattr(
        "src.vpin_toxicity.get_price_history", lambda *args, **kwargs: mock_df
    )

    res = evaluate_ticker_toxicity("NVDA")
    assert res["ticker"] == "NVDA"
    assert res["data_source"] == "daily_fallback"
    assert "vpin" in res
    assert 0.0 <= res["vpin"] <= 1.0


def test_cro_mahalanobis_anomaly_veto(monkeypatch):
    """Verify ChiefRiskOfficerAgent issues a hard VETO when Mahalanobis Sentinel detects out-of-distribution anomaly."""
    cro = ChiefRiskOfficerAgent()
    mock_reports = [
        {
            "agent_name": "Technical Momentum Specialist",
            "vote": "BUY",
            "conviction_score": 85.0,
        },
        {
            "agent_name": "Sentiment & Alternative Data Specialist",
            "vote": "BUY",
            "conviction_score": 85.0,
        },
    ]

    # Mock evaluate_mahalanobis_sentinel to report a critical multivariate anomaly
    monkeypatch.setattr(
        "src.risk_guardian.evaluate_mahalanobis_sentinel",
        lambda ticker: {
            "status": "SUCCESS",
            "ticker": ticker,
            "mahalanobis_distance": 6.82,
            "p_value": 0.00001,
            "is_anomaly": True,
            "regime_verdict": "SEVERE_REGIME_ANOMALY_UNSEEN_STATE",
        },
    )
    monkeypatch.setattr(
        "src.reddit_comment_miner.RedditCommentMiner.analyze_market_cashflow_and_shifts",
        lambda self, ticker: {
            "market_shift_verdict": "NORMAL_STABLE_FLOW",
            "metrics": {"double_crash_probability_pct": 5.0},
        },
    )

    res = cro.evaluate_and_sign_off(
        ticker="TSLA",
        spot_price=250.0,
        agent_reports=mock_reports,
        vix_level=15.0,
    )

    assert res["action_code"] == "VETO"
    assert "MAHALANOBIS ANOMALY" in res["final_resolution"]
    assert res["mahalanobis_veto_triggered"] is True
    assert res["approved_leverage"] == 0.0
    assert res["kelly_allocation_pct"] == 0.0


def test_symmetric_exit_friction_deducted(temp_portfolio_file):
    """Verify that on trade exit, 3.5 bps spread friction is symmetrically deducted from proceeds and net realized PnL."""
    broker = PaperBroker(portfolio_path=temp_portfolio_file, initial_cash=100000.0)

    # Buy position
    broker.execute_daily_signals(
        [
            {
                "ticker": "MSFT",
                "signal": "BUY",
                "confidence": 0.80,
                "current_price": 100.0,
                "take_profit": 110.0,
                "stop_loss": 90.0,
            }
        ]
    )
    assert "MSFT" in broker.state["open_positions"]
    bought_pos = broker.state["open_positions"]["MSFT"]
    bought_shares = bought_pos["shares"]
    cash_after_buy = broker.state["cash"]

    # Sell all bought shares @ $100.0 with severe breakdown signal (conf 0.20).
    broker.execute_daily_signals(
        [
            {
                "ticker": "MSFT",
                "signal": "SELL",
                "confidence": 0.20,
                "current_price": 100.0,
            }
        ]
    )
    assert "MSFT" not in broker.state["open_positions"]
    gross_proceeds = float(bought_shares * 100.0)
    exit_friction = float(gross_proceeds * 0.00035)
    net_proceeds = gross_proceeds - exit_friction
    assert broker.state["cash"] == pytest.approx(
        cash_after_buy + net_proceeds, rel=1e-3
    )
    assert broker.state["realized_pnl"] == pytest.approx(-exit_friction, rel=1e-3)


def test_scratch_trade_cooldown_quarantine(temp_portfolio_file):
    """Verify that any exit, including scratch (0% return), is quarantined for at least 24h to prevent ping-pong churn."""
    broker = PaperBroker(portfolio_path=temp_portfolio_file, initial_cash=100000.0)

    # Record a scratch exit (0% return) closed today
    from src.market_session import get_current_ny_time

    today_ny = get_current_ny_time().strftime("%Y-%m-%d")

    broker.state["closed_trades"].append(
        {
            "ticker": "NVDA",
            "exit_date": today_ny,
            "return_pct": 0.0,
            "pnl": 0.0,
            "reason": "MODEL_SELL",
        }
    )

    # Immediately try to generate BUY signal on the same day
    actions = broker.execute_daily_signals(
        [
            {
                "ticker": "NVDA",
                "signal": "BUY",
                "confidence": 0.85,
                "current_price": 120.0,
            }
        ]
    )
    # NVDA must NOT be bought because it was quarantined for 24h
    assert "NVDA" not in broker.state["open_positions"]
