"""
Unit tests for src/master_loop.py.
Verifies the 4-phase MasterTradingLoop lifecycle:
1. Pre-Market Reconnaissance
2. Market Open Whipsaw Shield
3. Intraday Execution & Council Arbitration
4. Post-Market Autopsy & Calibration
Uses mock broker and isolated tmp_path to strictly protect live portfolio state.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.master_loop import MasterTradingLoop
from src.paper_broker import PaperBroker


@pytest.fixture
def isolated_loop(tmp_path):
    p_file = str(tmp_path / "mock_master_portfolio.json")
    t_file = str(tmp_path / "mock_master_trades.csv")
    broker = PaperBroker(portfolio_path=p_file, trades_path=t_file)
    broker.state = {
        "cash": 100000.0,
        "total_equity": 100000.0,
        "realized_pnl": 0.0,
        "open_positions": {},
    }
    mock_alpaca = MagicMock()
    loop = MasterTradingLoop(
        broker=broker,
        portfolio_path=p_file,
        trades_path=t_file,
        alpaca_bridge=mock_alpaca,
    )
    return loop


def test_master_loop_premarket_phase(isolated_loop):
    with patch.object(
        isolated_loop.institutional_gateway,
        "get_macro_regime",
        return_value={"regime": "NEUTRAL", "vix": 16.0, "yield_spread": 0.20},
    ):
        with patch(
            "src.master_loop.evaluate_macro_volatility_blackout",
            return_value={"is_blackout_active": False},
        ):
            with patch.object(
                isolated_loop.sec_crawler, "fetch_recent_8k_filings", return_value=[]
            ):
                with patch(
                    "src.master_loop.compile_biotech_catalyst_radar", return_value={}
                ):
                    with patch.object(
                        isolated_loop.engine,
                        "run_premarket_briefing",
                        return_value={"status": "OK"},
                    ):
                        res = isolated_loop.run_premarket_phase(watchlist=["NVDA"])
                        assert "phase" in res
                        assert res["phase"] == "PRE_MARKET_RECONNAISSANCE"
                        assert "macro_regime" in res
                        assert "morning_briefing" in res


def test_master_loop_opening_shield_phase(isolated_loop):
    res = isolated_loop.run_opening_shield_phase(bypass_time_check=True)
    assert "phase" in res
    assert res["phase"] == "MARKET_OPEN_WHIPSAW_SHIELD"
    assert res["shield_active"] is True


def test_master_loop_intraday_phase(isolated_loop):
    with patch(
        "src.master_loop.fetch_live_quote",
        return_value={"price": 130.0, "change_pct": 1.2},
    ):
        with patch("src.master_loop.is_kill_switch_active", return_value=False):
            with patch(
                "src.master_loop.check_daily_loss_circuit_breaker", return_value=False
            ):
                with patch(
                    "src.master_loop.convene_trading_committee",
                    return_value={"verdict": "HOLD", "conviction": 0.50, "votes": {}},
                ):
                    res = isolated_loop.run_intraday_phase(
                        candidate_tickers=["NVDA"], max_candidates=1
                    )
                    assert "phase" in res
                    assert res["phase"] == "INTRADAY_EXECUTION"


def test_master_loop_postmarket_phase(isolated_loop):
    with patch(
        "src.master_loop.run_autonomous_trade_autopsy",
        return_value={"total_trades": 5, "win_rate_pct": 80.0},
    ):
        with patch.object(
            isolated_loop.engine,
            "_run_self_improvement_feedback_loop",
            return_value={"status": "OK"},
        ):
            res = isolated_loop.run_postmarket_phase()
            assert "phase" in res
            assert res["phase"] == "POST_MARKET_AUTOPSY"
            assert "final_equity" in res


def test_master_loop_full_daily_cycle(isolated_loop):
    with patch.object(
        isolated_loop,
        "run_premarket_phase",
        return_value={"phase": "PRE_MARKET_RECONNAISSANCE"},
    ):
        with patch.object(
            isolated_loop,
            "run_opening_shield_phase",
            return_value={"phase": "MARKET_OPEN_WHIPSAW_SHIELD"},
        ):
            with patch.object(
                isolated_loop,
                "run_intraday_phase",
                return_value={"phase": "INTRADAY_EXECUTION"},
            ):
                with patch.object(
                    isolated_loop,
                    "run_postmarket_phase",
                    return_value={
                        "phase": "POST_MARKET_AUTOPSY",
                        "final_equity": 100000.0,
                    },
                ):
                    summary = isolated_loop.run_full_daily_cycle(
                        watchlist=["NVDA"], dry_run=True, ignore_market_hours=True
                    )
                    assert "cycle_timestamp" in summary
                    assert summary["dry_run"] is True
                    assert "phase_1_premarket" in summary
                    assert "phase_2_opening_shield" in summary
                    assert "phase_3_intraday" in summary
                    assert "phase_4_postmarket" in summary
