import pytest
from src.paper_broker import PaperBroker


@pytest.fixture
def temp_portfolio_file(tmp_path):
    portfolio_path = tmp_path / "test_paper_portfolio.json"
    return str(portfolio_path)


def test_paper_broker_initialization(temp_portfolio_file):
    broker = PaperBroker(portfolio_path=temp_portfolio_file, initial_cash=100000.0)
    summary = broker.get_portfolio_summary()

    assert summary["total_equity"] == 100000.0
    assert summary["cash"] == 100000.0
    assert summary["open_positions_count"] == 0
    assert summary["total_trades"] == 0
    assert summary["realized_pnl"] == 0.0


def test_paper_broker_execute_buy_signals(temp_portfolio_file):
    broker = PaperBroker(portfolio_path=temp_portfolio_file, initial_cash=100000.0)

    signals = [
        {
            "ticker": "AMD",
            "signal": "BUY",
            "confidence": 0.73,
            "current_price": 400.0,
            "take_profit": 450.0,
            "stop_loss": 370.0,
            "regime": "BULLISH",
        },
        {
            "ticker": "TSLA",
            "signal": "BUY",
            "confidence": 0.60,
            "current_price": 200.0,
            "take_profit": 230.0,
            "stop_loss": 185.0,
            "regime": "BULLISH",
        },
    ]

    actions = broker.execute_daily_signals(signals)

    assert len(actions["buys"]) == 2
    assert "AMD" in broker.state["open_positions"]
    assert "TSLA" in broker.state["open_positions"]
    assert broker.state["cash"] < 100000.0
    assert broker.state["total_equity"] == 100000.0

    summary = broker.get_portfolio_summary()
    assert summary["open_positions_count"] == 2


def test_paper_broker_take_profit_exit(temp_portfolio_file):
    broker = PaperBroker(portfolio_path=temp_portfolio_file, initial_cash=100000.0)

    # 1. Buy position
    signals = [
        {
            "ticker": "META",
            "signal": "BUY",
            "confidence": 0.65,
            "current_price": 500.0,
            "take_profit": 550.0,
            "stop_loss": 470.0,
        }
    ]
    broker.execute_daily_signals(signals)
    assert "META" in broker.state["open_positions"]

    # 2. Next day price jumps past TP1 target ($560 >= $550) -> 50% Scale-Out
    next_day_signals = [
        {
            "ticker": "META",
            "signal": "BUY",
            "confidence": 0.65,
            "current_price": 560.0,
            "take_profit": 600.0,
            "stop_loss": 520.0,
        }
    ]
    actions = broker.execute_daily_signals(next_day_signals)

    assert len(actions["take_profits"]) == 1
    assert actions["take_profits"][0]["reason"] == "TAKE_PROFIT"
    assert broker.state["open_positions"]["META"]["scaled_out"] is True
    assert broker.state["realized_pnl"] > 0

    # 3. Day 3 price reaches TP2 runner target ($600 >= $590) -> Complete Exit
    day3_signals = [
        {
            "ticker": "META",
            "signal": "HOLD",
            "confidence": 0.50,
            "current_price": 600.0,
            "take_profit": 650.0,
            "stop_loss": 550.0,
        }
    ]
    actions3 = broker.execute_daily_signals(day3_signals)
    assert len(actions3) > 0
    assert "META" not in broker.state["open_positions"]
    assert broker.state["winning_trades"] >= 1


def test_paper_broker_stop_loss_exit(temp_portfolio_file):
    broker = PaperBroker(portfolio_path=temp_portfolio_file, initial_cash=100000.0)

    # 1. Buy position
    signals = [
        {
            "ticker": "NVDA",
            "signal": "BUY",
            "confidence": 0.55,
            "current_price": 200.0,
            "take_profit": 230.0,
            "stop_loss": 180.0,
        }
    ]
    broker.execute_daily_signals(signals)

    # 2. Next day price drops below Stop-Loss ($175 <= $180)
    next_day_signals = [
        {
            "ticker": "NVDA",
            "signal": "BUY",
            "confidence": 0.55,
            "current_price": 175.0,
            "take_profit": 210.0,
            "stop_loss": 160.0,
        }
    ]
    actions = broker.execute_daily_signals(next_day_signals)

    assert len(actions["stop_losses"]) == 1
    assert actions["stop_losses"][0]["reason"] == "STOP_LOSS"
    assert "NVDA" not in broker.state["open_positions"]
    assert broker.state["realized_pnl"] < 0
    assert broker.state["losing_trades"] == 1


def test_paper_broker_dataframes(temp_portfolio_file):
    broker = PaperBroker(portfolio_path=temp_portfolio_file, initial_cash=100000.0)
    signals = [
        {
            "ticker": "AAPL",
            "signal": "BUY",
            "confidence": 0.58,
            "current_price": 300.0,
            "take_profit": 330.0,
            "stop_loss": 280.0,
            "regime": "BULLISH",
        }
    ]
    broker.execute_daily_signals(signals)

    pos_df = broker.get_open_positions_df()
    assert not pos_df.empty
    assert "AAPL" in pos_df["Ticker"].values

    eq_df = broker.get_equity_curve_df()
    assert not eq_df.empty
    assert "total_equity" in eq_df.columns


def test_hysteresis_shield_prevents_premature_exit(temp_portfolio_file):
    """
    Test that a signal=SELL with confidence=0.50 does NOT prematurely exit
    a fresh position, preserving trades against daily churn.
    """
    broker = PaperBroker(portfolio_path=temp_portfolio_file, initial_cash=100000.0)

    # Day 1: Buy
    broker.execute_daily_signals(
        [
            {
                "ticker": "NVDA",
                "signal": "BUY",
                "confidence": 0.65,
                "current_price": 100.0,
                "take_profit": 115.0,
                "stop_loss": 95.0,
            }
        ]
    )
    assert "NVDA" in broker.state["open_positions"]

    # Day 2: Noise SELL (confidence 0.50, price still above stop loss)
    noise_signals = [
        {
            "ticker": "NVDA",
            "signal": "SELL",
            "confidence": 0.50,
            "current_price": 99.0,
            "take_profit": 115.0,
            "stop_loss": 95.0,
        }
    ]
    actions = broker.execute_daily_signals(noise_signals)
    assert len(actions["sells"]) == 0
    assert "NVDA" in broker.state["open_positions"]


def test_hysteresis_severe_breakdown_exits(temp_portfolio_file):
    """
    Test that a signal=SELL with confidence < 0.40 triggers MODEL_SELL exit.
    """
    broker = PaperBroker(portfolio_path=temp_portfolio_file, initial_cash=100000.0)

    # Buy
    broker.execute_daily_signals(
        [
            {
                "ticker": "NVDA",
                "signal": "BUY",
                "confidence": 0.65,
                "current_price": 100.0,
                "take_profit": 115.0,
                "stop_loss": 95.0,
            }
        ]
    )

    # Severe breakdown (< 0.40)
    breakdown_signals = [
        {
            "ticker": "NVDA",
            "signal": "SELL",
            "confidence": 0.35,
            "current_price": 98.0,
            "take_profit": 115.0,
            "stop_loss": 95.0,
        }
    ]
    actions = broker.execute_daily_signals(breakdown_signals)
    assert len(actions["sells"]) == 1
    assert "NVDA" not in broker.state["open_positions"]
    assert actions["sells"][0]["reason"] == "MODEL_SELL"


def test_quarantine_prevents_recent_exit_rebuy(temp_portfolio_file):
    """
    Test that a ticker closed within the last 3 days is quarantined from immediate re-entry.
    """
    broker = PaperBroker(portfolio_path=temp_portfolio_file, initial_cash=100000.0)

    # Manually add a trade closed yesterday
    broker.state["closed_trades"].append(
        {
            "ticker": "DE",
            "shares": 10,
            "entry_price": 700.0,
            "exit_price": 685.0,
            "entry_date": "2026-09-23",
            "exit_date": "2026-09-24",
            "pnl": -150.0,
            "reason": "STOP_LOSS",
        }
    )

    # Try to buy DE on 2026-09-25 (1 day later)
    buy_signals = [
        {
            "ticker": "DE",
            "signal": "BUY",
            "confidence": 0.85,
            "current_price": 690.0,
            "take_profit": 740.0,
            "stop_loss": 670.0,
        },
        {
            "ticker": "AAPL",
            "signal": "BUY",
            "confidence": 0.80,
            "current_price": 220.0,
            "take_profit": 240.0,
            "stop_loss": 210.0,
        },
    ]

    actions = broker.execute_daily_signals(buy_signals, date_str="2026-09-25")
    # DE should be quarantined; AAPL should be entered
    assert "DE" not in broker.state["open_positions"]
    assert "AAPL" in broker.state["open_positions"]


def test_chandelier_trailing_ratchet_locks_profit(temp_portfolio_file):
    """
    Test that an asset up +3% ratchets stop loss to lock in +1% profit.
    """
    broker = PaperBroker(portfolio_path=temp_portfolio_file, initial_cash=100000.0)

    # Buy at $100 with initial stop at $95
    broker.execute_daily_signals(
        [
            {
                "ticker": "MSFT",
                "signal": "BUY",
                "confidence": 0.70,
                "current_price": 100.0,
                "take_profit": 115.0,
                "stop_loss": 95.0,
            }
        ],
        date_str="2026-09-20",
    )

    # Price surges to $104 (+4% gain)
    broker.execute_daily_signals(
        [
            {
                "ticker": "MSFT",
                "signal": "HOLD",
                "confidence": 0.50,
                "current_price": 104.0,
                "take_profit": 115.0,
                "stop_loss": 95.0,
            }
        ],
        date_str="2026-09-21",
    )

    pos = broker.state["open_positions"]["MSFT"]
    # Stop should be ratcheted to at least $101.00 (+1% profit locked in)
    assert pos["sl_target"] >= 101.0
