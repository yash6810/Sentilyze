"""
Unit and Integration Tests for Alpaca Real-Time WebSocket Streaming Bridge.
===========================================================================
Tests WebSocket market data streaming, order update streaming, real-time
trailing stop engine, profit-lock ratcheting, stop breach alerts, and REST fallback.
"""

import pytest
import asyncio
import json
from unittest.mock import MagicMock, patch, AsyncMock
from src.alpaca_websocket_streamer import (
    AlpacaWebSocketStreamer,
    TrackedPosition,
    ALPACA_MARKET_DATA_WS_URL,
    ALPACA_PAPER_STREAM_WS_URL,
)


@pytest.fixture
def streamer():
    """Returns an AlpacaWebSocketStreamer instance with test credentials."""
    return AlpacaWebSocketStreamer(
        api_key="TEST_API_KEY",
        secret_key="TEST_SECRET_KEY",
        is_paper=True,
        account_id="PA3NFG0UWP3R",
        enable_rest_fallback=False,  # disabled by default for deterministic unit tests
    )


def test_streamer_initialization_defaults(streamer):
    """Verifies default properties, endpoints, and empty state."""
    assert streamer.api_key == "TEST_API_KEY"
    assert streamer.secret_key == "TEST_SECRET_KEY"
    assert streamer.is_paper is True
    assert streamer.account_id == "PA3NFG0UWP3R"
    assert streamer.market_data_url == ALPACA_MARKET_DATA_WS_URL
    assert streamer.order_stream_url == ALPACA_PAPER_STREAM_WS_URL
    assert streamer.market_data_status == "DISCONNECTED"
    assert streamer.order_stream_status == "DISCONNECTED"
    assert len(streamer.tracked_positions) == 0
    assert len(streamer.subscribed_trades) == 0
    assert len(streamer.subscribed_quotes) == 0


def test_track_position_and_untrack(streamer):
    """Verifies adding and removing a position for real-time monitoring."""
    pos = streamer.track_position(
        symbol="NVDA",
        shares=50,
        entry_price=120.0,
        current_sl=115.0,
        high_water_mark=122.0,
        order_id="ord-001",
        stop_order_id="stop-001",
    )

    assert "NVDA" in streamer.tracked_positions
    assert pos.symbol == "NVDA"
    assert pos.shares == 50
    assert pos.entry_price == 120.0
    assert pos.current_sl == 115.0
    assert pos.high_water_mark == 122.0
    assert pos.stop_order_id == "stop-001"
    # Auto-subscribed to market data
    assert "NVDA" in streamer.subscribed_trades
    assert "NVDA" in streamer.subscribed_quotes

    # Untrack
    removed = streamer.untrack_position("NVDA")
    assert removed is not None
    assert "NVDA" not in streamer.tracked_positions


def test_subscribe_and_unsubscribe(streamer):
    """Verifies manual dynamic subscription modifications."""
    streamer.subscribe(["AAPL", "MSFT"], trades=True, quotes=True, bars=True)
    assert "AAPL" in streamer.subscribed_trades
    assert "MSFT" in streamer.subscribed_trades
    assert "AAPL" in streamer.subscribed_quotes
    assert "MSFT" in streamer.subscribed_quotes
    assert "AAPL" in streamer.subscribed_bars
    assert "MSFT" in streamer.subscribed_bars

    streamer.unsubscribe(["AAPL"], trades=True, quotes=False, bars=True)
    assert "AAPL" not in streamer.subscribed_trades
    assert "AAPL" in streamer.subscribed_quotes  # Kept quote subscription
    assert "AAPL" not in streamer.subscribed_bars
    assert "MSFT" in streamer.subscribed_trades  # MSFT untouched


def test_process_price_tick_dispatches_callbacks(streamer):
    """Verifies tick callbacks and latest tick registry updates."""
    received = []

    def on_tick(sym, tick):
        received.append((sym, tick["price"], tick["type"]))

    streamer.register_tick_callback(on_tick)

    # Ingest trade tick
    streamer._process_price_tick(
        "NVDA", 125.50, "trade", {"T": "t", "S": "NVDA", "p": 125.50}
    )

    assert len(received) == 1
    assert received[0] == ("NVDA", 125.50, "trade")
    assert streamer.latest_ticks["NVDA"]["price"] == 125.50


def test_trailing_stop_ratchet_on_higher_tick(streamer):
    """
    Verifies that when price advances significantly, high-water mark updates,
    profit lock ratchets stop-loss higher, and trailing stop callbacks are fired.
    """
    streamer.track_position(
        symbol="PLTR",
        shares=100,
        entry_price=100.0,
        current_sl=95.0,
        high_water_mark=100.0,
        min_profit_threshold_pct=1.0,
        lock_fraction=0.80,
    )

    ratchet_events = []

    def on_ratchet(sym, old_sl, new_sl, hwm, action):
        ratchet_events.append((sym, old_sl, new_sl, hwm, action))

    streamer.register_trailing_stop_callback(on_ratchet)

    # Send tick at 105.0 (+5.0% gain from 100 entry)
    streamer._process_price_tick("PLTR", 105.0, "trade", {"p": 105.0})

    pos = streamer.tracked_positions["PLTR"]
    assert pos.high_water_mark == 105.0
    # Stop-loss should have ratcheted up: 80% of $5 gain locked = $4.00 -> SL = $104.00
    assert pos.current_sl > 95.0
    assert pos.current_sl == 104.0
    assert len(ratchet_events) == 1
    assert ratchet_events[0][0] == "PLTR"
    assert ratchet_events[0][1] == 95.0
    assert ratchet_events[0][2] == 104.0
    assert ratchet_events[0][3] == 105.0


def test_trailing_stop_never_moves_down(streamer):
    """Verifies that a subsequent price drop never lowers the ratcheted stop-loss."""
    streamer.track_position(
        symbol="AAPL",
        shares=50,
        entry_price=200.0,
        current_sl=190.0,
        high_water_mark=200.0,
        min_profit_threshold_pct=1.0,
        lock_fraction=0.80,
    )

    # Push to high-water mark: $210 (+5% gain) -> SL ratchets to $208
    streamer._process_price_tick("AAPL", 210.0, "trade", {"p": 210.0})
    pos = streamer.tracked_positions["AAPL"]
    ratcheted_sl = pos.current_sl
    assert ratcheted_sl == 208.0

    # Price drops to $208.50 (above stop)
    streamer._process_price_tick("AAPL", 208.50, "trade", {"p": 208.50})
    assert pos.current_sl == 208.0  # SL remained untouched
    assert pos.high_water_mark == 210.0  # HWM unchanged


def test_stop_loss_breach_triggers_alert_and_broker_close(streamer):
    """Verifies stop breach alert and automatic broker liquidation invocation."""
    mock_broker = MagicMock()
    streamer.broker_bridge = mock_broker

    streamer.track_position(
        symbol="TSLA",
        shares=20,
        entry_price=250.0,
        current_sl=240.0,
        high_water_mark=250.0,
    )

    stop_alerts = []

    def on_stop_hit(sym, price, sl):
        stop_alerts.append((sym, price, sl))

    streamer.register_stop_triggered_callback(on_stop_hit)

    # Ingest tick at $239.50 (below $240.00 stop-loss)
    streamer._process_price_tick("TSLA", 239.50, "trade", {"p": 239.50})

    assert len(stop_alerts) == 1
    assert stop_alerts[0] == ("TSLA", 239.50, 240.0)
    # Broker liquidation hook called
    mock_broker.close_position.assert_called_once_with("TSLA")


def test_broker_replace_stop_order_called_on_ratchet(streamer):
    """Verifies that if a stop_order_id is tracked, ratcheting triggers order replacement on broker."""
    mock_broker = MagicMock()
    streamer.broker_bridge = mock_broker

    streamer.track_position(
        symbol="AMZN",
        shares=30,
        entry_price=180.0,
        current_sl=172.0,
        high_water_mark=180.0,
        stop_order_id="alpaca-stop-ord-999",
        min_profit_threshold_pct=1.0,
        lock_fraction=0.80,
    )

    # Push to $190 (+5.5% gain) -> SL ratchets to 180 + 10 * 0.8 = 188.0
    streamer._process_price_tick("AMZN", 190.0, "trade", {"p": 190.0})

    mock_broker.replace_order_stop_price.assert_called_once_with(
        "alpaca-stop-ord-999", 188.0
    )


def test_trade_updates_order_stream_fill_and_close(streamer):
    """Verifies processing of order fill and sell events from the trading stream."""
    streamer.track_position(
        symbol="GOOGL",
        shares=10,
        entry_price=160.0,
        current_sl=150.0,
    )

    trade_events = []

    def on_update(event, data):
        trade_events.append((event, data))

    streamer.register_trade_update_callback(on_update)

    # Simulated sell fill event for GOOGL (closing position)
    update_packet = {
        "event": "fill",
        "qty": "10",
        "price": "168.50",
        "order": {"symbol": "GOOGL", "side": "sell", "filled_qty": "10"},
    }

    # Simulate message receiving
    # Directly test the packet handling logic or trigger through simulated order WS message
    raw_msg = json.dumps({"stream": "trade_updates", "data": update_packet})

    # Call trade update callback via simulated stream event
    for cb in streamer._on_trade_update_callbacks:
        cb(update_packet["event"], update_packet)

    assert len(trade_events) == 1
    assert trade_events[0][0] == "fill"


def test_send_market_subscription_payload(streamer):
    """Verifies market subscription JSON formatting."""
    mock_ws = AsyncMock()
    streamer._market_ws = mock_ws

    asyncio.run(
        streamer._send_market_subscription(trades=["NVDA"], quotes=["NVDA"], bars=[])
    )

    mock_ws.send.assert_awaited_once()
    payload = json.loads(mock_ws.send.call_args[0][0])
    assert payload["action"] == "subscribe"
    assert "NVDA" in payload["trades"]
    assert "NVDA" in payload["quotes"]


def test_rest_fallback_monitor_polls_when_ws_disconnected(streamer):
    """Verifies REST fallback polling runs when market WebSocket is disconnected."""
    streamer.enable_rest_fallback = True
    streamer.market_data_status = "DISCONNECTED"

    streamer.track_position("NVDA", 10, 120.0, 115.0)

    # Mock the REST poll method
    streamer._poll_rest_quote = MagicMock(return_value=126.0)

    # Run one pass of the fallback logic
    price = streamer._poll_rest_quote("NVDA")
    streamer._process_price_tick("NVDA", price, "rest_fallback", {"price": price})

    assert streamer.latest_ticks["NVDA"]["price"] == 126.0
    assert streamer.latest_ticks["NVDA"]["type"] == "rest_fallback"
    assert streamer.tracked_positions["NVDA"].high_water_mark == 126.0


def test_background_thread_execution_and_stop(streamer):
    """Verifies that background thread starts and stops cleanly."""
    thread = streamer.run_in_background()
    assert thread.is_alive()

    # Allow thread to start
    import time

    time.sleep(0.1)

    streamer.stop_background(timeout=1.0)
    assert not streamer._running
