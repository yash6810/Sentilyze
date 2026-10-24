"""
Alpaca Real-Time WebSocket Tick & Fill Bridge for Sentilyze.
=============================================================
Institutional-grade WebSocket streaming engine implementing:
1. Alpaca Market Data Stream: wss://stream.data.alpaca.markets/v2/iex
   - Subscribes to real-time quotes, trades, and minute bars for active positions and watchlists.
2. Alpaca Order Updates Stream: wss://paper-api.alpaca.markets/stream
   - Subscribes to live trade_updates (fill, partial_fill, canceled, rejected) for Account PA3NFG0UWP3R.
3. Real-Time Trailing Stop Engine:
   - Evaluates incoming ticks against position high-water marks.
   - Instantly ratchets stop-loss floors higher using structural profit lock logic without cron latency.
   - Triggers rapid stop-loss alerts and liquidation hooks when price breaches stop floor.
4. Resiliency & High Availability:
   - Ping/pong heartbeat health checks.
   - Exponential backoff auto-reconnection (1s to 30s).
   - Dynamic subscription state synchronization upon reconnection.
   - Fallback REST quote polling when WebSocket disconnects.
"""

from typing import Dict, List, Set, Any, Optional, Callable, Union
import os
import sys
import json
import time
import asyncio
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from src.utils import get_logger
from src.smart_trader_engine import apply_high_watermark_profit_lock

logger = get_logger(__name__)

# Default Alpaca WebSocket Endpoints
ALPACA_MARKET_DATA_WS_URL = "wss://stream.data.alpaca.markets/v2/iex"
ALPACA_PAPER_STREAM_WS_URL = "wss://paper-api.alpaca.markets/stream"
ALPACA_LIVE_STREAM_WS_URL = "wss://api.alpaca.markets/stream"


@dataclass
class TrackedPosition:
    """Represents an active position tracked by the real-time trailing stop engine."""

    symbol: str
    shares: float
    entry_price: float
    current_sl: float
    high_water_mark: float
    order_id: Optional[str] = None
    stop_order_id: Optional[str] = None
    min_profit_threshold_pct: float = 1.2
    lock_fraction: float = 0.80
    last_price: float = 0.0
    last_action: str = "INITIAL_STOP"
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class AlpacaWebSocketStreamer:
    """
    Institutional Real-Time WebSocket Streaming & Execution Bridge.
    Connects to both Alpaca Market Data and Trading Order streams simultaneously.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        is_paper: bool = True,
        account_id: str = "PA3NFG0UWP3R",
        market_data_url: Optional[str] = None,
        order_stream_url: Optional[str] = None,
        broker_bridge: Optional[Any] = None,
        rest_fallback_interval: float = 5.0,
        enable_rest_fallback: bool = True,
    ):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "")
        self.is_paper = is_paper
        self.account_id = account_id

        self.market_data_url = market_data_url or os.getenv(
            "ALPACA_MARKET_DATA_WS_URL", ALPACA_MARKET_DATA_WS_URL
        )
        default_order_url = (
            ALPACA_PAPER_STREAM_WS_URL if is_paper else ALPACA_LIVE_STREAM_WS_URL
        )
        self.order_stream_url = order_stream_url or os.getenv(
            "ALPACA_ORDER_STREAM_WS_URL", default_order_url
        )

        self.broker_bridge = broker_bridge
        self.rest_fallback_interval = rest_fallback_interval
        self.enable_rest_fallback = enable_rest_fallback

        # Subscriptions and tracking state
        self.subscribed_trades: Set[str] = set()
        self.subscribed_quotes: Set[str] = set()
        self.subscribed_bars: Set[str] = set()
        self.tracked_positions: Dict[str, TrackedPosition] = {}
        self.latest_ticks: Dict[str, Dict[str, Any]] = {}

        # Connection status: "DISCONNECTED", "CONNECTING", "AUTHENTICATED", "STREAMING", "ERROR"
        self.market_data_status: str = "DISCONNECTED"
        self.order_stream_status: str = "DISCONNECTED"
        self._running: bool = False

        # WebSockets references
        self._market_ws: Optional[websockets.WebSocketClientProtocol] = None
        self._order_ws: Optional[websockets.WebSocketClientProtocol] = None

        # Callbacks
        self._on_tick_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        self._on_trailing_stop_callbacks: List[
            Callable[[str, float, float, float, str], None]
        ] = []
        self._on_stop_triggered_callbacks: List[Callable[[str, float, float], None]] = (
            []
        )
        self._on_trade_update_callbacks: List[Callable[[str, Dict[str, Any]], None]] = (
            []
        )
        self._on_status_change_callbacks: List[Callable[[str, str], None]] = []

        # Threading / event loop management
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = asyncio.Lock() if sys.version_info >= (3, 11) else None

    # =========================================================================
    # Callback Registration
    # =========================================================================

    def register_tick_callback(
        self, callback: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """Registers a callback invoked on each trade/quote tick: cb(symbol, tick_data)."""
        self._on_tick_callbacks.append(callback)

    def register_trailing_stop_callback(
        self, callback: Callable[[str, float, float, float, str], None]
    ) -> None:
        """
        Registers a callback invoked when a trailing stop ratchets higher:
        cb(symbol, old_sl, new_sl, high_water_mark, action)
        """
        self._on_trailing_stop_callbacks.append(callback)

    def register_stop_triggered_callback(
        self, callback: Callable[[str, float, float], None]
    ) -> None:
        """
        Registers a callback invoked when price breaches stop-loss:
        cb(symbol, trigger_price, stop_loss)
        """
        self._on_stop_triggered_callbacks.append(callback)

    def register_trade_update_callback(
        self, callback: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """Registers a callback invoked on order stream updates: cb(event_type, order_data)."""
        self._on_trade_update_callbacks.append(callback)

    def register_status_change_callback(
        self, callback: Callable[[str, str], None]
    ) -> None:
        """Registers a callback for connection status changes: cb(stream_name, status)."""
        self._on_status_change_callbacks.append(callback)

    def _set_status(self, stream_name: str, status: str) -> None:
        if stream_name == "market_data":
            self.market_data_status = status
        elif stream_name == "order_stream":
            self.order_stream_status = status

        for cb in self._on_status_change_callbacks:
            try:
                cb(stream_name, status)
            except Exception as e:
                logger.error(f"Status callback error: {e}")

    # =========================================================================
    # Position & Subscription Management
    # =========================================================================

    def track_position(
        self,
        symbol: str,
        shares: float,
        entry_price: float,
        current_sl: float,
        high_water_mark: Optional[float] = None,
        order_id: Optional[str] = None,
        stop_order_id: Optional[str] = None,
        min_profit_threshold_pct: float = 1.2,
        lock_fraction: float = 0.80,
    ) -> TrackedPosition:
        """
        Adds or updates a position to be actively monitored by the Real-Time Trailing Stop Engine.
        Automatically subscribes the symbol to the market data feed.
        """
        sym = symbol.upper()
        hwm = high_water_mark if high_water_mark is not None else entry_price
        hwm = max(hwm, entry_price)

        pos = TrackedPosition(
            symbol=sym,
            shares=shares,
            entry_price=entry_price,
            current_sl=current_sl,
            high_water_mark=hwm,
            order_id=order_id,
            stop_order_id=stop_order_id,
            min_profit_threshold_pct=min_profit_threshold_pct,
            lock_fraction=lock_fraction,
            last_price=entry_price,
            last_action="TRACKING_INITIALIZED",
        )
        self.tracked_positions[sym] = pos
        logger.info(
            f"🎯 [STREAMER TRACKING] {sym}: {shares} shares @ Entry ${entry_price:.2f} | "
            f"SL ${current_sl:.2f} | HWM ${hwm:.2f}"
        )

        # Ensure market feed is subscribed
        self.subscribe([sym], trades=True, quotes=True)
        return pos

    def untrack_position(self, symbol: str) -> Optional[TrackedPosition]:
        """Removes a symbol from active trailing stop tracking."""
        sym = symbol.upper()
        return self.tracked_positions.pop(sym, None)

    def subscribe(
        self,
        symbols: Union[str, List[str]],
        trades: bool = True,
        quotes: bool = True,
        bars: bool = False,
    ) -> None:
        """Adds symbols to market data subscriptions."""
        if isinstance(symbols, str):
            symbols = [symbols]
        for s in symbols:
            sym = s.upper()
            if trades:
                self.subscribed_trades.add(sym)
            if quotes:
                self.subscribed_quotes.add(sym)
            if bars:
                self.subscribed_bars.add(sym)

        # If connected, immediately send dynamic subscription update
        if self._market_ws and self.market_data_status in [
            "AUTHENTICATED",
            "STREAMING",
        ]:
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._send_market_subscription(
                        trades=list(self.subscribed_trades),
                        quotes=list(self.subscribed_quotes),
                        bars=list(self.subscribed_bars),
                    ),
                    self._loop,
                )

    def unsubscribe(
        self,
        symbols: Union[str, List[str]],
        trades: bool = True,
        quotes: bool = True,
        bars: bool = False,
    ) -> None:
        """Removes symbols from subscriptions."""
        if isinstance(symbols, str):
            symbols = [symbols]
        removed_trades = []
        removed_quotes = []
        removed_bars = []
        for s in symbols:
            sym = s.upper()
            if trades and sym in self.subscribed_trades:
                self.subscribed_trades.remove(sym)
                removed_trades.append(sym)
            if quotes and sym in self.subscribed_quotes:
                self.subscribed_quotes.remove(sym)
                removed_quotes.append(sym)
            if bars and sym in self.subscribed_bars:
                self.subscribed_bars.remove(sym)
                removed_bars.append(sym)

        if self._market_ws and self.market_data_status in [
            "AUTHENTICATED",
            "STREAMING",
        ]:
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._send_market_unsubscription(
                        trades=removed_trades,
                        quotes=removed_quotes,
                        bars=removed_bars,
                    ),
                    self._loop,
                )

    # =========================================================================
    # Real-Time Trailing Stop Engine
    # =========================================================================

    def _process_price_tick(
        self, symbol: str, price: float, tick_type: str, raw_tick: Dict[str, Any]
    ) -> None:
        """
        Core Real-Time Trailing Stop Engine.
        Executes on every incoming tick from market data or REST fallback.
        """
        if price <= 0:
            return

        sym = symbol.upper()
        now_str = datetime.now(timezone.utc).isoformat()

        # Update latest ticks registry
        self.latest_ticks[sym] = {
            "symbol": sym,
            "price": price,
            "type": tick_type,
            "timestamp": now_str,
            "raw": raw_tick,
        }

        # Dispatch tick callback
        for cb in self._on_tick_callbacks:
            try:
                cb(sym, self.latest_ticks[sym])
            except Exception as e:
                logger.error(f"Tick callback error for {sym}: {e}")

        # Check if position is actively tracked
        pos = self.tracked_positions.get(sym)
        if not pos:
            return

        pos.last_price = price
        pos.updated_at = now_str

        # 1. Stop-Loss Breach Check
        if price <= pos.current_sl:
            logger.warning(
                f"🚨 [STOP-LOSS BREACHED] {sym} price (${price:.2f}) <= Stop-Loss (${pos.current_sl:.2f})! "
                f"Entry: ${pos.entry_price:.2f} | HWM: ${pos.high_water_mark:.2f}"
            )
            for cb in self._on_stop_triggered_callbacks:
                try:
                    cb(sym, price, pos.current_sl)
                except Exception as e:
                    logger.error(f"Stop triggered callback error for {sym}: {e}")

            # If broker bridge attached, optionally trigger liquidation
            if self.broker_bridge and hasattr(self.broker_bridge, "close_position"):
                try:
                    logger.info(
                        f"⚡ [AUTO-LIQUIDATE] Calling broker close_position for {sym}"
                    )
                    self.broker_bridge.close_position(sym)
                except Exception as be:
                    logger.error(f"Broker liquidation error for {sym}: {be}")
            return

        # 2. High-Water Mark & Ratchet Check
        if price > pos.high_water_mark:
            old_hwm = pos.high_water_mark
            pos.high_water_mark = price
            logger.debug(
                f"📈 [NEW HIGH-WATER MARK] {sym}: ${old_hwm:.2f} -> ${price:.2f}"
            )

        # Apply structural profit lock logic
        new_sl, new_hwm, action = apply_high_watermark_profit_lock(
            current_price=price,
            entry_price=pos.entry_price,
            highest_price_seen=pos.high_water_mark,
            current_sl=pos.current_sl,
            min_profit_threshold_pct=pos.min_profit_threshold_pct,
            lock_fraction=pos.lock_fraction,
        )

        pos.high_water_mark = max(pos.high_water_mark, new_hwm)

        # Ratchet stop-loss strictly upwards
        if new_sl > pos.current_sl:
            old_sl = pos.current_sl
            pos.current_sl = round(new_sl, 2)
            pos.last_action = action
            logger.info(
                f"🛡️ [STOP-LOSS RATCHETED] {sym}: ${old_sl:.2f} -> ${pos.current_sl:.2f} "
                f"(Action: {action}) | Spot: ${price:.2f} | HWM: ${pos.high_water_mark:.2f}"
            )

            # Fire trailing stop callbacks
            for cb in self._on_trailing_stop_callbacks:
                try:
                    cb(sym, old_sl, pos.current_sl, pos.high_water_mark, action)
                except Exception as e:
                    logger.error(f"Trailing stop callback error for {sym}: {e}")

            # If stop_order_id exists on broker bridge, update/replace order
            if (
                pos.stop_order_id
                and self.broker_bridge
                and hasattr(self.broker_bridge, "replace_order_stop_price")
            ):
                try:
                    self.broker_bridge.replace_order_stop_price(
                        pos.stop_order_id, pos.current_sl
                    )
                except Exception as be:
                    logger.error(f"Failed to update stop order on broker: {be}")

    # =========================================================================
    # WebSocket Protocol Messages
    # =========================================================================

    async def _send_market_subscription(
        self, trades: List[str], quotes: List[str], bars: List[str]
    ) -> None:
        """Sends subscription payload to Alpaca Market Data Stream."""
        if not self._market_ws:
            return
        msg: Dict[str, Any] = {"action": "subscribe"}
        if trades:
            msg["trades"] = list(set(trades))
        if quotes:
            msg["quotes"] = list(set(quotes))
        if bars:
            msg["bars"] = list(set(bars))

        if len(msg) > 1:
            try:
                await self._market_ws.send(json.dumps(msg))
                logger.info(
                    f"📡 [ALPACA MARKET SUB] Subscribed {len(trades)} trades, {len(quotes)} quotes, {len(bars)} bars"
                )
            except Exception as e:
                logger.error(f"Market subscription send error: {e}")

    async def _send_market_unsubscription(
        self, trades: List[str], quotes: List[str], bars: List[str]
    ) -> None:
        """Sends unsubscribe payload to Alpaca Market Data Stream."""
        if not self._market_ws:
            return
        msg: Dict[str, Any] = {"action": "unsubscribe"}
        if trades:
            msg["trades"] = list(set(trades))
        if quotes:
            msg["quotes"] = list(set(quotes))
        if bars:
            msg["bars"] = list(set(bars))

        if len(msg) > 1:
            try:
                await self._market_ws.send(json.dumps(msg))
            except Exception as e:
                logger.error(f"Market unsubscription send error: {e}")

    # =========================================================================
    # Stream 1: Market Data WebSocket Runner
    # =========================================================================

    async def _run_market_data_stream(self) -> None:
        """
        Manages the Market Data WebSocket stream connection with exponential backoff.
        Subscribes to live quotes/trades for active open positions and watchlists.
        """
        backoff = 1.0
        max_backoff = 30.0

        while self._running:
            self._set_status("market_data", "CONNECTING")
            logger.info(
                f"🔌 [ALPACA MARKET DATA] Connecting to {self.market_data_url}..."
            )
            try:
                async with websockets.connect(
                    self.market_data_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._market_ws = ws
                    backoff = 1.0  # Reset backoff upon successful socket connection

                    # 1. Await initial connected greeting
                    greeting = await ws.recv()
                    logger.debug(f"Market WS greeting: {greeting}")

                    # 2. Authenticate
                    auth_msg = {
                        "action": "auth",
                        "key": self.api_key,
                        "secret": self.secret_key,
                    }
                    await ws.send(json.dumps(auth_msg))

                    auth_resp = await ws.recv()
                    logger.debug(f"Market WS auth response: {auth_resp}")
                    auth_data = json.loads(auth_resp)

                    is_auth = False
                    if isinstance(auth_data, list):
                        for item in auth_data:
                            if (
                                item.get("T") == "success"
                                and item.get("msg") == "authenticated"
                            ):
                                is_auth = True
                                break

                    if not is_auth:
                        logger.error(
                            f"❌ [ALPACA MARKET DATA] Authentication failed: {auth_resp}"
                        )
                        self._set_status("market_data", "ERROR")
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, max_backoff)
                        continue

                    self._set_status("market_data", "AUTHENTICATED")
                    logger.info("✅ [ALPACA MARKET DATA] Authenticated successfully.")

                    # 3. Send Subscriptions
                    trades = list(self.subscribed_trades)
                    quotes = list(self.subscribed_quotes)
                    bars = list(self.subscribed_bars)
                    if trades or quotes or bars:
                        await self._send_market_subscription(trades, quotes, bars)

                    self._set_status("market_data", "STREAMING")

                    # 4. Message Loop
                    async for raw_msg in ws:
                        if not self._running:
                            break
                        try:
                            msg_list = json.loads(raw_msg)
                            if not isinstance(msg_list, list):
                                msg_list = [msg_list]

                            for packet in msg_list:
                                msg_type = packet.get("T")
                                symbol = packet.get("S")

                                if msg_type == "t":  # Trade tick
                                    price = float(packet.get("p", 0.0))
                                    self._process_price_tick(
                                        symbol, price, "trade", packet
                                    )
                                elif msg_type == "q":  # Quote tick
                                    # Use mid-price or bid-price for trailing stop evaluation
                                    bp = float(packet.get("bp", 0.0))
                                    ap = float(packet.get("ap", 0.0))
                                    price = (
                                        (bp + ap) / 2.0
                                        if (bp > 0 and ap > 0)
                                        else (bp or ap)
                                    )
                                    self._process_price_tick(
                                        symbol, price, "quote", packet
                                    )
                                elif msg_type == "b":  # Bar tick
                                    c_price = float(packet.get("c", 0.0))
                                    self._process_price_tick(
                                        symbol, c_price, "bar", packet
                                    )
                                elif msg_type == "subscription":
                                    logger.info(
                                        f"📋 [ALPACA MARKET DATA] Subscription confirmed: {packet}"
                                    )
                                elif msg_type == "error":
                                    logger.warning(
                                        f"⚠️ [ALPACA MARKET DATA] Error packet: {packet}"
                                    )
                        except Exception as parse_err:
                            logger.error(
                                f"Market WS message parsing error: {parse_err}"
                            )

            except (ConnectionClosed, WebSocketException) as ws_err:
                logger.warning(f"Market WS disconnected ({ws_err}). Reconnecting...")
            except Exception as e:
                logger.error(f"Market WS unexpected error: {e}")
            finally:
                self._market_ws = None
                self._set_status("market_data", "DISCONNECTED")

            if self._running:
                logger.info(
                    f"Retrying Market WS connection in {backoff:.1f} seconds..."
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    # =========================================================================
    # Stream 2: Order Updates WebSocket Runner (Trading API)
    # =========================================================================

    async def _run_order_stream(self) -> None:
        """
        Manages the Order Updates WebSocket stream connection with exponential backoff.
        Subscribes to live trade_updates for account PA3NFG0UWP3R.
        """
        backoff = 1.0
        max_backoff = 30.0

        while self._running:
            self._set_status("order_stream", "CONNECTING")
            logger.info(
                f"🔌 [ALPACA ORDER STREAM] Connecting to {self.order_stream_url}..."
            )
            try:
                async with websockets.connect(
                    self.order_stream_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._order_ws = ws
                    backoff = 1.0

                    # 1. Authenticate with Trading API protocol
                    auth_msg = {
                        "action": "authenticate",
                        "data": {
                            "key_id": self.api_key,
                            "secret_key": self.secret_key,
                        },
                    }
                    await ws.send(json.dumps(auth_msg))

                    auth_resp = await ws.recv()
                    logger.debug(f"Order WS auth response: {auth_resp}")
                    auth_data = json.loads(auth_resp)

                    stream_status = (
                        auth_data.get("data", {}).get("status")
                        if isinstance(auth_data, dict)
                        else ""
                    )
                    if stream_status != "authorized":
                        logger.error(
                            f"❌ [ALPACA ORDER STREAM] Authentication failed: {auth_resp}"
                        )
                        self._set_status("order_stream", "ERROR")
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, max_backoff)
                        continue

                    self._set_status("order_stream", "AUTHENTICATED")
                    logger.info(
                        f"✅ [ALPACA ORDER STREAM] Authenticated for account {self.account_id}."
                    )

                    # 2. Listen to trade_updates
                    listen_msg = {
                        "action": "listen",
                        "data": {"streams": ["trade_updates"]},
                    }
                    await ws.send(json.dumps(listen_msg))
                    self._set_status("order_stream", "STREAMING")

                    # 3. Message Loop
                    async for raw_msg in ws:
                        if not self._running:
                            break
                        try:
                            msg_dict = json.loads(raw_msg)
                            stream_type = msg_dict.get("stream")
                            if stream_type == "trade_updates":
                                update_data = msg_dict.get("data", {})
                                event = update_data.get("event")
                                order = update_data.get("order", {})
                                symbol = order.get("symbol")

                                logger.info(
                                    f"🔔 [ORDER UPDATE EVENT] {symbol} -> {event.upper()} | "
                                    f"Qty: {update_data.get('qty')} @ Price: ${update_data.get('price', 'N/A')}"
                                )

                                # If position filled or closed, update tracking state
                                if event in ["fill", "partial_fill"] and symbol:
                                    fill_price = float(
                                        update_data.get("price")
                                        or order.get("filled_avg_price")
                                        or 0.0
                                    )
                                    fill_qty = float(
                                        update_data.get("qty")
                                        or order.get("filled_qty")
                                        or 0.0
                                    )
                                    side = order.get("side", "").lower()
                                    if side == "buy" and fill_price > 0:
                                        # Auto-track or update position
                                        if symbol in self.tracked_positions:
                                            self.tracked_positions[symbol].shares = (
                                                fill_qty
                                            )
                                    elif side == "sell":
                                        # Liquidation / stop hit
                                        if symbol in self.tracked_positions:
                                            rem_shares = max(
                                                0.0,
                                                self.tracked_positions[symbol].shares
                                                - fill_qty,
                                            )
                                            if rem_shares <= 0:
                                                self.untrack_position(symbol)

                                # Dispatch trade update callbacks
                                for cb in self._on_trade_update_callbacks:
                                    try:
                                        cb(event, update_data)
                                    except Exception as e:
                                        logger.error(
                                            f"Trade update callback error: {e}"
                                        )

                        except Exception as parse_err:
                            logger.error(f"Order WS message error: {parse_err}")

            except (ConnectionClosed, WebSocketException) as ws_err:
                logger.warning(f"Order WS disconnected ({ws_err}). Reconnecting...")
            except Exception as e:
                logger.error(f"Order WS error: {e}")
            finally:
                self._order_ws = None
                self._set_status("order_stream", "DISCONNECTED")

            if self._running:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    # =========================================================================
    # Resiliency: REST Quote Polling Fallback
    # =========================================================================

    async def _run_rest_fallback_monitor(self) -> None:
        """
        Polls REST quotes if WebSocket is disconnected or stalled, ensuring
        trailing stops are continuously evaluated without blind spots.
        """
        while self._running:
            try:
                # Only activate fallback if enabled and market WS is not streaming
                if (
                    self.enable_rest_fallback
                    and self.market_data_status != "STREAMING"
                    and self.tracked_positions
                ):
                    for sym in list(self.tracked_positions.keys()):
                        try:
                            price = self._poll_rest_quote(sym)
                            if price > 0:
                                self._process_price_tick(
                                    sym, price, "rest_fallback", {"price": price}
                                )
                        except Exception as pe:
                            logger.debug(f"REST fallback poll notice for {sym}: {pe}")

                await asyncio.sleep(self.rest_fallback_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"REST fallback monitor error: {e}")
                await asyncio.sleep(self.rest_fallback_interval)

    def _poll_rest_quote(self, symbol: str) -> float:
        """Fetches latest quote via REST API."""
        # 1. Try Alpaca Data REST endpoint if API keys present
        if self.api_key and self.secret_key:
            try:
                import requests

                headers = {
                    "APCA-API-KEY-ID": self.api_key,
                    "APCA-API-SECRET-KEY": self.secret_key,
                }
                res = requests.get(
                    f"https://data.alpaca.markets/v2/stocks/{symbol.upper()}/quotes/latest",
                    headers=headers,
                    timeout=3,
                )
                if res.status_code == 200:
                    q = res.json().get("quote", {})
                    bp = float(q.get("bp", 0))
                    ap = float(q.get("ap", 0))
                    if bp > 0 and ap > 0:
                        return (bp + ap) / 2.0
                    return bp or ap
            except Exception:
                pass

        # 2. General fallback via realtime tracker
        try:
            from src.realtime_tracker import fetch_live_quote

            q = fetch_live_quote(symbol)
            return float(q.get("price", 0.0))
        except Exception:
            return 0.0

    # =========================================================================
    # Lifecycle Management: Start / Stop
    # =========================================================================

    async def start(self) -> None:
        """Starts the WebSocket streamer asynchronously."""
        if self._running:
            return
        self._running = True
        self._loop = asyncio.get_running_loop()

        logger.info(
            f"🚀 [ALPACA WEBSOCKET STREAMER] Starting Bridge (Paper={self.is_paper})..."
        )
        await asyncio.gather(
            self._run_market_data_stream(),
            self._run_order_stream(),
            self._run_rest_fallback_monitor(),
        )

    async def stop(self) -> None:
        """Stops the WebSocket streamer and closes active connections."""
        self._running = False
        if self._market_ws:
            try:
                await self._market_ws.close()
            except Exception:
                pass
        if self._order_ws:
            try:
                await self._order_ws.close()
            except Exception:
                pass

        self._set_status("market_data", "DISCONNECTED")
        self._set_status("order_stream", "DISCONNECTED")
        logger.info("🛑 [ALPACA WEBSOCKET STREAMER] Stopped.")

    def run_in_background(self) -> threading.Thread:
        """
        Runs the async streaming loop in a dedicated background daemon thread.
        Enables seamless integration with synchronous frameworks and Streamlit.
        """
        if self._thread and self._thread.is_alive():
            return self._thread

        def _target():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            try:
                loop.run_until_complete(self.start())
            except Exception as e:
                logger.error(f"Background streamer loop exited: {e}")
            finally:
                loop.close()

        self._thread = threading.Thread(
            target=_target, name="AlpacaWebSocketStreamerThread", daemon=True
        )
        self._thread.start()
        logger.info("🧵 [ALPACA WEBSOCKET STREAMER] Running in background thread.")
        return self._thread

    def stop_background(self, timeout: float = 3.0) -> None:
        """Stops the background thread and awaits termination."""
        if self._running and self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.stop(), self._loop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            logger.info("🧵 [ALPACA WEBSOCKET STREAMER] Background thread joined.")
