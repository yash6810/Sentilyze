"""
FinMem Working Memory Tape: Sliding Micro-Tape for Real-Time Order Flow & Tick Dynamics.
========================================================================================
Institutional Microstructure Working Memory Layer:
1. Thread-safe, low-latency, sliding 50-tick de-que holding intraday micro-events.
2. In-flight computation of micro-price velocity, acceleration, volume delta flow toxicity,
   and bid-ask spread expansion hazard.
3. Provides immediate, sub-millisecond context to the 5-Agent Council during live deliberation.
4. Operates strictly in RAM (< 5 MB footprint) with optional lightweight snapshot persistence.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Deque
from collections import deque
import threading
import time
import numpy as np
from datetime import datetime, timezone
from src.utils import get_logger

logger = get_logger("working_memory_tape")


@dataclass
class MicroTick:
    timestamp: float
    iso_time: str
    price: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread_bps: float = 0.0
    tick_direction: int = 0  # +1 buy (uptick), -1 sell (downtick), 0 flat
    volume_delta: float = 0.0  # signed volume


class WorkingMemoryTape:
    """
    Sliding 50-tick working memory tape per ticker.
    Thread-safe, sub-millisecond query speed, zero external dependencies.
    """

    def __init__(self, maxlen: int = 50):
        self.maxlen = maxlen
        self._tapes: Dict[str, Deque[MicroTick]] = {}
        self._lock = threading.Lock()

    def record_tick(
        self,
        ticker: str,
        price: float,
        volume: float = 1.0,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> MicroTick:
        """
        Records a single price/orderbook tick into the sliding working memory tape.
        """
        now_ts = timestamp or time.time()
        iso_str = datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat()
        price = float(price)
        volume = max(float(volume), 1e-6)

        # Estimate spread in basis points
        spread_bps = 0.0
        if bid is not None and ask is not None and ask > bid and price > 0:
            spread_bps = round(((ask - bid) / price) * 10000.0, 2)

        with self._lock:
            if ticker not in self._tapes:
                self._tapes[ticker] = deque(maxlen=self.maxlen)

            tape = self._tapes[ticker]
            prev_price = tape[-1].price if len(tape) > 0 else price

            if price > prev_price:
                direction = 1
            elif price < prev_price:
                direction = -1
            else:
                direction = 0

            volume_delta = float(direction * volume)

            tick = MicroTick(
                timestamp=now_ts,
                iso_time=iso_str,
                price=price,
                volume=volume,
                bid=bid,
                ask=ask,
                spread_bps=spread_bps,
                tick_direction=direction,
                volume_delta=volume_delta,
            )
            tape.append(tick)
            return tick

    def get_tape_length(self, ticker: str) -> int:
        with self._lock:
            if ticker in self._tapes:
                return len(self._tapes[ticker])
            return 0

    def get_microstructure_state(self, ticker: str) -> Dict[str, Any]:
        """
        Computes real-time micro-metrics over the sliding tape.
        """
        with self._lock:
            tape = list(self._tapes.get(ticker, []))

        n = len(tape)
        if n < 3:
            return {
                "ticker": ticker,
                "tape_length": n,
                "status": "INSUFFICIENT_TICKS",
                "velocity_10": 0.0,
                "velocity_50": 0.0,
                "acceleration": 0.0,
                "flow_toxicity_score": 0.0,
                "spread_expansion_hazard": False,
                "volatility_burst_flag": False,
                "tape_pressure": "EQUILIBRIUM",
                "median_spread_bps": 0.0,
                "current_spread_bps": 0.0,
            }

        prices = np.array([t.price for t in tape], dtype=np.float64)
        volumes = np.array([t.volume for t in tape], dtype=np.float64)
        vol_deltas = np.array([t.volume_delta for t in tape], dtype=np.float64)
        spreads = np.array(
            [t.spread_bps for t in tape if t.spread_bps > 0], dtype=np.float64
        )

        # 1. Micro-velocity
        curr_price = prices[-1]
        v10_idx = max(0, n - 10)
        v10 = float((curr_price - prices[v10_idx]) / max(prices[v10_idx], 1e-6))
        v50 = float((curr_price - prices[0]) / max(prices[0], 1e-6))

        # 2. Acceleration (d^2 P / dt^2)
        if n >= 6:
            mid_idx = n // 2
            v_early = (prices[mid_idx] - prices[0]) / max(prices[0], 1e-6)
            v_late = (curr_price - prices[mid_idx]) / max(prices[mid_idx], 1e-6)
            accel = float(v_late - v_early)
        else:
            accel = 0.0

        # 3. Flow Toxicity / Order Flow Imbalance (Cumulative Volume Delta ratio)
        tot_vol = np.sum(volumes)
        net_delta = np.sum(vol_deltas)
        flow_toxicity = float(np.clip(net_delta / max(tot_vol, 1e-6), -1.0, 1.0))

        # 4. Spread expansion check
        current_spread = tape[-1].spread_bps
        median_spread = float(np.median(spreads)) if len(spreads) > 0 else 0.0
        spread_hazard = False
        if (
            current_spread > 0
            and median_spread > 0
            and current_spread >= (2.0 * median_spread)
            and current_spread > 10.0
        ):
            spread_hazard = True

        # 5. Volatility burst
        returns = np.diff(prices) / np.maximum(prices[:-1], 1e-6)
        if len(returns) >= 10:
            recent_vol = np.std(returns[-5:]) if len(returns) >= 5 else 0.0
            baseline_vol = np.std(returns) if len(returns) > 0 else 1e-6
            vol_burst = bool(
                recent_vol > 2.5 * max(baseline_vol, 1e-6) and recent_vol > 0.002
            )
        else:
            vol_burst = False

        # 6. Pressure categorization
        if spread_hazard:
            pressure = "SPREAD_EXPANSION_DANGER"
        elif flow_toxicity > 0.35 and v10 > 0.001:
            pressure = "BULLISH_ABSORPTION"
        elif flow_toxicity < -0.35 and v10 < -0.001:
            pressure = "BEARISH_DUMPING"
        elif vol_burst:
            pressure = "VOLATILITY_EXPANSION"
        else:
            pressure = "EQUILIBRIUM"

        return {
            "ticker": ticker,
            "tape_length": n,
            "status": "ACTIVE",
            "velocity_10": round(v10, 5),
            "velocity_50": round(v50, 5),
            "acceleration": round(accel, 5),
            "flow_toxicity_score": round(flow_toxicity, 3),
            "spread_expansion_hazard": spread_hazard,
            "volatility_burst_flag": vol_burst,
            "tape_pressure": pressure,
            "median_spread_bps": round(median_spread, 2),
            "current_spread_bps": round(current_spread, 2),
        }

    def format_deliberation_context(self, ticker: str) -> str:
        """
        Formats a clean, human-readable summary string of working memory for Council deliberators.
        """
        state = self.get_microstructure_state(ticker)
        if state["status"] == "INSUFFICIENT_TICKS":
            return f"[FinMem Tape: {state['tape_length']} ticks (Buffering)]"

        return (
            f"[FinMem Micro-Tape ({state['tape_length']} ticks): Pressure={state['tape_pressure']} | "
            f"V10={state['velocity_10']:+.2%} | Toxicity={state['flow_toxicity_score']:+.2f} | "
            f"Spread={state['current_spread_bps']:.1f}bps (Hazard={state['spread_expansion_hazard']}) | "
            f"VolBurst={state['volatility_burst_flag']}]"
        )

    def clear(self, ticker: Optional[str] = None):
        """Clears memory tape for a single ticker or all tickers."""
        with self._lock:
            if ticker:
                if ticker in self._tapes:
                    self._tapes[ticker].clear()
            else:
                self._tapes.clear()


# Global Singleton Instance
_GLOBAL_TAPE: Optional[WorkingMemoryTape] = None
_TAPE_LOCK = threading.Lock()


def get_working_memory_tape() -> WorkingMemoryTape:
    """Returns the process-wide singleton WorkingMemoryTape instance."""
    global _GLOBAL_TAPE
    with _TAPE_LOCK:
        if _GLOBAL_TAPE is None:
            _GLOBAL_TAPE = WorkingMemoryTape(maxlen=50)
        return _GLOBAL_TAPE
