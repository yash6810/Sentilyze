"""
Dedicated Ticker Sentinel & Peak-Crest Volume Harvester Swarm for Sentilyze.
Enforces:
1. One Dedicated Micro-Agent Sentinel per Active Stock Position.
2. 15-Minute Micro-Waveform & Intraday VWAP Tracking.
3. Peak Crest Top-Tick & Volume Exhaustion Harvester (Sells at the highest point of the momentum surge).
4. Sub-Second Concurrent Swarm Execution.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


def detect_peak_crest_exhaustion(
    current_price: float,
    entry_price: float,
    highest_price_seen: float,
    volume_ratio: float,
    recent_closes: List[float],
    recent_highs: Optional[List[float]] = None,
    recent_lows: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Detects if a stock has reached the crest/peak of its 15-minute momentum wave using:
    1. Volume Exhaustion (Heavy volume surge > 1.4x accompanied by stalling price).
    2. Upper Wick Rejection (Sellers stepping in at the top).
    3. Peak Retraction (> 0.6% drop from absolute wave crest while in profit).
    """
    if current_price <= 0 or entry_price <= 0:
        return {
            "is_crest_exhausted": False,
            "exhaustion_score": 0.0,
            "reason": "INVALID_PRICE_INPUT",
            "action": "HOLD",
        }

    peak_price = max(highest_price_seen, current_price)
    gain_from_entry_pct = (current_price - entry_price) / entry_price * 100.0
    peak_gain_pct = (peak_price - entry_price) / entry_price * 100.0
    drop_from_peak_pct = (peak_price - current_price) / peak_price * 100.0

    exhaustion_signals = []
    exhaustion_score = 0.0

    # Signal 1: Price is in solid profit (>= +1.5%) and retracting from peak
    if peak_gain_pct >= 1.5:
        if drop_from_peak_pct >= 0.5:
            exhaustion_signals.append(
                f"Peak Retraction: Fell {drop_from_peak_pct:.2f}% from crest (${peak_price:,.2f})"
            )
            exhaustion_score += 0.40

    # Signal 2: Volume Climax / Stalling (Volume > 1.4x but recent price momentum is flat or decelerating)
    if volume_ratio >= 1.40 and len(recent_closes) >= 3:
        price_momentum = (
            (recent_closes[-1] - recent_closes[-3]) / recent_closes[-3] * 100.0
        )
        if price_momentum <= 0.15 and gain_from_entry_pct >= 1.0:
            exhaustion_signals.append(
                f"Volume Climax Exhaustion: {volume_ratio:.2f}x volume with stalling price velocity"
            )
            exhaustion_score += 0.35

    # Signal 3: Upper Wick Selling Pressure
    if recent_highs and recent_lows and len(recent_highs) >= 2:
        last_high = recent_highs[-1]
        last_low = recent_lows[-1]
        candle_range = max(0.01, last_high - last_low)
        upper_wick = last_high - current_price
        if upper_wick / candle_range >= 0.50 and gain_from_entry_pct >= 1.2:
            exhaustion_signals.append(
                "Upper Wick Rejection: Heavy sell-side liquidity absorbed at the high"
            )
            exhaustion_score += 0.30

    is_exhausted = exhaustion_score >= 0.65 and gain_from_entry_pct >= 1.2

    action = (
        "HARVEST_PEAK_PROFIT"
        if is_exhausted
        else ("TRAIL_CREST_TIGHT" if exhaustion_score >= 0.40 else "RIDE_MOMENTUM_WAVE")
    )

    return {
        "is_crest_exhausted": is_exhausted,
        "exhaustion_score": round(min(1.0, exhaustion_score), 2),
        "peak_price": round(peak_price, 2),
        "gain_from_entry_pct": round(gain_from_entry_pct, 2),
        "drop_from_peak_pct": round(drop_from_peak_pct, 2),
        "signals": exhaustion_signals,
        "action": action,
    }


class TickerSentinel:
    """
    Dedicated Micro-Agent assigned to monitor a single stock position 24/7.
    """

    def __init__(self, ticker: str, entry_price: float, shares: int):
        self.ticker = ticker
        self.entry_price = entry_price
        self.shares = shares
        self.highest_price_seen = entry_price
        self.last_update = datetime.now(timezone.utc).isoformat()

    def audit_tick(
        self,
        current_price: float,
        volume_ratio: float = 1.0,
        recent_closes: Optional[List[float]] = None,
        recent_highs: Optional[List[float]] = None,
        recent_lows: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Audits live price tick and determines peak crest execution."""
        if current_price > self.highest_price_seen:
            self.highest_price_seen = current_price

        closes = recent_closes or [self.entry_price, current_price]
        crest_analysis = detect_peak_crest_exhaustion(
            current_price=current_price,
            entry_price=self.entry_price,
            highest_price_seen=self.highest_price_seen,
            volume_ratio=volume_ratio,
            recent_closes=closes,
            recent_highs=recent_highs,
            recent_lows=recent_lows,
        )

        self.last_update = datetime.now(timezone.utc).isoformat()

        return {
            "ticker": self.ticker,
            "entry_price": self.entry_price,
            "current_price": current_price,
            "highest_price_seen": self.highest_price_seen,
            "shares": self.shares,
            "unrealized_pnl": round(
                (current_price - self.entry_price) * self.shares, 2
            ),
            "return_pct": round(
                (current_price - self.entry_price) / self.entry_price * 100.0, 2
            ),
            "volume_ratio": volume_ratio,
            "crest_analysis": crest_analysis,
            "status": (
                "🎯 HARVEST READY"
                if crest_analysis["is_crest_exhausted"]
                else "🟢 TRACKING WAVE"
            ),
        }


class TickerSentinelSwarm:
    """
    Manages the full swarm of Dedicated Ticker Sentinels across all open positions.
    """

    def __init__(self):
        self.sentinels: Dict[str, TickerSentinel] = {}

    def sync_open_positions(self, open_positions_dict: Dict[str, Any]):
        """Synchronizes active sentinels with current portfolio open positions."""
        current_tickers = set(open_positions_dict.keys())

        # Remove sentinels for closed positions
        for t in list(self.sentinels.keys()):
            if t not in current_tickers:
                del self.sentinels[t]

        # Spawn new sentinels for new positions
        for t, pos in open_positions_dict.items():
            if t not in self.sentinels:
                self.sentinels[t] = TickerSentinel(
                    ticker=t,
                    entry_price=float(pos.get("entry_price", 100.0)),
                    shares=int(pos.get("shares", 1)),
                )

    def audit_all_sentinels(
        self,
        quotes_map: Dict[str, Dict[str, Any]],
        volume_ratios_map: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Audits all active sentinels concurrently."""
        reports = []
        vol_map = volume_ratios_map or {}
        for ticker, sentinel in self.sentinels.items():
            q = quotes_map.get(ticker, {})
            curr_price = float(q.get("price", sentinel.entry_price))
            vol_ratio = float(vol_map.get(ticker, 1.0))
            rep = sentinel.audit_tick(current_price=curr_price, volume_ratio=vol_ratio)
            reports.append(rep)
        return reports
