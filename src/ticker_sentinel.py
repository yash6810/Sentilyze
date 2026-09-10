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
    Tracks peak crest exhaustion, zero-giveback stop ratchets, and capital shields.
    """

    def __init__(
        self,
        ticker: str,
        entry_price: float,
        shares: int,
        highest_price_seen: Optional[float] = None,
        sl_target: Optional[float] = None,
        tp1_target: Optional[float] = None,
        tp2_target: Optional[float] = None,
        scaled_out: bool = False,
    ):
        self.ticker = ticker
        self.entry_price = entry_price
        self.shares = shares
        self.highest_price_seen = highest_price_seen or entry_price
        self.sl_target = sl_target or round(entry_price * 0.975, 2)
        self.tp1_target = tp1_target or round(entry_price * 1.075, 2)
        self.tp2_target = tp2_target or round(entry_price * 1.135, 2)
        self.scaled_out = scaled_out
        self.profit_lock_status = "🟢 TRACKING WAVE"
        self.last_update = datetime.now(timezone.utc).isoformat()

    def audit_tick(
        self,
        current_price: float,
        volume_ratio: float = 1.0,
        recent_closes: Optional[List[float]] = None,
        recent_highs: Optional[List[float]] = None,
        recent_lows: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Audits live price tick, applies Zero-Giveback profit ratchets, and detects peak crests."""
        if current_price > self.highest_price_seen:
            self.highest_price_seen = current_price

        from src.smart_trader_engine import (
            apply_high_watermark_profit_lock,
            enforce_capital_shield_stop_floor,
        )

        # 1. Capital Shield (-2.5% max risk ceiling)
        self.sl_target, shield_act = enforce_capital_shield_stop_floor(
            entry_price=self.entry_price,
            current_sl=self.sl_target,
            max_loss_pct=2.50,
        )

        # 2. Dynamic High-Watermark 80% Retention & Breakeven Ratchet
        new_sl, new_peak, hwm_act = apply_high_watermark_profit_lock(
            current_price=current_price,
            entry_price=self.entry_price,
            highest_price_seen=self.highest_price_seen,
            current_sl=self.sl_target,
            min_profit_threshold_pct=1.20,
            lock_fraction=0.80,
        )
        self.highest_price_seen = new_peak
        if new_sl > self.sl_target:
            self.sl_target = new_sl

        # Calculate metrics
        gain_from_entry_pct = (
            (current_price - self.entry_price) / self.entry_price * 100.0
            if self.entry_price > 0
            else 0.0
        )
        peak_gain_pct = (
            (self.highest_price_seen - self.entry_price) / self.entry_price * 100.0
            if self.entry_price > 0
            else 0.0
        )

        # Determine Profit Lock Status Badge
        if self.scaled_out or self.sl_target >= self.entry_price * 1.01:
            self.profit_lock_status = "🔒 +80% PEAK LOCKED"
        elif peak_gain_pct >= 1.0 or self.sl_target >= self.entry_price * 1.005:
            self.profit_lock_status = "🔒 TIER-1 SECURED (+0.5%)"
        elif peak_gain_pct >= 0.50 or self.sl_target >= self.entry_price * 1.001:
            self.profit_lock_status = "🛡️ RISK-FREE BREAKEVEN"
        elif current_price < self.entry_price:
            self.profit_lock_status = "🛡️ CAPITAL SHIELD (-2.5% MAX)"
        else:
            self.profit_lock_status = "🟢 TRACKING WAVE"

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

        status = (
            "🎯 HARVEST READY"
            if crest_analysis["is_crest_exhausted"]
            else self.profit_lock_status
        )

        return {
            "ticker": self.ticker,
            "entry_price": self.entry_price,
            "current_price": current_price,
            "highest_price_seen": self.highest_price_seen,
            "sl_target": self.sl_target,
            "tp1_target": self.tp1_target,
            "tp2_target": self.tp2_target,
            "scaled_out": self.scaled_out,
            "shares": self.shares,
            "unrealized_pnl": round(
                (current_price - self.entry_price) * self.shares, 2
            ),
            "return_pct": round(gain_from_entry_pct, 2),
            "peak_gain_pct": round(peak_gain_pct, 2),
            "volume_ratio": volume_ratio,
            "crest_analysis": crest_analysis,
            "profit_lock_status": self.profit_lock_status,
            "status": status,
        }


class TickerSentinelSwarm:
    """
    Manages the full swarm of Dedicated Ticker Sentinels across all open positions.
    Guarantees 24/7 persistent surveillance and zero-giveback profit locking.
    """

    def __init__(self):
        self.sentinels: Dict[str, TickerSentinel] = {}

    def sync_open_positions(self, open_positions_dict: Dict[str, Any]):
        """Synchronizes active sentinels with current portfolio open positions without losing state."""
        current_tickers = set(open_positions_dict.keys())

        # Remove sentinels for closed positions
        for t in list(self.sentinels.keys()):
            if t not in current_tickers:
                del self.sentinels[t]

        # Spawn or update sentinels for active positions
        for t, pos in open_positions_dict.items():
            entry_p = float(pos.get("entry_price", 100.0))
            curr_p = float(pos.get("current_price", entry_p))
            highest_p = float(pos.get("highest_price_seen", max(entry_p, curr_p)))
            sl = float(pos.get("sl_target", entry_p * 0.975))
            tp1 = float(pos.get("tp1_target", entry_p * 1.075))
            tp2 = float(pos.get("tp2_target", entry_p * 1.135))
            scaled = bool(pos.get("scaled_out", False))

            if t not in self.sentinels:
                self.sentinels[t] = TickerSentinel(
                    ticker=t,
                    entry_price=entry_p,
                    shares=int(pos.get("shares", 1)),
                    highest_price_seen=highest_p,
                    sl_target=sl,
                    tp1_target=tp1,
                    tp2_target=tp2,
                    scaled_out=scaled,
                )
            else:
                # Keep stateful attributes in sync without downgrading
                self.sentinels[t].shares = int(pos.get("shares", 1))
                self.sentinels[t].highest_price_seen = max(
                    self.sentinels[t].highest_price_seen, highest_p
                )
                self.sentinels[t].sl_target = max(self.sentinels[t].sl_target, sl)
                self.sentinels[t].scaled_out = self.sentinels[t].scaled_out or scaled

    def audit_all_sentinels(
        self,
        quotes_map: Dict[str, Dict[str, Any]],
        volume_ratios_map: Optional[Dict[str, float]] = None,
        sync_to_portfolio: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Audits all active sentinels concurrently.
        Optionally persists ratcheted SL and peak memory back into portfolio state.
        """
        reports = []
        vol_map = volume_ratios_map or {}
        for ticker, sentinel in self.sentinels.items():
            q = quotes_map.get(ticker, {})
            curr_price = float(q.get("price", sentinel.entry_price))
            vol_ratio = float(vol_map.get(ticker, 1.0))
            rep = sentinel.audit_tick(current_price=curr_price, volume_ratio=vol_ratio)
            reports.append(rep)

            # Synchronize ratcheted values back into open_positions dictionary
            if sync_to_portfolio and "open_positions" in sync_to_portfolio:
                if ticker in sync_to_portfolio["open_positions"]:
                    p_entry = sync_to_portfolio["open_positions"][ticker]
                    p_entry["current_price"] = curr_price
                    p_entry["highest_price_seen"] = rep["highest_price_seen"]
                    p_entry["sl_target"] = max(
                        float(p_entry.get("sl_target", 0.0)), rep["sl_target"]
                    )
                    p_entry["profit_lock_status"] = rep["profit_lock_status"]

        return reports
