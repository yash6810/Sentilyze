"""
Apple Watch & Wear OS Glance Complications API for Sentilyze.
Pillar 7 Mobile & Omnichannel Module:
- Delivers ultra-low-bandwidth, sub-millisecond JSON payloads optimized for smartwatch complications.
- Formats circular, modular, and rectangular glance widgets with live portfolio P&L and top AI signals.
"""

from typing import Any, Dict
from src.utils import get_logger

logger = get_logger(__name__)


def generate_smartwatch_glance_payload(
    total_equity: float = 104250.0,
    daily_pnl_pct: float = 2.45,
    top_active_ticker: str = "NVDA",
    top_active_pnl_pct: float = 4.80,
    top_signal_ticker: str = "AMD",
    top_signal_confidence: float = 0.78,
) -> Dict[str, Any]:
    """
    Generates structured complication JSON for Apple Watch (watchOS) and Wear OS.
    """
    pnl_sign = "+" if daily_pnl_pct >= 0 else ""

    from datetime import datetime, timezone

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "complications": {
            "circular_gauge": {
                "label": "P&L",
                "value_text": f"{pnl_sign}{daily_pnl_pct:.1f}%",
                "fill_ratio": min(1.0, max(0.0, (daily_pnl_pct + 5.0) / 10.0)),
                "tint_color": "#00D4AA" if daily_pnl_pct >= 0 else "#EF4444",
            },
            "modular_large": {
                "header": "SENTILYZE PORTFOLIO",
                "equity_text": f"${total_equity:,.0f}",
                "body_line1": f"Active: {top_active_ticker} ({pnl_sign}{top_active_pnl_pct:.1f}%)",
                "body_line2": f"Top AI: BUY {top_signal_ticker} ({top_signal_confidence*100:.0f}%)",
                "status_indicator": "ACTIVE_PULSE_GREEN",
            },
            "corner_small": {
                "text": f"{top_active_ticker}",
                "subtext": f"{pnl_sign}{top_active_pnl_pct:.1f}%",
            },
        },
        "haptic_alert_triggers": {
            "tp1_reached": False,
            "stop_loss_hit": False,
            "new_high_conviction_signal": top_signal_confidence >= 0.75,
        },
    }
    return payload
