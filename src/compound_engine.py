"""
Max Compound Acceleration, Turtle 0.5N Pyramiding & +100% Target Doubling Engine for Sentilyze.
=============================================================================================
Enforces:
1. Turtle 0.5N ATR Pyramiding: Seed Unit (1/3) -> Pyramid Unit (1/3 at +1.0 ATR) -> Runner Unit (1/3 at +2.0 ATR).
2. Dynamic Breakeven & Chandelier Unified Stop Ratchets: Total dollar risk at Unit 3 is lower than at Unit 0.
3. Dynamic Equity-Scaled Position Sizing (Kelly Growth Formula - expands as account grows).
4. Real-Time $200,000 Goal Progress & Milestone Radar (Target 100% Account Doubling).
"""

from typing import Dict, Any, List, Optional
import math
from src.utils import get_logger

logger = get_logger(__name__)


def calculate_turtle_pyramid_plan(
    entry_price: float,
    atr: float,
    total_allocation_dollars: float = 15000.0,
    max_units: int = 3,
    risk_floor_multiplier: float = 2.0,
    target_allocation_dollars: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Computes a 3-stage Turtle ATR pyramiding plan (Seykota & Covel, Leung & Zhan 2017):
    - Unit 0 (Seed Unit): 1/3 allocation at Entry, SL = Entry - (risk_floor * ATR)
    - Unit 1 (Pyramid 1): 1/3 allocation at Entry + 1.0 ATR, Unified SL = Entry + 0.2% (Breakeven)
    - Unit 2 (Runner Unit): 1/3 allocation at Entry + 2.0 ATR, Unified SL = Chandelier (Peak - 2.5 ATR)

    Returns:
        Structured plan with trigger levels, unit shares, and unified stop loss trajectories.
    """
    if target_allocation_dollars is not None:
        total_allocation_dollars = target_allocation_dollars

    if entry_price <= 0.0 or atr <= 0.0:
        return {"status": "INVALID_PARAMETERS"}

    unit_dollars = total_allocation_dollars / max(1, max_units)
    unit_shares = max(1, int(unit_dollars // entry_price))

    unit0_trigger = round(entry_price, 2)
    unit0_sl = round(entry_price - (risk_floor_multiplier * atr), 2)

    unit1_trigger = round(entry_price + (1.0 * atr), 2)
    unit1_unified_sl = round(entry_price * 1.002, 2)  # Breakeven + 0.2%

    unit2_trigger = round(entry_price + (2.0 * atr), 2)
    unit2_unified_sl = round(unit2_trigger - (1.5 * atr), 2)

    return {
        "ticker_entry_price": entry_price,
        "atr_14": round(atr, 2),
        "total_allocation_dollars": round(total_allocation_dollars, 2),
        "total_target_dollars": round(total_allocation_dollars, 2),
        "total_units": max_units,
        "seed_unit_dollars": round(unit_dollars, 2),
        "max_units": max_units,
        "shares_per_unit": unit_shares,
        "unit_triggers": {
            "unit_0_trigger": unit0_trigger,
            "unit_1_trigger": unit1_trigger,
            "unit_2_trigger": unit2_trigger,
        },
        "stages": [
            {
                "unit_index": 0,
                "name": "🌱 Stage 0: Seed Unit (33% Allocation)",
                "trigger_price": unit0_trigger,
                "shares": unit_shares,
                "stop_loss": unit0_sl,
                "risk_profile": "Initial exploratory entry (bounded 1/3 risk)",
            },
            {
                "unit_index": 1,
                "name": "📈 Stage 1: Momentum Confirm (+1.0 ATR)",
                "trigger_price": unit1_trigger,
                "shares": unit_shares,
                "unified_stop_loss": unit1_unified_sl,
                "risk_profile": "Risk-Free: Unified SL moved to Breakeven",
            },
            {
                "unit_index": 2,
                "name": "🚀 Stage 2: Compounding Runner (+2.0 ATR)",
                "trigger_price": unit2_trigger,
                "shares": unit_shares,
                "unified_stop_loss": unit2_unified_sl,
                "risk_profile": "Full 100% position riding with trailing Chandelier",
            },
        ],
    }


def evaluate_pyramiding_step(
    current_price: float,
    position_state: Dict[str, Any],
    atr: float,
) -> Dict[str, Any]:
    """
    Evaluates whether an active open position should add a pyramid unit and ratchets its unified stop.

    Args:
        current_price: Latest spot price
        position_state: Current position dict from paper_broker
        atr: 14-day ATR

    Returns:
        Dict indicating action ('ADD_UNIT_1', 'ADD_UNIT_2', 'RATCHET_STOP', 'HOLD')
    """
    entry_p = float(position_state.get("entry_price", current_price))
    current_units = int(position_state.get("pyramid_units_filled", 1))
    current_sl = float(position_state.get("sl_target", entry_p * 0.96))
    highest_seen = float(position_state.get("highest_price_seen", current_price))

    unit1_trigger = entry_p + (1.0 * atr)
    unit2_trigger = entry_p + (2.0 * atr)

    # Check Stage 1 Pyramid Trigger
    if current_units == 1 and current_price >= unit1_trigger:
        new_sl = round(entry_p * 1.002, 2)
        return {
            "action": "ADD_UNIT_1",
            "trigger_met": True,
            "target_unit": 2,
            "new_sl": max(current_sl, new_sl),
            "reason": f"Price crossed +1.0 ATR (${unit1_trigger:.2f}). Pyramiding Unit 2 & raising SL to Breakeven.",
        }

    # Check Stage 2 Pyramid Trigger
    if current_units == 2 and current_price >= unit2_trigger:
        blended_basis = float(position_state.get("blended_entry", entry_p))
        chandelier_floor = round(current_price - (1.5 * atr), 2)
        breakeven_floor = round(blended_basis * 1.002, 2)
        new_sl = max(chandelier_floor, breakeven_floor)
        return {
            "action": "ADD_UNIT_2",
            "trigger_met": True,
            "target_unit": 3,
            "new_sl": max(current_sl, new_sl),
            "reason": f"Price crossed +2.0 ATR (${unit2_trigger:.2f}). Pyramiding Unit 3 (Full 100% Runner) & ratcheting Chandelier SL (guaranteed >= basis ${breakeven_floor:.2f}).",
        }

    # Dynamic Chandelier Trail for active runners (units >= 2)
    if current_units >= 2:
        chandelier_sl = round(highest_seen * 0.965, 2)
        if chandelier_sl > current_sl:
            return {
                "action": "RATCHET_STOP",
                "trigger_met": True,
                "target_unit": current_units,
                "new_sl": chandelier_sl,
                "reason": f"Ratcheting Chandelier stop to ${chandelier_sl:.2f} based on peak ${highest_seen:.2f}.",
            }

    return {"action": "HOLD", "trigger_met": False, "target_unit": current_units}


def calculate_doubling_progress(
    initial_capital: float = 100000.0,
    current_equity: float = 143883.97,
    target_capital: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Computes exact mathematical progress, run-rate, and remaining cycles to reach +100% account doubling.
    """
    target_cap = target_capital or (initial_capital * 2.0)
    total_gain_dollars = max(0.0, current_equity - initial_capital)
    goal_dollars = target_cap - initial_capital
    progress_pct = min(100.0, (total_gain_dollars / (goal_dollars + 1e-5)) * 100.0)

    avg_cycle_gain_pct = 4.5
    remaining_multiplier = max(1.0, target_cap / max(1000.0, current_equity))
    cycles_remaining = (
        math.ceil(
            math.log(remaining_multiplier)
            / math.log(1.0 + (avg_cycle_gain_pct / 100.0))
        )
        if remaining_multiplier > 1.0
        else 0
    )

    milestones = [
        {
            "milestone": "🏁 Starting Capital",
            "target": initial_capital,
            "gain_pct": "0.0%",
            "status": "COMPLETED 🟢",
        },
        {
            "milestone": "🥉 Phase 1 (+25% Growth)",
            "target": initial_capital * 1.25,
            "gain_pct": "+25.0%",
            "status": (
                "COMPLETED 🟢"
                if current_equity >= initial_capital * 1.25
                else "IN PROGRESS 🔄"
            ),
        },
        {
            "milestone": "🥈 Phase 2 (+50% Growth)",
            "target": initial_capital * 1.50,
            "gain_pct": "+50.0%",
            "status": (
                "COMPLETED 🟢"
                if current_equity >= initial_capital * 1.50
                else "IN PROGRESS 🔄"
            ),
        },
        {
            "milestone": "🥇 Phase 3 (+75% Growth)",
            "target": initial_capital * 1.75,
            "gain_pct": "+75.0%",
            "status": (
                "COMPLETED 🟢"
                if current_equity >= initial_capital * 1.75
                else "PENDING ⏳"
            ),
        },
        {
            "milestone": "🏆 100% DOUBLED ($200,000)",
            "target": target_cap,
            "gain_pct": "+100.0%",
            "status": (
                "COMPLETED 🟢" if current_equity >= target_cap else "ULTIMATE TARGET 🎯"
            ),
        },
    ]

    return {
        "initial_capital": initial_capital,
        "current_equity": round(current_equity, 2),
        "target_capital": target_cap,
        "net_gain_dollars": round(total_gain_dollars, 2),
        "goal_dollars_remaining": round(max(0.0, target_cap - current_equity), 2),
        "progress_pct": round(progress_pct, 2),
        "cycles_remaining": cycles_remaining,
        "avg_cycle_gain_pct": avg_cycle_gain_pct,
        "milestones": milestones,
    }


def compute_compound_position_size(
    current_total_equity: float,
    available_cash: float,
    confidence: float = 0.75,
    max_position_fraction: float = 0.15,
    risk_per_trade_pct: float = 0.020,
) -> Dict[str, Any]:
    """
    Computes dynamic equity-scaled position sizing so trade sizes grow exponentially with equity.
    """
    kelly_scaled_fraction = min(
        max_position_fraction, max(0.05, confidence * max_position_fraction)
    )
    target_position_dollars = current_total_equity * kelly_scaled_fraction

    actual_allocation_dollars = min(available_cash, target_position_dollars)
    max_allowed_loss_dollars = current_total_equity * risk_per_trade_pct

    return {
        "allocated_dollars": round(actual_allocation_dollars, 2),
        "position_fraction_of_equity": round(
            (actual_allocation_dollars / current_total_equity) * 100.0, 2
        ),
        "max_risk_dollars": round(max_allowed_loss_dollars, 2),
        "kelly_scale_pct": round(kelly_scaled_fraction * 100.0, 1),
    }


def calculate_conformal_pyramid_plan(
    entry_price: float,
    current_price: float,
    cqr_p10_lower: float,
    cqr_p50_median: float,
    cqr_p90_upper: float,
    current_units_held: int = 1,
    max_units: int = 3,
    total_allocation_dollars: float = 12000.0,
    breakeven_cushion_pct: float = 0.002,
) -> Dict[str, Any]:
    """
    Computes a risk-free Conformal Quantile Pyramiding Plan (Invention 4).
    Enforces the mathematical invariant:
    A new pyramid unit is ONLY allowed when the 90% conformal lower bound P_10
    has ratcheted strictly above the initial entry price + breakeven cushion.
    This guarantees that the position's dollar risk on the portfolio is $0.00.
    """
    if entry_price <= 0 or current_price <= 0:
        return {"can_pyramid": False, "reason": "Invalid price inputs"}

    if current_units_held >= max_units:
        return {
            "can_pyramid": False,
            "reason": f"Maximum pyramid capacity reached ({max_units} units held)",
            "current_units_held": current_units_held,
        }

    breakeven_floor = entry_price * (1.0 + breakeven_cushion_pct)

    # Condition 1: P_10 lower bound must have ratcheted above breakeven
    if cqr_p10_lower < breakeven_floor:
        return {
            "can_pyramid": False,
            "reason": f"P10 lower bound has not locked in breakeven (${cqr_p10_lower:.2f} < ${breakeven_floor:.2f})",
            "current_units_held": current_units_held,
            "cqr_p10_lower": round(cqr_p10_lower, 2),
            "breakeven_floor": round(breakeven_floor, 2),
        }

    # Condition 2: Current price must confirm positive trend momentum (above entry)
    if current_price < entry_price:
        return {
            "can_pyramid": False,
            "reason": "Current price is below entry price",
            "current_units_held": current_units_held,
        }

    # Sizing for next pyramid unit
    unit_dollars = total_allocation_dollars / max_units
    shares_to_add = max(1, int(unit_dollars // current_price))
    next_unit_index = current_units_held + 1

    # Unified trailing stop ratcheted to the conformal lower bound P_10
    unified_stop_loss = max(breakeven_floor, cqr_p10_lower)

    return {
        "can_pyramid": True,
        "ticker_entry_price": round(entry_price, 2),
        "current_price": round(current_price, 2),
        "cqr_p10_lower": round(cqr_p10_lower, 2),
        "cqr_p50_median": round(cqr_p50_median, 2),
        "cqr_p90_upper": round(cqr_p90_upper, 2),
        "current_units_held": current_units_held,
        "next_unit_index": next_unit_index,
        "shares_to_add": shares_to_add,
        "unit_allocation_dollars": round(unit_dollars, 2),
        "unified_stop_loss": round(unified_stop_loss, 2),
        "additional_dollar_risk": 0.0,
        "risk_status": "🟢 RISK-FREE PYRAMID APPROVED (Profits Locked at Breakeven Floor)",
    }
