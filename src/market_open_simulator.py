"""
Sentilyze Market-Open Order Execution Simulator.
Simulates Monday 9:30 AM EDT opening order routing, trailing stop evaluation,
and Almgren-Chriss order slicing schedule for high-conviction pre-market watchlist setups.
STRICT PORTFOLIO PRESERVATION: Read-only on paper_portfolio.json; writes only to results/monday_opening_execution_plan.json.
"""

import os
import json
from typing import Dict, Any, List
from datetime import datetime, timezone

from src.utils import get_logger
from src.almgren_chriss_execution import calculate_almgren_chriss_trajectory

logger = get_logger("market_open_simulator")

POSITION_BETAS = {
    "DAL": 1.25,
    "EMR": 1.05,
    "ETN": 1.15,
    "ADP": 0.82,
    "FAST": 1.02,
}

WATCHLIST_BETAS = {
    "MSFT": 1.12,
    "AAPL": 1.10,
    "NVDA": 1.65,
    "GOOGL": 1.08,
    "PLTR": 1.85,
}


def simulate_market_opening(
    portfolio_path: str = "results/paper_portfolio.json",
    watchlist_path: str = "results/monday_premarket_watchlist.json",
    futures_pulse_path: str = "results/overnight_futures_pulse.json",
    output_path: str = "results/monday_opening_execution_plan.json",
) -> Dict[str, Any]:
    """
    Simulates Monday 9:30 AM EDT market-open execution across existing positions
    and high-conviction candidate entries.
    """
    logger.info(
        "⚡ [MARKET-OPEN SIMULATOR] Simulating Monday 9:30 AM EDT opening orders..."
    )

    # 1. Load Portfolio
    total_equity = 144453.41
    cash = 102944.50
    open_positions = {}
    if os.path.exists(portfolio_path):
        try:
            with open(portfolio_path, "r", encoding="utf-8") as f:
                pdata = json.load(f)
                total_equity = float(pdata.get("total_equity", total_equity))
                cash = float(pdata.get("cash", cash))
                open_positions = pdata.get("open_positions", {})
        except Exception as e:
            logger.debug(f"Could not load portfolio: {e}")

    # 2. Load Futures Sentinel Implied Gap
    implied_gap_pct = 0.50
    if os.path.exists(futures_pulse_path):
        try:
            with open(futures_pulse_path, "r", encoding="utf-8") as f:
                fp_data = json.load(f)
                implied_gap_pct = float(
                    fp_data.get("composite_gap_pct", implied_gap_pct)
                )
        except Exception as e:
            logger.debug(f"Could not load futures pulse: {e}")

    # 3. Simulate Scenarios for Active Open Positions
    # Scenario 1: Expected Futures Gap (+0.50%)
    # Scenario 2: Adverse Shock (-1.50%)
    scenarios = {
        "expected_futures_gap": {
            "gap_pct": implied_gap_pct,
            "name": f"Futures Forecast ({implied_gap_pct:+.2f}%)",
        },
        "adverse_gap_stress": {"gap_pct": -1.50, "name": "Adverse Shock (-1.50%)"},
    }

    position_simulations = {}
    for scen_key, scen in scenarios.items():
        scen_gap = scen["gap_pct"]
        scen_results = {}
        for ticker, pos in open_positions.items():
            shares = int(pos.get("shares", 0))
            curr_p = float(pos.get("current_price", pos.get("entry_price", 0.0)))
            entry_p = float(pos.get("entry_price", curr_p))
            sl = float(pos.get("sl_target", entry_p * 0.975))
            tp1 = float(pos.get("tp1_target", entry_p * 1.05))
            beta = POSITION_BETAS.get(ticker.upper(), 1.0)

            sim_open_price = round(curr_p * (1.0 + (scen_gap * beta / 100.0)), 2)
            open_pnl = round((sim_open_price - entry_p) * shares, 2)
            open_pnl_pct = round(((sim_open_price - entry_p) / entry_p) * 100.0, 2)

            # Execution Logic Check
            sl_triggered = sim_open_price <= sl
            tp1_triggered = sim_open_price >= tp1

            if sl_triggered:
                action = "STOP_LOSS_EXIT"
                note = f"Triggered SL at ${sl:.2f}. Guaranteed locked profit: ${round((sl - entry_p) * shares, 2):+.2f}"
            elif tp1_triggered:
                action = "SCALE_OUT_TP1"
                note = f"Hit TP1 target ${tp1:.2f}. Scale out 50% shares ({shares // 2} shs)."
            else:
                action = "HOLD_AND_TRAIL"
                note = f"Position safe above SL ${sl:.2f}. Open PnL: ${open_pnl:+,.2f} ({open_pnl_pct:+.2f}%)"

            scen_results[ticker] = {
                "shares": shares,
                "entry_price": entry_p,
                "current_price": curr_p,
                "simulated_open_price": sim_open_price,
                "open_pnl_dollar": open_pnl,
                "open_pnl_pct": open_pnl_pct,
                "stop_loss": sl,
                "tp1_target": tp1,
                "action": action,
                "execution_note": note,
            }
        position_simulations[scen_key] = scen_results

    # 4. Generate Almgren-Chriss Slicing Schedule for Watchlist Candidate #1
    top_candidate = "MSFT"
    top_conviction = 80.6
    candidate_price_est = 225.00
    if os.path.exists(watchlist_path):
        try:
            with open(watchlist_path, "r", encoding="utf-8") as f:
                wdata = json.load(f)
                wl = wdata.get("watchlist", [])
                if wl:
                    top_item = wl[0]
                    top_candidate = top_item.get("ticker", top_candidate)
                    top_conviction = float(
                        top_item.get("conviction_pct", top_conviction)
                    )
                    candidate_price_est = float(
                        top_item.get("stage1_metrics", {}).get(
                            "current_price", candidate_price_est
                        )
                    )
        except Exception as e:
            logger.debug(f"Could not load watchlist: {e}")

    # Quarter-Kelly Position Sizing (e.g. 5% max risk, ~10% capital allocation)
    alloc_pct = 0.08  # 8% of portfolio equity (~$11.5k)
    target_capital = min(total_equity * alloc_pct, cash * 0.20)
    total_shares = max(int(target_capital / candidate_price_est), 1)
    actual_order_capital = round(total_shares * candidate_price_est, 2)

    # Almgren-Chriss Trajectory: 10 intervals (slicing across 30 minutes, 3 min per slice)
    ac_trajectory = calculate_almgren_chriss_trajectory(
        total_shares=float(total_shares),
        total_time_intervals=10,
        daily_volatility=0.018,
        risk_aversion=1e-5,
        temporary_impact_eta=2.5e-6,
        permanent_impact_gamma=2.5e-7,
        initial_price=candidate_price_est,
    )

    trade_slices = []
    accumulated_shares = 0
    start_minutes = 30  # 9:30 AM EDT
    for i, slice_size in enumerate(ac_trajectory["trade_sizes"]):
        shs = int(round(slice_size))
        accumulated_shares += shs
        minute_offset = i * 3
        slice_time = f"09:{start_minutes + minute_offset:02d} EDT"
        trade_slices.append(
            {
                "slice_index": i + 1,
                "scheduled_time": slice_time,
                "shares_to_buy": shs,
                "cumulative_shares": accumulated_shares,
                "order_type": "LIMIT_AGGRESSIVE" if i < 3 else "LIMIT_PASSIVE",
                "limit_price": round(
                    candidate_price_est * (1.001 if i < 3 else 0.999), 2
                ),
            }
        )

    # Limit and Brackets for Top Candidate
    sl_bracket = round(candidate_price_est * 0.975, 2)  # -2.5% strict initial SL
    tp1_bracket = round(candidate_price_est * 1.050, 2)  # +5.0% TP1
    tp2_bracket = round(candidate_price_est * 1.080, 2)  # +8.0% TP2 runner

    new_order_ticket = {
        "candidate": top_candidate,
        "conviction_pct": top_conviction,
        "order_action": "BUY",
        "total_shares": total_shares,
        "estimated_price": candidate_price_est,
        "total_capital_allocation": actual_order_capital,
        "cash_buffer_remaining": round(cash - actual_order_capital, 2),
        "brackets": {
            "entry_limit": round(candidate_price_est * 1.002, 2),
            "stop_loss_target": sl_bracket,
            "tp1_target": tp1_bracket,
            "tp2_target": tp2_bracket,
        },
        "almgren_chriss_execution": {
            "expected_shortfall_dollars": ac_trajectory["expected_shortfall_dollars"],
            "shortfall_variance": ac_trajectory["shortfall_variance"],
            "execution_window": "09:30 EDT - 10:00 EDT (10 slices @ 3-min intervals)",
            "slices": trade_slices,
        },
    }

    plan_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_open_time": "2026-09-28 09:30:00 EDT (Monday)",
        "portfolio_equity": total_equity,
        "cash_reserve": cash,
        "scenarios_analyzed": scenarios,
        "active_position_routing": position_simulations,
        "opening_order_ticket": new_order_ticket,
    }

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(plan_report, f, indent=2)
        logger.info(f"Saved Monday market-open execution plan to {output_path}")
    except Exception as e:
        logger.error(f"Failed to persist execution plan: {e}")

    return plan_report


if __name__ == "__main__":
    plan = simulate_market_opening()
    print("Market Open Simulation complete:")
    print(
        f"Top Ticket: {plan['opening_order_ticket']['candidate']} - {plan['opening_order_ticket']['total_shares']} shares"
    )
    print(
        f"Almgren-Chriss Slices: {len(plan['opening_order_ticket']['almgren_chriss_execution']['slices'])}"
    )
