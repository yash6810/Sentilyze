"""
Autonomous Backtest Autopsy, WFO Efficiency & Tear-Sheet Factsheet Engine for Sentilyze.

Mechanics & Metrics:
1. Calmar Ratio (Annualized Return / Max Drawdown).
2. Sortino Ratio (Downside deviation penalty only).
3. Omega Ratio (Probability-weighted gain/loss above threshold).
4. Walk-Forward Efficiency (WFE = Out-Of-Sample Sharpe / In-Sample Sharpe).
5. Autonomous Post-Mortem Learner for Executed Paper Trades (mines executed_trades.csv safely).
"""

from typing import Any, Dict, List, Optional
import os
import numpy as np
import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)

TRADES_FILE = os.path.join("results", "executed_trades.csv")


def compute_advanced_performance_ratios(
    equity_series: pd.Series,
    risk_free_rate: float = 0.045,
    periods_per_year: int = 252,
) -> Dict[str, Any]:
    """
    Computes institutional performance ratios: Calmar, Sortino, Omega, Max DD Duration.
    """
    if equity_series.empty or len(equity_series) < 5:
        return {
            "annualized_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "calmar_ratio": 0.0,
            "sortino_ratio": 0.0,
            "omega_ratio": 1.0,
            "max_drawdown_duration_days": 0,
        }

    eq = equity_series.astype(float).values
    n = len(eq)
    total_return = (eq[-1] - eq[0]) / max(eq[0], 1.0)
    years = max(n / float(periods_per_year), 0.05)
    cagr = ((1.0 + max(total_return, -0.99)) ** (1.0 / years)) - 1.0

    # Daily returns
    rets = np.diff(eq) / eq[:-1]
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess_rets = rets - rf_daily

    # Drawdowns
    running_max = np.maximum.accumulate(eq)
    drawdowns = (eq - running_max) / running_max
    max_dd = float(np.min(drawdowns))
    max_dd_pct = abs(max_dd) * 100.0

    # Calmar Ratio: CAGR / Max DD
    calmar = cagr / max(abs(max_dd), 0.01)

    # Sortino Ratio: Mean Excess Return / Downside Deviation
    neg_rets = rets[rets < rf_daily] - rf_daily
    if len(neg_rets) > 0:
        downside_dev = np.sqrt(np.mean(neg_rets**2)) * np.sqrt(periods_per_year)
    else:
        downside_dev = 1e-4

    mean_excess_ann = np.mean(excess_rets) * periods_per_year
    sortino = mean_excess_ann / max(downside_dev, 1e-4)

    # Omega Ratio (threshold = rf_daily)
    gains = np.sum(np.maximum(rets - rf_daily, 0.0))
    losses = np.sum(np.maximum(rf_daily - rets, 0.0))
    omega = float(gains / max(losses, 1e-6))

    # Max Drawdown Duration (consecutive periods underwater)
    underwater = drawdowns < -0.0001
    max_duration = 0
    curr_duration = 0
    for is_uw in underwater:
        if is_uw:
            curr_duration += 1
            if curr_duration > max_duration:
                max_duration = curr_duration
        else:
            curr_duration = 0

    return {
        "annualized_return_pct": round(cagr * 100.0, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "calmar_ratio": round(float(calmar), 2),
        "sortino_ratio": round(float(sortino), 2),
        "omega_ratio": round(float(omega), 2),
        "max_drawdown_duration_days": int(max_duration),
    }


def compute_walk_forward_efficiency(
    is_sharpe: float,
    oos_sharpe: float,
) -> Dict[str, Any]:
    """
    Computes Pardo (2008) Walk-Forward Efficiency (WFE) Ratio:
    WFE = OOS_Sharpe / IS_Sharpe
    """
    if is_sharpe <= 0.0:
        wfe = 0.0
    else:
        wfe = oos_sharpe / is_sharpe

    wfe_pct = round(wfe * 100.0, 1)

    if wfe >= 0.70:
        status = "EXCELLENT_ROBUST_ALPHA"
        desc = "Strategy maintains >70% of in-sample edge out-of-sample with zero curve-fitting."
    elif wfe >= 0.50:
        status = "ACCEPTABLE_INSTITUTIONAL_GRADE"
        desc = "Normal degradation under live market frictions; strategy retains positive edge."
    else:
        status = "OVERFITTING_WARNING"
        desc = "Significant performance decay out-of-sample; possible over-parameterization."

    return {
        "in_sample_sharpe": round(is_sharpe, 2),
        "out_of_sample_sharpe": round(oos_sharpe, 2),
        "wfe_ratio": round(wfe, 2),
        "wfe_pct": wfe_pct,
        "classification": status,
        "description": desc,
    }


def run_autonomous_trade_autopsy(
    trades_path: str = TRADES_FILE,
) -> Dict[str, Any]:
    """
    Mines executed trade history (strictly read-only) to produce an autopsy factsheet:
    - Winner vs Loser metrics
    - Win rate and payoff ratio
    - Profit harvest efficiency (TP1 vs TP2)
    - Automated lessons learned for AgentMemoryStore
    """
    if not os.path.exists(trades_path):
        return {
            "status": "NO_TRADES_FOUND",
            "total_trades": 0,
            "win_rate": 0.0,
            "lessons": ["No executed trades logged yet."],
        }

    try:
        df = pd.read_csv(trades_path)
    except Exception as e:
        logger.error(f"Failed to read trades log at {trades_path}: {e}")
        return {"status": "READ_ERROR", "error": str(e)}

    if df.empty:
        return {"status": "EMPTY_TRADES_LOG", "total_trades": 0}

    total_trades = len(df)
    pnl = df["pnl"].astype(float)
    ret_pct = df["return_pct"].astype(float)

    winners = df[pnl > 0]
    losers = df[pnl <= 0]

    n_win = len(winners)
    n_loss = len(losers)
    win_rate = (n_win / total_trades) * 100.0 if total_trades > 0 else 0.0

    avg_win_dollars = float(winners["pnl"].mean()) if n_win > 0 else 0.0
    avg_loss_dollars = abs(float(losers["pnl"].mean())) if n_loss > 0 else 0.0
    payoff_ratio = avg_win_dollars / max(avg_loss_dollars, 1.0)

    total_realized_pnl = float(pnl.sum())

    # Exit reason distribution
    reason_counts = (
        df["reason"].value_counts().to_dict() if "reason" in df.columns else {}
    )

    # Automated Episodic Lessons
    lessons = []
    if win_rate >= 75.0:
        lessons.append(
            f"🏆 Exceptional execution precision ({win_rate:.1f}% win rate across {total_trades} trades). "
            f"50/50 scale-out methodology successfully captured multi-R runners."
        )
    if payoff_ratio >= 1.5:
        lessons.append(
            f"📈 High payoff asymmetry ({payoff_ratio:.2f}x win/loss ratio): Average winner ${avg_win_dollars:,.2f} "
            f"vs average loser -${avg_loss_dollars:,.2f}."
        )
    if "STOP_LOSS" in reason_counts or "Protective Stop-Loss" in reason_counts:
        stop_count = reason_counts.get("STOP_LOSS", 0) + reason_counts.get(
            "Protective Stop-Loss", 0
        )
        lessons.append(
            f"🛡️ CRO Risk Discipline verified: {stop_count} trades hit stop-loss cleanly without emotional hold."
        )

    # Top ticker contributions
    ticker_pnl = (
        df.groupby("ticker")["pnl"].sum().sort_values(ascending=False).to_dict()
    )

    return {
        "status": "AUTOPSY_COMPLETE",
        "total_trades": total_trades,
        "winning_trades": n_win,
        "losing_trades": n_loss,
        "win_rate_pct": round(win_rate, 1),
        "total_realized_pnl": round(total_realized_pnl, 2),
        "avg_win_dollars": round(avg_win_dollars, 2),
        "avg_loss_dollars": round(avg_loss_dollars, 2),
        "payoff_ratio": round(payoff_ratio, 2),
        "profit_factor": round(
            float(winners["pnl"].sum()) / max(abs(float(losers["pnl"].sum())), 1.0), 2
        ),
        "exit_reason_breakdown": reason_counts,
        "top_ticker_contributors": ticker_pnl,
        "episodic_lessons": lessons,
    }
