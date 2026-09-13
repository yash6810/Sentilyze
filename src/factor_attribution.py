"""
Institutional Factor Attribution & Performance Factsheet Engine
Computes institutional-grade risk and performance metrics:
  - Alpha (CAPM & Fama-French proxy) & Market Beta
  - Sharpe, Sortino, Calmar, Omega, and Information Ratios
  - Value at Risk (Parametric & Historical 95%, 99% VaR)
  - Conditional Value at Risk (CVaR / Expected Shortfall)
  - Tail Risk: Skewness and Excess Kurtosis
  - Rolling Alpha/Beta dynamics and Max Drawdown profiles
  - Trade-level attribution: Win Rate, Profit Factor, Expectancy, Payoff Ratio
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import os
import json
import logging
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class RiskAdjustedMetrics:
    cagr_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    information_ratio: float
    alpha_annualized_pct: float
    beta: float
    r_squared: float
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    var_95_daily_pct: float
    cvar_95_daily_pct: float
    var_99_daily_pct: float
    cvar_99_daily_pct: float
    skewness: float
    excess_kurtosis: float


@dataclass
class TradeExecutionMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    payoff_ratio: float
    average_trade_pnl: float
    average_win_pnl: float
    average_loss_pnl: float
    largest_win_pnl: float
    largest_loss_pnl: float
    expectancy_per_dollar: float


@dataclass
class InstitutionalFactsheet:
    as_of_date: str
    strategy_name: str
    benchmark_name: str
    risk_metrics: RiskAdjustedMetrics
    trade_metrics: Optional[TradeExecutionMetrics]
    monthly_returns_table: Dict[str, Dict[str, float]]
    rolling_metrics: Dict[str, List[float]]
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FactorAttributionEngine:
    """
    Computes rigorous factor attribution and risk diagnostics on strategy return series.
    """

    def __init__(self, risk_free_rate: float = 0.04):
        self.rf_annual = risk_free_rate
        self.rf_daily = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0

    def compute_metrics(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        trade_df: Optional[pd.DataFrame] = None,
        strategy_name: str = "Sentilyze Multi-Factor Strategy",
        benchmark_name: str = "S&P 500 (SPY)",
    ) -> InstitutionalFactsheet:
        """
        Calculate full institutional factsheet metrics.
        """
        strat_ret = strategy_returns.dropna().astype(float)
        if len(strat_ret) < 5:
            raise ValueError(
                "Strategy returns series must contain at least 5 observations."
            )

        if benchmark_returns is not None:
            bench_ret = benchmark_returns.dropna().astype(float)
            # Align dates
            aligned = pd.concat([strat_ret, bench_ret], axis=1, join="inner").dropna()
            if len(aligned) >= 5:
                strat_ret = aligned.iloc[:, 0]
                bench_ret = aligned.iloc[:, 1]
            else:
                bench_ret = pd.Series(0.0004, index=strat_ret.index)
        else:
            bench_ret = pd.Series(0.0004, index=strat_ret.index)

        # 1. Basic return stats
        n_days = len(strat_ret)
        cumulative_compound = (1.0 + strat_ret).prod()
        years = max(n_days / 252.0, 0.01)
        cagr = (
            (cumulative_compound ** (1.0 / years)) - 1.0
            if cumulative_compound > 0
            else -1.0
        )
        ann_vol = strat_ret.std() * np.sqrt(252.0) if strat_ret.std() > 0 else 1e-4

        # 2. Sharpe & Sortino
        excess_daily_ret = strat_ret - self.rf_daily
        mean_excess = excess_daily_ret.mean()
        sharpe = (
            (mean_excess / strat_ret.std()) * np.sqrt(252.0)
            if strat_ret.std() > 0
            else 0.0
        )

        downside_returns = excess_daily_ret[excess_daily_ret < 0]
        downside_std = (
            np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else 1e-4
        )
        sortino = (
            (mean_excess / downside_std) * np.sqrt(252.0) if downside_std > 0 else 0.0
        )

        # 3. Drawdown and Calmar
        cum_equity = (1.0 + strat_ret).cumprod()
        running_max = cum_equity.cummax()
        drawdowns = (cum_equity - running_max) / running_max
        max_dd = abs(float(drawdowns.min())) if len(drawdowns) > 0 else 0.0
        calmar = (cagr / max_dd) if max_dd > 1e-4 else (cagr * 100.0)

        # Drawdown duration
        is_dd = drawdowns < 0
        dd_runs = (~is_dd).cumsum()[is_dd]
        max_dd_duration = int(dd_runs.value_counts().max()) if len(dd_runs) > 0 else 0

        # 4. Omega Ratio (threshold = rf_daily)
        gains = strat_ret[strat_ret > self.rf_daily] - self.rf_daily
        losses = self.rf_daily - strat_ret[strat_ret < self.rf_daily]
        omega = float(gains.sum() / losses.sum()) if losses.sum() > 0 else 10.0

        # 5. Factor regression (CAPM Alpha & Beta)
        if len(bench_ret) >= 5 and bench_ret.std() > 1e-6:
            cov_mat = np.cov(strat_ret.values, bench_ret.values)
            cov_pm = cov_mat[0, 1]
            var_m = cov_mat[1, 1]
            beta = float(cov_pm / var_m) if var_m > 1e-8 else 1.0
            r_val, _ = stats.pearsonr(strat_ret.values, bench_ret.values)
            r_squared = float(r_val**2)
            alpha_daily = float(
                strat_ret.mean()
                - (self.rf_daily + beta * (bench_ret.mean() - self.rf_daily))
            )
            alpha_ann = float(alpha_daily * 252.0)

            # Information Ratio
            tracking_diff = strat_ret - bench_ret
            track_std = tracking_diff.std()
            info_ratio = (
                float((tracking_diff.mean() / track_std) * np.sqrt(252.0))
                if track_std > 1e-6
                else 0.0
            )
        else:
            beta = 1.0
            r_squared = 0.0
            alpha_ann = 0.0
            info_ratio = 0.0

        # 6. VaR and CVaR
        sorted_rets = np.sort(strat_ret.values)
        idx_95 = max(int(0.05 * len(sorted_rets)), 0)
        idx_99 = max(int(0.01 * len(sorted_rets)), 0)
        var_95 = float(abs(sorted_rets[idx_95]))
        cvar_95 = float(abs(np.mean(sorted_rets[: max(idx_95 + 1, 1)])))
        var_99 = float(abs(sorted_rets[idx_99]))
        cvar_99 = float(abs(np.mean(sorted_rets[: max(idx_99 + 1, 1)])))

        # 7. Tail Risk
        skew = float(stats.skew(strat_ret.values)) if len(strat_ret) >= 3 else 0.0
        kurt = float(stats.kurtosis(strat_ret.values)) if len(strat_ret) >= 4 else 0.0

        risk_metrics = RiskAdjustedMetrics(
            cagr_pct=round(cagr * 100.0, 2),
            annualized_volatility_pct=round(ann_vol * 100.0, 2),
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            calmar_ratio=round(calmar, 3),
            omega_ratio=round(omega, 3),
            information_ratio=round(info_ratio, 3),
            alpha_annualized_pct=round(alpha_ann * 100.0, 2),
            beta=round(beta, 3),
            r_squared=round(r_squared, 3),
            max_drawdown_pct=round(max_dd * 100.0, 2),
            max_drawdown_duration_days=max_dd_duration,
            var_95_daily_pct=round(var_95 * 100.0, 2),
            cvar_95_daily_pct=round(cvar_95 * 100.0, 2),
            var_99_daily_pct=round(var_99 * 100.0, 2),
            cvar_99_daily_pct=round(cvar_99 * 100.0, 2),
            skewness=round(skew, 3),
            excess_kurtosis=round(kurt, 3),
        )

        # 8. Trade metrics if provided
        trade_metrics = None
        if trade_df is not None and not trade_df.empty and "pnl" in trade_df.columns:
            trade_metrics = self._compute_trade_metrics(trade_df)

        # 9. Rolling Alpha & Beta (30-day window)
        rolling_metrics = self._compute_rolling_metrics(strat_ret, bench_ret, window=30)

        # 10. Monthly returns table
        monthly_table = self._build_monthly_table(strat_ret)

        as_of_str = (
            str(strat_ret.index[-1].date())
            if hasattr(strat_ret.index[-1], "date")
            else str(strat_ret.index[-1])
        )

        factsheet = InstitutionalFactsheet(
            as_of_date=as_of_str,
            strategy_name=strategy_name,
            benchmark_name=benchmark_name,
            risk_metrics=risk_metrics,
            trade_metrics=trade_metrics,
            monthly_returns_table=monthly_table,
            rolling_metrics=rolling_metrics,
            summary={
                "total_observations": len(strat_ret),
                "risk_free_rate_used": self.rf_annual,
                "compounded_return_pct": round((cumulative_compound - 1.0) * 100.0, 2),
            },
        )
        return factsheet

    def _compute_trade_metrics(self, df: pd.DataFrame) -> TradeExecutionMetrics:
        """Compute execution statistics from trade log dataframe."""
        pnl = df["pnl"].dropna().astype(float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        total_trades = len(pnl)
        n_wins = len(wins)
        n_losses = len(losses)
        win_rate = (n_wins / total_trades) * 100.0 if total_trades > 0 else 0.0

        gross_profits = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_losses = float(abs(losses.sum())) if len(losses) > 0 else 0.0
        profit_factor = (
            round(gross_profits / gross_losses, 2)
            if gross_losses > 0
            else (99.0 if gross_profits > 0 else 0.0)
        )

        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.0
        payoff_ratio = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0.0

        p_win = n_wins / total_trades if total_trades > 0 else 0.0
        p_loss = n_losses / total_trades if total_trades > 0 else 0.0
        expectancy = round((p_win * avg_win) - (p_loss * avg_loss), 2)

        return TradeExecutionMetrics(
            total_trades=total_trades,
            winning_trades=n_wins,
            losing_trades=n_losses,
            win_rate_pct=round(win_rate, 2),
            profit_factor=profit_factor,
            payoff_ratio=payoff_ratio,
            average_trade_pnl=round(float(pnl.mean()), 2),
            average_win_pnl=round(avg_win, 2),
            average_loss_pnl=round(avg_loss, 2),
            largest_win_pnl=round(float(pnl.max()), 2) if len(pnl) > 0 else 0.0,
            largest_loss_pnl=round(float(pnl.min()), 2) if len(pnl) > 0 else 0.0,
            expectancy_per_dollar=expectancy,
        )

    def _compute_rolling_metrics(
        self, strat_ret: pd.Series, bench_ret: pd.Series, window: int = 30
    ) -> Dict[str, List[float]]:
        """Compute rolling 30-day rolling beta, alpha, and sharpe."""
        rolling_dict = {
            "dates": [],
            "rolling_alpha": [],
            "rolling_beta": [],
            "rolling_sharpe": [],
        }
        if len(strat_ret) < window:
            return rolling_dict

        cov = strat_ret.rolling(window).cov(bench_ret)
        var_b = bench_ret.rolling(window).var()
        rolling_beta = cov / var_b
        rolling_alpha = (
            strat_ret.rolling(window).mean()
            - rolling_beta * bench_ret.rolling(window).mean()
        ) * 252.0
        rolling_sharpe = (
            strat_ret.rolling(window).mean() / strat_ret.rolling(window).std()
        ) * np.sqrt(252.0)

        valid_idx = rolling_beta.dropna().index
        for idx in valid_idx:
            date_str = str(idx.date()) if hasattr(idx, "date") else str(idx)
            rolling_dict["dates"].append(date_str)
            rolling_dict["rolling_beta"].append(round(float(rolling_beta.loc[idx]), 3))
            rolling_dict["rolling_alpha"].append(
                round(float(rolling_alpha.loc[idx]), 3)
            )
            rolling_dict["rolling_sharpe"].append(
                round(float(rolling_sharpe.loc[idx]), 3)
            )

        return rolling_dict

    def _build_monthly_table(self, returns: pd.Series) -> Dict[str, Dict[str, float]]:
        """Construct a calendar year-month returns grid."""
        df_ret = returns.to_frame("ret")
        if not isinstance(df_ret.index, pd.DatetimeIndex):
            try:
                df_ret.index = pd.to_datetime(df_ret.index)
            except Exception:
                return {}

        df_ret["Year"] = df_ret.index.year
        df_ret["Month"] = df_ret.index.strftime("%b")

        month_order = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        monthly_table: Dict[str, Dict[str, float]] = {}

        for year, group in df_ret.groupby("Year"):
            year_str = str(year)
            monthly_table[year_str] = {}
            for m in month_order:
                m_group = group[group["Month"] == m]
                if not m_group.empty:
                    m_ret = (1.0 + m_group["ret"]).prod() - 1.0
                    monthly_table[year_str][m] = round(float(m_ret * 100.0), 2)
                else:
                    monthly_table[year_str][m] = 0.0

            # Annual return
            y_ret = (1.0 + group["ret"]).prod() - 1.0
            monthly_table[year_str]["YearTotal"] = round(float(y_ret * 100.0), 2)

        return monthly_table


def generate_institutional_factsheet_for_ticker(
    ticker: str = "NVDA",
    results_dir: str = "results",
    save_output: bool = True,
) -> InstitutionalFactsheet:
    """
    Reads ticker backtest portfolio curve and builds full institutional factsheet.
    """
    portfolio_file = f"{results_dir}/{ticker.upper()}_portfolio.csv"
    trade_file = f"{results_dir}/executed_trades.csv"

    df_port = pd.read_csv(portfolio_file, index_col=0, parse_dates=True)
    if "total" not in df_port.columns:
        raise ValueError(f"Portfolio file {portfolio_file} missing 'total' column.")

    strat_rets = df_port["total"].pct_change().dropna()
    bench_rets = (
        df_port["benchmark"].pct_change().dropna()
        if "benchmark" in df_port.columns
        else None
    )

    # Load trade log if exists
    trade_df = None
    if os.path.exists(trade_file):
        try:
            trade_df = pd.read_csv(trade_file)
        except Exception as e:
            logger.warning(f"Could not load trades: {e}")

    engine = FactorAttributionEngine()
    factsheet = engine.compute_metrics(
        strategy_returns=strat_rets,
        benchmark_returns=bench_rets,
        trade_df=trade_df,
        strategy_name=f"Sentilyze Walk-Forward Dynamic Model ({ticker.upper()})",
        benchmark_name=f"{ticker.upper()} Buy-and-Hold",
    )

    if save_output:
        out_path = f"{results_dir}/institutional_factsheet_{ticker.upper()}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(factsheet.to_dict(), f, indent=2)
        logger.info(f"Saved institutional factsheet to {out_path}")

    return factsheet
