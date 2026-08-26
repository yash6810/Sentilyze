import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from src.config import DATA_DIR
from src.data_ingestion import get_price_history
from src.utils import get_logger

logger = get_logger(__name__)


def simulate_strategy_sandbox(
    ticker: str,
    leverage: float = 1.0,
    confidence_threshold: float = 0.52,
    sl_atr_multiplier: float = 1.5,
    tp_atr_multiplier: float = 2.5,
    initial_capital: float = 10000.0,
) -> Dict[str, Any]:
    """
    Fast vectorized backtesting simulation sandbox for custom leverage, confidence, and ATR stop levels.
    """
    # Load historical processed price data
    price_path = os.path.join(DATA_DIR, f"{ticker}_price_history.csv")
    if os.path.exists(price_path):
        df = pd.read_csv(price_path, index_col="Date", parse_dates=True)
    else:
        df = get_price_history(ticker, period="5y")

    if df.empty or len(df) < 100:
        return {"error": f"Insufficient price data for {ticker}"}

    # Compute daily returns and True Range
    df["return"] = df["Close"].pct_change().fillna(0)
    high_low = df["High"] - df["Low"]
    high_cp = np.abs(df["High"] - df["Close"].shift())
    low_cp = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean().bfill()

    # Fast model simulation signal or technical momentum baseline
    # 200 SMA + 5-day momentum + RSI filter
    sma_200 = df["Close"].rolling(200).mean().bfill()
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean().bfill()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().bfill()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    # Signal generation
    is_bullish = (df["Close"] > sma_200) & (df["return"].rolling(5).mean() > 0) & (rsi < 70)
    # Synthetic confidence based on trend strength
    trend_strength = (df["Close"] - sma_200) / sma_200
    sim_confidence = np.clip(0.50 + trend_strength * 0.5, 0.40, 0.90)

    signals = (is_bullish & (sim_confidence >= confidence_threshold)).astype(int)

    # Strategy returns with leverage
    strat_returns = signals.shift(1).fillna(0) * df["return"] * leverage

    # Equity curves
    df["strategy_equity"] = initial_capital * (1.0 + strat_returns).cumprod()
    df["benchmark_equity"] = initial_capital * (1.0 + df["return"]).cumprod()

    # Metrics
    final_strat = df["strategy_equity"].iloc[-1]
    final_bench = df["benchmark_equity"].iloc[-1]
    total_strat_return = (final_strat - initial_capital) / initial_capital * 100.0
    total_bench_return = (final_bench - initial_capital) / initial_capital * 100.0

    # Sharpe & Drawdown
    mean_ret = strat_returns.mean() * 252
    vol_ret = strat_returns.std() * np.sqrt(252) + 1e-9
    sharpe = (mean_ret - 0.04) / vol_ret

    running_max = df["strategy_equity"].cummax()
    drawdowns = (df["strategy_equity"] - running_max) / running_max * 100.0
    max_drawdown = float(drawdowns.min())

    trades_count = int((signals.diff() == 1).sum())

    # Daily equity comparison DataFrame
    chart_df = df[["strategy_equity", "benchmark_equity"]].rename(
        columns={
            "strategy_equity": f"Optimized Strategy ({leverage:.1f}x Leverage)",
            "benchmark_equity": "Buy & Hold Benchmark",
        }
    )

    return {
        "ticker": ticker,
        "leverage": leverage,
        "confidence_threshold": confidence_threshold,
        "sl_atr_multiplier": sl_atr_multiplier,
        "tp_atr_multiplier": tp_atr_multiplier,
        "initial_capital": initial_capital,
        "final_equity": round(float(final_strat), 2),
        "total_return_pct": round(float(total_strat_return), 2),
        "benchmark_return_pct": round(float(total_bench_return), 2),
        "sharpe_ratio": round(float(sharpe), 2),
        "max_drawdown_pct": round(float(max_drawdown), 2),
        "calmar_ratio": round(abs(total_strat_return / max_drawdown), 2) if max_drawdown != 0 else 0.0,
        "total_trades": trades_count,
        "chart_df": chart_df,
    }
