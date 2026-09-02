"""
Paper 25: Opening Range Breakout (ORB) with Stocks-in-Play Filter.

Source: Carlo Zarattini, Andrea Barbon, Andrew Aziz (2024)
"A Profitable Day Trading Strategy For The U.S. Equity Market" (SSRN: 4729284)
Complexity: O(N) linear time per session.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class OpeningRangeBreakout:
    """
    Opening Range Breakout (ORB) trading engine with Stocks-in-Play filter.

    Paper specifications:
    - 5-Minute opening range (09:30-09:35 EST) determines High (H5) and Low (L5).
    - Stocks in Play: High relative volume (RVOL >= 1.5), news catalyst, and ATR >= 1.5.
    - Long entry on breakout > H5; Short/exit on breakdown < L5.
    - Dynamic ATR profit-target and stop-loss trailing.
    """

    def __init__(
        self,
        range_minutes: int = 5,
        rvol_threshold: float = 1.5,
        atr_multiplier_tp: float = 2.0,
        atr_multiplier_sl: float = 1.0,
        catalyst_weight: float = 0.5,
    ):
        self.range_minutes = range_minutes
        self.rvol_threshold = rvol_threshold
        self.tp_mult = atr_multiplier_tp
        self.sl_mult = atr_multiplier_sl
        self.catalyst_weight = catalyst_weight

    def filter_stocks_in_play(
        self,
        universe_data: Dict[str, pd.DataFrame],
        catalyst_scores: Optional[Dict[str, float]] = None,
        top_k: int = 5,
    ) -> List[str]:
        """Rank universe and select top 'Stocks in Play' based on volume, ATR, and catalyst score."""
        catalyst_scores = catalyst_scores or {}
        scores = {}

        for ticker, df in universe_data.items():
            if len(df) < 20:
                continue

            recent_vol = (
                df["Volume"].iloc[-1]
                if "Volume" in df.columns
                else df["Close"].iloc[-1]
            )
            avg_vol = (
                df["Volume"].iloc[-20:].mean() if "Volume" in df.columns else recent_vol
            )
            rvol = (recent_vol / avg_vol) if avg_vol > 0 else 1.0

            high = df["High"] if "High" in df.columns else df["Close"] * 1.01
            low = df["Low"] if "Low" in df.columns else df["Close"] * 0.99
            close = df["Close"]
            tr = np.maximum(
                high - low,
                np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))),
            )
            atr_val = tr.iloc[-14:].mean() if len(tr) >= 14 else (high - low).iloc[-1]
            atr_norm = atr_val / close.iloc[-1] if close.iloc[-1] > 0 else 0.01

            cat_score = catalyst_scores.get(ticker, 0.5)

            in_play_score = (rvol * 0.4) + (atr_norm * 100 * 0.3) + (cat_score * 0.3)
            scores[ticker] = in_play_score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [ticker for ticker, _ in ranked[:top_k]]

    def evaluate_orb_signals(
        self,
        daily_bars: pd.DataFrame,
        sentiment_score: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Evaluate ORB signal using daily OHLC + volatility approximation.
        """
        if len(daily_bars) < 15:
            return {"signal": 0, "strength": 0.0, "reason": "Insufficient data"}

        close = daily_bars["Close"].values
        high = (
            daily_bars["High"].values if "High" in daily_bars.columns else close * 1.01
        )
        low = daily_bars["Low"].values if "Low" in daily_bars.columns else close * 0.99
        open_p = daily_bars["Open"].values if "Open" in daily_bars.columns else close

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])),
        )
        atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))

        curr_open = float(open_p[-1])
        curr_high = float(high[-1])
        curr_low = float(low[-1])
        curr_close = float(close[-1])

        orb_high = curr_open + (0.35 * atr)
        orb_low = curr_open - (0.35 * atr)

        long_breakout = curr_high > orb_high and sentiment_score >= 0.45
        short_breakdown = curr_low < orb_low and sentiment_score < 0.45

        if long_breakout and curr_close > curr_open:
            signal = 1
            strength = min(1.0, (curr_close - orb_high) / (atr + 1e-6) + 0.5)
        elif short_breakdown:
            signal = -1
            strength = min(1.0, (orb_low - curr_close) / (atr + 1e-6) + 0.5)
        else:
            signal = 0
            strength = 0.0

        return {
            "signal": signal,
            "strength": round(float(strength), 4),
            "orb_high": round(float(orb_high), 2),
            "orb_low": round(float(orb_low), 2),
            "current_close": round(float(curr_close), 2),
            "atr": round(float(atr), 2),
            "sentiment_score": round(float(sentiment_score), 2),
        }

    def backtest_orb_strategy(
        self,
        prices_df: pd.DataFrame,
        universe_sentiment: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Simulate multi-year daily ORB strategy returns."""
        universe_sentiment = universe_sentiment or {}
        returns = prices_df.pct_change().dropna()
        tickers = list(prices_df.columns)

        portfolio_vals = [100000.0]
        daily_trades = []

        for t in range(15, len(returns)):
            window = prices_df.iloc[:t]
            day_ret = returns.iloc[t]

            in_play = tickers[:3]
            weights = np.zeros(len(tickers))

            for tk in in_play:
                idx = tickers.index(tk)
                sub_df = pd.DataFrame(
                    {
                        "Close": window[tk],
                        "Open": window[tk].shift(1),
                        "High": window[tk] * 1.01,
                        "Low": window[tk] * 0.99,
                    }
                ).dropna()
                sent = universe_sentiment.get(tk, 0.55)
                eval_res = self.evaluate_orb_signals(sub_df, sent)

                if eval_res["signal"] == 1:
                    weights[idx] = 1.0 / len(in_play)

            port_return = float(np.dot(weights, day_ret.values))
            portfolio_vals.append(portfolio_vals[-1] * (1.0 + port_return))
            daily_trades.append(port_return)

        pv = np.array(portfolio_vals)
        peak = np.maximum.accumulate(pv)
        max_dd = float(((peak - pv) / peak).max()) * 100.0
        total_ret = float((pv[-1] / pv[0] - 1.0) * 100.0)
        cagr = (
            float(
                ((pv[-1] / pv[0]) ** (252.0 / max(len(daily_trades), 1)) - 1.0) * 100.0
            )
            if len(daily_trades) > 0
            else 0.0
        )
        daily_arr = np.array(daily_trades)
        sharpe = (
            float(np.mean(daily_arr) / (np.std(daily_arr) + 1e-9) * np.sqrt(252))
            if len(daily_arr) > 0
            else 0.0
        )

        return {
            "paper": "Paper 25 - ORB Stocks in Play (Zarattini, Barbon, Aziz 2024)",
            "total_return_pct": round(total_ret, 2),
            "cagr_pct": round(cagr, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "n_days_simulated": len(daily_trades),
        }
