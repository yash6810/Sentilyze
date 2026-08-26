import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from src.utils import get_logger

logger = get_logger(__name__)

PORTFOLIO_FILE = os.path.join("results", "paper_portfolio.json")
INITIAL_CAPITAL = 100000.00


class PaperBroker:
    """
    Autonomous Virtual Paper Trading Broker & Portfolio Tracker.
    Simulates institutional trade execution, dynamic position sizing,
    Take-Profit / Stop-Loss management, and real-time PnL accounting with $100,000 starting cash.
    """

    def __init__(self, portfolio_path: str = PORTFOLIO_FILE, initial_cash: float = INITIAL_CAPITAL):
        self.portfolio_path = portfolio_path
        self.initial_cash = initial_cash
        self.state = self._load_or_initialize()

    def _load_or_initialize(self) -> Dict[str, Any]:
        """Loads existing portfolio state from JSON or initializes a fresh $100k account."""
        if os.path.exists(self.portfolio_path):
            try:
                with open(self.portfolio_path, "r") as f:
                    data = json.load(f)
                    logger.info(f"Loaded paper portfolio from {self.portfolio_path} (Total Equity: ${data.get('total_equity', self.initial_cash):,.2f})")
                    return data
            except Exception as e:
                logger.error(f"Error loading portfolio state ({e}). Re-initializing fresh account.")

        now_str = datetime.now(timezone.utc).isoformat()
        initial_state = {
            "initial_capital": self.initial_cash,
            "cash": self.initial_cash,
            "total_equity": self.initial_cash,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "open_positions": {},  # {ticker: {shares, entry_price, current_price, entry_date, tp_target, sl_target, confidence, atr}}
            "closed_trades": [],   # [{ticker, shares, entry_price, exit_price, entry_date, exit_date, pnl, return_pct, reason}]
            "equity_history": [
                {
                    "date": now_str[:10],
                    "timestamp": now_str,
                    "total_equity": self.initial_cash,
                    "cash": self.initial_cash,
                    "invested": 0.0,
                    "daily_return": 0.0
                }
            ],
            "last_updated": now_str
        }
        self._save(initial_state)
        return initial_state

    def _save(self, state: Optional[Dict[str, Any]] = None):
        """Persists portfolio ledger to disk."""
        save_data = state or self.state
        save_data["last_updated"] = datetime.now(timezone.utc).isoformat()
        os.makedirs(os.path.dirname(self.portfolio_path), exist_ok=True)
        with open(self.portfolio_path, "w") as f:
            json.dump(save_data, f, indent=2)

    def execute_daily_signals(self, signals_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes daily quantitative scan results:
        1. Evaluates open positions for Take-Profit or Stop-Loss hits on latest prices.
        2. Closes positions if Model Signal turned SELL.
        3. Opens new positions for high-conviction BUY signals using risk-controlled allocation.
        """
        now_utc = datetime.now(timezone.utc)
        now_str = now_utc.isoformat()
        date_str = now_str[:10]

        executed_actions = {
            "buys": [],
            "take_profits": [],
            "stop_losses": [],
            "sells": []
        }

        signals_by_ticker = {s["ticker"]: s for s in signals_list}

        # Step 1: Manage Open Positions (TP / SL / Model SELL)
        open_tickers = list(self.state["open_positions"].keys())
        for ticker in open_tickers:
            pos = self.state["open_positions"][ticker]
            signal_data = signals_by_ticker.get(ticker)
            curr_price = float(signal_data["current_price"]) if signal_data else float(pos["current_price"])
            pos["current_price"] = curr_price

            shares = pos["shares"]
            entry_price = pos["entry_price"]
            tp_target = pos["tp_target"]
            sl_target = pos["sl_target"]

            exit_reason = None

            # Check Take-Profit Trigger
            if curr_price >= tp_target:
                exit_reason = "TAKE_PROFIT"
            # Check Stop-Loss Trigger
            elif curr_price <= sl_target:
                exit_reason = "STOP_LOSS"
            # Check Model Exit
            elif signal_data and signal_data.get("signal") == "SELL":
                exit_reason = "MODEL_SELL"

            if exit_reason:
                proceeds = float(shares * curr_price)
                cost_basis = float(shares * entry_price)
                pnl = float(proceeds - cost_basis)
                return_pct = float((curr_price - entry_price) / entry_price * 100.0)

                self.state["cash"] += proceeds
                self.state["realized_pnl"] += pnl
                self.state["total_trades"] += 1
                if pnl > 0:
                    self.state["winning_trades"] += 1
                else:
                    self.state["losing_trades"] += 1

                trade_record = {
                    "ticker": ticker,
                    "shares": shares,
                    "entry_price": entry_price,
                    "exit_price": curr_price,
                    "entry_date": pos["entry_date"],
                    "exit_date": date_str,
                    "pnl": round(pnl, 2),
                    "return_pct": round(return_pct, 2),
                    "reason": exit_reason
                }
                self.state["closed_trades"].append(trade_record)
                del self.state["open_positions"][ticker]

                if exit_reason == "TAKE_PROFIT":
                    executed_actions["take_profits"].append(trade_record)
                elif exit_reason == "STOP_LOSS":
                    executed_actions["stop_losses"].append(trade_record)
                else:
                    executed_actions["sells"].append(trade_record)

                logger.info(f"Closed position in {ticker} ({exit_reason}): PnL: ${pnl:+,.2f} ({return_pct:+.2f}%)")

        # Step 2: Open New Positions for BUY Signals
        buy_signals = [s for s in signals_list if s["signal"] == "BUY" and s["ticker"] not in self.state["open_positions"]]
        # Sort BUY signals by confidence descending
        buy_signals = sorted(buy_signals, key=lambda x: x.get("confidence", 0), reverse=True)

        if buy_signals and self.state["cash"] > 2000.0:
            # Allocate up to max 5 concurrent positions, sizing each position ~$15k-$20k or equal cash share
            max_new_positions = max(1, 5 - len(self.state["open_positions"]))
            target_buys = buy_signals[:max_new_positions]

            allocation_per_stock = min(self.state["cash"] * 0.95 / len(target_buys), 20000.0)

            for s in target_buys:
                ticker = s["ticker"]
                price = float(s["current_price"])
                if price <= 0:
                    continue

                shares = int(allocation_per_stock // price)
                if shares <= 0:
                    continue

                cost = float(shares * price)
                if cost > self.state["cash"]:
                    continue

                self.state["cash"] -= cost
                self.state["open_positions"][ticker] = {
                    "shares": shares,
                    "entry_price": price,
                    "current_price": price,
                    "entry_date": date_str,
                    "tp_target": float(s.get("take_profit", price * 1.06)),
                    "sl_target": float(s.get("stop_loss", price * 0.95)),
                    "confidence": float(s.get("confidence", 0.5)),
                    "regime": s.get("regime", "BULLISH")
                }

                buy_record = {
                    "ticker": ticker,
                    "shares": shares,
                    "entry_price": price,
                    "cost": round(cost, 2),
                    "entry_date": date_str,
                    "tp_target": round(float(s.get("take_profit", price * 1.06)), 2),
                    "sl_target": round(float(s.get("stop_loss", price * 0.95)), 2),
                    "confidence": round(float(s.get("confidence", 0.5)) * 100, 1)
                }
                executed_actions["buys"].append(buy_record)
                logger.info(f"Opened new position in {ticker}: {shares} shares @ ${price:.2f} (Total: ${cost:,.2f})")

        # Step 3: Recalculate Portfolio Values
        self._recalculate_metrics(date_str, now_str)
        self._save()
        return executed_actions

    def _recalculate_metrics(self, date_str: str, now_str: str):
        """Updates total equity, unrealized PnL, and win rates."""
        invested_val = sum(
            pos["shares"] * pos["current_price"] for pos in self.state["open_positions"].values()
        )
        cost_basis = sum(
            pos["shares"] * pos["entry_price"] for pos in self.state["open_positions"].values()
        )
        self.state["unrealized_pnl"] = round(invested_val - cost_basis, 2)
        total_equity = round(self.state["cash"] + invested_val, 2)
        prev_equity = self.state["equity_history"][-1]["total_equity"] if self.state["equity_history"] else self.initial_cash
        daily_return = round(((total_equity - prev_equity) / prev_equity) * 100.0, 2) if prev_equity > 0 else 0.0

        self.state["total_equity"] = total_equity

        total_closed = self.state["total_trades"]
        self.state["win_rate"] = round((self.state["winning_trades"] / total_closed) * 100.0, 1) if total_closed > 0 else 0.0

        # Append or update today's equity history point
        if self.state["equity_history"] and self.state["equity_history"][-1]["date"] == date_str:
            self.state["equity_history"][-1]["total_equity"] = total_equity
            self.state["equity_history"][-1]["cash"] = round(self.state["cash"], 2)
            self.state["equity_history"][-1]["invested"] = round(invested_val, 2)
            self.state["equity_history"][-1]["daily_return"] = daily_return
        else:
            self.state["equity_history"].append(
                {
                    "date": date_str,
                    "timestamp": now_str,
                    "total_equity": total_equity,
                    "cash": round(self.state["cash"], 2),
                    "invested": round(invested_val, 2),
                    "daily_return": daily_return
                }
            )

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Returns high-level KPI metrics for the portfolio dashboard."""
        return {
            "total_equity": self.state["total_equity"],
            "cash": self.state["cash"],
            "invested": self.state["total_equity"] - self.state["cash"],
            "unrealized_pnl": self.state["unrealized_pnl"],
            "realized_pnl": self.state["realized_pnl"],
            "total_pnl": round(self.state["total_equity"] - self.initial_cash, 2),
            "total_return_pct": round(((self.state["total_equity"] - self.initial_cash) / self.initial_cash) * 100.0, 2),
            "win_rate": self.state["win_rate"],
            "open_positions_count": len(self.state["open_positions"]),
            "total_trades": self.state["total_trades"],
            "winning_trades": self.state["winning_trades"],
            "losing_trades": self.state["losing_trades"]
        }

    def get_open_positions_df(self) -> pd.DataFrame:
        """Returns a DataFrame of current open holdings."""
        if not self.state["open_positions"]:
            return pd.DataFrame()
        rows = []
        for ticker, pos in self.state["open_positions"].items():
            entry_p = pos["entry_price"]
            curr_p = pos["current_price"]
            shares = pos["shares"]
            cost = shares * entry_p
            curr_val = shares * curr_p
            pnl = curr_val - cost
            ret_pct = (curr_p - entry_p) / entry_p * 100.0
            rows.append({
                "Ticker": ticker,
                "Shares": shares,
                "Entry Price": f"${entry_p:,.2f}",
                "Current Price": f"${curr_p:,.2f}",
                "Position Value": f"${curr_val:,.2f}",
                "Unrealized PnL ($)": f"${pnl:+,.2f}",
                "Return (%)": f"{ret_pct:+.2f}%",
                "Take-Profit Target": f"${pos['tp_target']:,.2f}",
                "Stop-Loss Target": f"${pos['sl_target']:,.2f}",
                "Regime": pos.get("regime", "N/A"),
                "Entry Date": pos["entry_date"]
            })
        return pd.DataFrame(rows)

    def get_closed_trades_df(self) -> pd.DataFrame:
        """Returns a DataFrame of trade history."""
        if not self.state["closed_trades"]:
            return pd.DataFrame()
        df = pd.DataFrame(self.state["closed_trades"])
        df.rename(columns={
            "ticker": "Ticker",
            "shares": "Shares",
            "entry_price": "Entry Price ($)",
            "exit_price": "Exit Price ($)",
            "entry_date": "Entry Date",
            "exit_date": "Exit Date",
            "pnl": "Net PnL ($)",
            "return_pct": "Return (%)",
            "reason": "Exit Reason"
        }, inplace=True)
        return df

    def get_equity_curve_df(self) -> pd.DataFrame:
        """Returns equity history as a DatetimeIndex DataFrame."""
        if not self.state["equity_history"]:
            return pd.DataFrame()
        df = pd.DataFrame(self.state["equity_history"])
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df
