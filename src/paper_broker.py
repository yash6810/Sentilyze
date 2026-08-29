import os
import json
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from src.utils import get_logger

logger = get_logger(__name__)

PORTFOLIO_FILE = os.path.join("results", "paper_portfolio.json")
INITIAL_CAPITAL = 100000.00


class PaperBroker:
    """
    Institutional Multi-Stage Quantitative Execution Broker ($100k Account).
    Features:
    1. Concentrated Conviction Allocation (Top-2 Highest AI signals, ~$45k each).
    2. 50/50 Scale-Out & Free Ride (50% profit take at +2.5 ATR, 50% runner at +4.5 ATR).
    3. Dynamic Break-Even & Trailing Profit Ratchets.
    4. Full PnL accounting, win-rate tracking, and equity curve persistence.
    """

    def __init__(
        self,
        portfolio_path: str = PORTFOLIO_FILE,
        initial_cash: float = INITIAL_CAPITAL,
    ):
        self.portfolio_path = portfolio_path
        self.initial_cash = initial_cash
        self.state = self._load_or_initialize()

    def _load_or_initialize(self) -> Dict[str, Any]:
        """Loads existing portfolio state from JSON or initializes a fresh $100k account."""
        if os.path.exists(self.portfolio_path):
            try:
                with open(self.portfolio_path, "r") as f:
                    data = json.load(f)
                    logger.info(
                        f"Loaded paper portfolio from {self.portfolio_path} (Total Equity: ${data.get('total_equity', self.initial_cash):,.2f})"
                    )
                    return data
            except Exception as e:
                logger.error(
                    f"Error loading portfolio state ({e}). Re-initializing fresh account."
                )

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
            "open_positions": {},
            "closed_trades": [],
            "equity_history": [
                {
                    "date": now_str[:10],
                    "timestamp": now_str,
                    "total_equity": self.initial_cash,
                    "cash": self.initial_cash,
                    "invested": 0.0,
                    "daily_return": 0.0,
                }
            ],
            "last_updated": now_str,
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

    def execute_daily_signals(
        self, signals_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Executes daily quantitative scan results using the Concentrated Top-2 + Scale-Out Model:
        1. Evaluates open holdings for TP1 (+2.5 ATR), TP2 (+4.5 ATR), Break-Even, and Stop-Loss.
        2. Closes positions if Model Signal turned SELL.
        3. Concentrates new BUY entries into the Top-2 highest conviction setups (~$45k each).
        """
        now_utc = datetime.now(timezone.utc)
        now_str = now_utc.isoformat()
        date_str = now_str[:10]

        executed_actions = {
            "buys": [],
            "take_profits": [],
            "take_profits_tp1": [],
            "take_profits_tp2": [],
            "stop_losses": [],
            "sells": [],
        }

        signals_by_ticker = {s["ticker"]: s for s in signals_list}
        closed_today_tickers = set()

        # Step 1: Manage Open Positions (Multi-Stage Exits)
        open_tickers = list(self.state["open_positions"].keys())
        for ticker in open_tickers:
            pos = self.state["open_positions"][ticker]
            signal_data = signals_by_ticker.get(ticker)
            curr_price = (
                float(signal_data["current_price"])
                if signal_data
                else float(pos["current_price"])
            )
            pos["current_price"] = curr_price

            shares = pos["shares"]
            entry_price = pos["entry_price"]
            tp1_target = pos["tp1_target"]
            tp2_target = pos["tp2_target"]
            sl_target = pos["sl_target"]
            scaled_out = pos.get("scaled_out", False)

            # Check Stage 1 Scale-Out (+2.5 ATR)
            if not scaled_out and curr_price >= tp1_target:
                half_shares = max(1, shares // 2)
                proceeds = float(half_shares * curr_price)
                cost_basis = float(half_shares * entry_price)
                pnl = float(proceeds - cost_basis)
                ret_pct = float((curr_price - entry_price) / entry_price * 100.0)

                self.state["cash"] += proceeds
                self.state["realized_pnl"] += pnl
                pos["shares"] = shares - half_shares
                pos["scaled_out"] = True
                # Move Stop-Loss to Break-Even + small buffer (Risk-Free Trade)
                pos["sl_target"] = round(entry_price * 1.002, 2)

                trade_record = {
                    "ticker": ticker,
                    "shares": half_shares,
                    "entry_price": entry_price,
                    "exit_price": curr_price,
                    "entry_date": pos["entry_date"],
                    "exit_date": date_str,
                    "pnl": round(pnl, 2),
                    "return_pct": round(ret_pct, 2),
                    "reason": "TAKE_PROFIT",
                    "scale_stage": "STAGE_1_50PCT",
                    "status": "RISK_FREE_RUNNER",
                }
                self.state["closed_trades"].append(trade_record)
                executed_actions["take_profits_tp1"].append(trade_record)
                executed_actions["take_profits"].append(trade_record)
                logger.info(
                    f"🎯 [STAGE 1 SCALE-OUT] Banked 50% of {ticker} ({half_shares} shares @ ${curr_price:.2f}) | PnL: ${pnl:+,.2f} ({ret_pct:+.2f}%) | SL Moved to Break-Even ${pos['sl_target']:.2f}"
                )

            # Check Stage 2 Runner Target (+4.5 ATR) on remaining shares
            elif scaled_out and curr_price >= tp2_target:
                proceeds = float(pos["shares"] * curr_price)
                cost_basis = float(pos["shares"] * entry_price)
                pnl = float(proceeds - cost_basis)
                ret_pct = float((curr_price - entry_price) / entry_price * 100.0)

                self.state["cash"] += proceeds
                self.state["realized_pnl"] += pnl
                self.state["total_trades"] += 1
                self.state["winning_trades"] += 1

                trade_record = {
                    "ticker": ticker,
                    "shares": pos["shares"],
                    "entry_price": entry_price,
                    "exit_price": curr_price,
                    "entry_date": pos["entry_date"],
                    "exit_date": date_str,
                    "pnl": round(pnl, 2),
                    "return_pct": round(ret_pct, 2),
                    "reason": "TAKE_PROFIT",
                    "scale_stage": "STAGE_2_RUNNER",
                }
                self.state["closed_trades"].append(trade_record)
                del self.state["open_positions"][ticker]
                closed_today_tickers.add(ticker)
                executed_actions["take_profits_tp2"].append(trade_record)
                executed_actions["take_profits"].append(trade_record)
                logger.info(
                    f"🏆 [STAGE 2 RUNNER EXIT] Closed remaining {ticker} ({trade_record['shares']} shares @ ${curr_price:.2f}) | PnL: ${pnl:+,.2f} ({ret_pct:+.2f}%)"
                )

            # Check Stop-Loss Trigger (or Break-Even stop)
            elif curr_price <= sl_target:
                proceeds = float(pos["shares"] * curr_price)
                cost_basis = float(pos["shares"] * entry_price)
                pnl = float(proceeds - cost_basis)
                ret_pct = float((curr_price - entry_price) / entry_price * 100.0)

                self.state["cash"] += proceeds
                self.state["realized_pnl"] += pnl
                self.state["total_trades"] += 1
                if (pnl > 0) or scaled_out:
                    self.state["winning_trades"] += 1
                else:
                    self.state["losing_trades"] += 1

                reason = "BREAK_EVEN_EXIT" if scaled_out else "STOP_LOSS"
                trade_record = {
                    "ticker": ticker,
                    "shares": pos["shares"],
                    "entry_price": entry_price,
                    "exit_price": curr_price,
                    "entry_date": pos["entry_date"],
                    "exit_date": date_str,
                    "pnl": round(pnl, 2),
                    "return_pct": round(ret_pct, 2),
                    "reason": reason,
                }
                self.state["closed_trades"].append(trade_record)
                del self.state["open_positions"][ticker]
                closed_today_tickers.add(ticker)
                executed_actions["stop_losses"].append(trade_record)
                logger.info(
                    f"🛑 [{reason}] Exited {ticker} @ ${curr_price:.2f} | PnL: ${pnl:+,.2f} ({ret_pct:+.2f}%)"
                )

            # Check Model Exit
            elif signal_data and signal_data.get("signal") == "SELL":
                proceeds = float(pos["shares"] * curr_price)
                cost_basis = float(pos["shares"] * entry_price)
                pnl = float(proceeds - cost_basis)
                ret_pct = float((curr_price - entry_price) / entry_price * 100.0)

                self.state["cash"] += proceeds
                self.state["realized_pnl"] += pnl
                self.state["total_trades"] += 1
                if pnl > 0:
                    self.state["winning_trades"] += 1
                else:
                    self.state["losing_trades"] += 1

                trade_record = {
                    "ticker": ticker,
                    "shares": pos["shares"],
                    "entry_price": entry_price,
                    "exit_price": curr_price,
                    "entry_date": pos["entry_date"],
                    "exit_date": date_str,
                    "pnl": round(pnl, 2),
                    "return_pct": round(ret_pct, 2),
                    "reason": "MODEL_SELL",
                }
                self.state["closed_trades"].append(trade_record)
                del self.state["open_positions"][ticker]
                closed_today_tickers.add(ticker)
                executed_actions["sells"].append(trade_record)
                logger.info(
                    f"🟡 [MODEL SELL] Closed {ticker} @ ${curr_price:.2f} | PnL: ${pnl:+,.2f}"
                )

        # Step 2: Open New Positions (Top-2 Concentrated Sizing, ~$45k each)
        buy_signals = [
            s
            for s in signals_list
            if s["signal"] == "BUY"
            and s["ticker"] not in self.state["open_positions"]
            and s["ticker"] not in closed_today_tickers
        ]
        buy_signals = sorted(
            buy_signals, key=lambda x: x.get("confidence", 0), reverse=True
        )

        max_allowed_positions = 2  # Focus capital into Top-2 highest conviction
        open_count = len(self.state["open_positions"])

        if (
            buy_signals
            and open_count < max_allowed_positions
            and self.state["cash"] > 5000.0
        ):
            slots_available = max_allowed_positions - open_count
            target_buys = buy_signals[:slots_available]

            # Allocate up to $45,000 per position (or equal available cash)
            allocation_per_stock = min(
                self.state["cash"] * 0.95 / len(target_buys), 45000.0
            )

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

                # ATR calculation for targets
                atr_val = float(s.get("take_profit", price * 1.06)) - price
                atr_base = max(
                    price * 0.025, atr_val / 2.5 if atr_val > 0 else price * 0.03
                )

                tp1_target = round(price + (2.5 * atr_base), 2)
                tp2_target = round(price + (4.5 * atr_base), 2)
                sl_target = round(price - (1.5 * atr_base), 2)

                self.state["open_positions"][ticker] = {
                    "shares": shares,
                    "initial_shares": shares,
                    "entry_price": price,
                    "current_price": price,
                    "entry_date": date_str,
                    "tp1_target": tp1_target,
                    "tp2_target": tp2_target,
                    "sl_target": sl_target,
                    "scaled_out": False,
                    "confidence": float(s.get("confidence", 0.5)),
                    "regime": s.get("regime", "BULLISH"),
                }

                buy_record = {
                    "ticker": ticker,
                    "shares": shares,
                    "entry_price": price,
                    "cost": round(cost, 2),
                    "entry_date": date_str,
                    "tp1_target": tp1_target,
                    "tp2_target": tp2_target,
                    "sl_target": sl_target,
                    "confidence": round(float(s.get("confidence", 0.5)) * 100, 1),
                }
                executed_actions["buys"].append(buy_record)
                logger.info(
                    f"🚀 [CONCENTRATED ENTRY] Bought {shares} shares of {ticker} @ ${price:.2f} (Total: ${cost:,.2f} | TP1: ${tp1_target:.2f} | TP2: ${tp2_target:.2f} | SL: ${sl_target:.2f})"
                )

        # Step 3: Recalculate Portfolio Values
        self._recalculate_metrics(date_str, now_str)
        self._save()
        return executed_actions

    def _recalculate_metrics(self, date_str: str, now_str: str):
        """Updates total equity, unrealized PnL, and win rates."""
        invested_val = sum(
            pos["shares"] * pos["current_price"]
            for pos in self.state["open_positions"].values()
        )
        cost_basis = sum(
            pos["shares"] * pos["entry_price"]
            for pos in self.state["open_positions"].values()
        )
        self.state["unrealized_pnl"] = round(invested_val - cost_basis, 2)
        total_equity = round(self.state["cash"] + invested_val, 2)
        prev_equity = (
            self.state["equity_history"][-1]["total_equity"]
            if self.state["equity_history"]
            else self.initial_cash
        )
        daily_return = (
            round(((total_equity - prev_equity) / prev_equity) * 100.0, 2)
            if prev_equity > 0
            else 0.0
        )

        self.state["total_equity"] = total_equity

        total_closed = self.state["total_trades"]
        self.state["win_rate"] = (
            round((self.state["winning_trades"] / total_closed) * 100.0, 1)
            if total_closed > 0
            else 0.0
        )

        # Append or update today's equity history point
        if (
            self.state["equity_history"]
            and self.state["equity_history"][-1]["date"] == date_str
        ):
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
                    "daily_return": daily_return,
                }
            )

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Returns high-level KPI metrics for the portfolio dashboard."""
        invested = max(0.0, self.state["total_equity"] - self.state["cash"])
        unrealized_pnl_pct = (
            round((self.state["unrealized_pnl"] / invested) * 100.0, 2)
            if invested > 0
            else 0.0
        )

        now_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        eq_history = self.state.get("equity_history", [])
        start_of_day_equity = float(self.initial_cash)
        if eq_history:
            prior_snapshots = [h for h in eq_history if h.get("date", "") < now_date]
            if prior_snapshots:
                start_of_day_equity = float(
                    prior_snapshots[-1].get("total_equity", self.initial_cash)
                )
            else:
                start_of_day_equity = float(
                    eq_history[0].get("total_equity", self.initial_cash)
                )

        daily_pnl = round(float(self.state["total_equity"]) - start_of_day_equity, 2)
        daily_return_pct = (
            round((daily_pnl / start_of_day_equity) * 100.0, 2)
            if start_of_day_equity > 0
            else 0.0
        )

        return {
            "total_equity": self.state["total_equity"],
            "cash": self.state["cash"],
            "invested": invested,
            "start_of_day_equity": start_of_day_equity,
            "daily_pnl": daily_pnl,
            "daily_return_pct": daily_return_pct,
            "unrealized_pnl": self.state["unrealized_pnl"],
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "realized_pnl": self.state["realized_pnl"],
            "total_pnl": round(self.state["total_equity"] - self.initial_cash, 2),
            "total_return_pct": round(
                ((self.state["total_equity"] - self.initial_cash) / self.initial_cash)
                * 100.0,
                2,
            ),
            "win_rate": self.state["win_rate"],
            "open_positions_count": len(self.state["open_positions"]),
            "total_trades": self.state["total_trades"],
            "winning_trades": self.state["winning_trades"],
            "losing_trades": self.state["losing_trades"],
        }

    def get_open_positions_df(self) -> pd.DataFrame:
        """Returns a DataFrame of current open holdings with Scale-Out status."""
        if not self.state["open_positions"]:
            return pd.DataFrame()
        rows = []
        from src.config import COMPANY_NAMES

        for ticker, pos in self.state["open_positions"].items():
            entry_p = pos["entry_price"]
            curr_p = pos["current_price"]
            shares = pos["shares"]
            cost = shares * entry_p
            curr_val = shares * curr_p
            pnl = curr_val - cost
            ret_pct = (curr_p - entry_p) / entry_p * 100.0
            scaled_out = pos.get("scaled_out", False)

            status_badge = (
                "🛡️ RISK-FREE (50% Banked)" if scaled_out else "⚡ ACTIVE 100%"
            )

            tp1_val = float(pos.get("tp1_target", entry_p * 1.06))
            tp0_val = float(
                pos.get("tp0_target", entry_p + (tp1_val - entry_p) * 0.40)
            )  # +1.0 ATR early bank target
            tp2_val = float(pos.get("tp2_target", entry_p * 1.12))
            sl_val = float(pos.get("sl_target", entry_p * 0.95))

            rows.append(
                {
                    "Ticker": ticker,
                    "Company Name": COMPANY_NAMES.get(ticker, ticker),
                    "Shares": shares,
                    "Entry Price": f"${entry_p:,.2f}",
                    "Current Price": f"${curr_p:,.2f}",
                    "Position Value": f"${curr_val:,.2f}",
                    "Unrealized PnL ($)": f"${pnl:+,.2f}",
                    "Return (%)": f"{ret_pct:+.2f}%",
                    "TP0 Early Bank (+1.0 ATR)": f"${tp0_val:,.2f}",
                    "TP1 Target (+2.5 ATR)": f"${tp1_val:,.2f}",
                    "TP2 Target (+4.5 ATR)": f"${tp2_val:,.2f}",
                    "Stop-Loss Target": f"${sl_val:,.2f}",
                    "Strategy Status": status_badge,
                    "Entry Date": pos["entry_date"],
                }
            )
        return pd.DataFrame(rows)

    def get_closed_trades_df(self) -> pd.DataFrame:
        """Returns a DataFrame of trade history with full company names."""
        if not self.state["closed_trades"]:
            return pd.DataFrame()
        from src.config import COMPANY_NAMES

        df = pd.DataFrame(self.state["closed_trades"])
        if "ticker" in df.columns:
            df["company_name"] = df["ticker"].map(lambda t: COMPANY_NAMES.get(t, t))

        df.rename(
            columns={
                "ticker": "Ticker",
                "company_name": "Company Name",
                "shares": "Shares",
                "entry_price": "Entry Price ($)",
                "exit_price": "Exit Price ($)",
                "entry_date": "Entry Date",
                "exit_date": "Exit Date",
                "pnl": "Net PnL ($)",
                "return_pct": "Return (%)",
                "reason": "Exit Reason",
            },
            inplace=True,
        )
        cols = ["Ticker", "Company Name"] + [
            c for c in df.columns if c not in ["Ticker", "Company Name"]
        ]
        return df[[c for c in cols if c in df.columns]]

    def get_equity_curve_df(self) -> pd.DataFrame:
        """Returns equity history as a DatetimeIndex DataFrame."""
        if not self.state["equity_history"]:
            return pd.DataFrame()
        df = pd.DataFrame(self.state["equity_history"])
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df

    def execute_manual_buy(
        self,
        ticker: str,
        shares: int,
        price: float,
        atr: Optional[float] = None,
        confidence: float = 0.85,
    ) -> Dict[str, Any]:
        """Executes an immediate manual live/simulated BUY order from UI."""
        if price <= 0 or shares <= 0:
            return {"success": False, "error": "Invalid price or shares count"}

        cost = float(shares * price)
        if cost > self.state["cash"]:
            return {
                "success": False,
                "error": f"Insufficient cash (${self.state['cash']:,.2f} available, required ${cost:,.2f})",
            }

        now_utc = datetime.now(timezone.utc)
        date_str = now_utc.strftime("%Y-%m-%d")
        now_str = now_utc.isoformat()

        atr_base = atr if atr and atr > 0 else max(price * 0.025, price * 0.03)
        tp1_target = round(price + (2.5 * atr_base), 2)
        tp2_target = round(price + (4.5 * atr_base), 2)
        sl_target = round(price - (1.5 * atr_base), 2)

        self.state["cash"] -= cost
        self.state["open_positions"][ticker] = {
            "ticker": ticker,
            "shares": shares,
            "initial_shares": shares,
            "entry_price": price,
            "current_price": price,
            "entry_date": date_str,
            "tp1_target": tp1_target,
            "tp2_target": tp2_target,
            "sl_target": sl_target,
            "scaled_out": False,
            "confidence": confidence,
            "regime": "MANUAL_LIVE_ORDER",
        }

        self._recalculate_metrics(date_str, now_str)
        self._save()
        logger.info(
            f"⚡ [MANUAL LIVE BUY] Executed {shares} shares of {ticker} @ ${price:.2f} (Total: ${cost:,.2f})"
        )
        return {
            "success": True,
            "ticker": ticker,
            "shares": shares,
            "price": price,
            "cost": cost,
            "tp1_target": tp1_target,
            "tp2_target": tp2_target,
            "sl_target": sl_target,
        }

    def _save_state(self):
        """Alias for _save to ensure 100% backward compatibility."""
        return self._save()

    def execute_manual_sell(
        self,
        ticker: str,
        price: Optional[float] = None,
        reason: str = "MANUAL_MARKET_EXIT",
    ) -> Dict[str, Any]:
        """Executes an immediate manual live/simulated exit of an open position."""
        if ticker not in self.state["open_positions"]:
            return {"success": False, "error": f"No open position for {ticker}"}

        pos = self.state["open_positions"][ticker]
        exit_price = float(
            price
            if price and price > 0
            else pos.get("current_price", pos["entry_price"])
        )
        shares = int(pos["shares"])
        entry_price = float(pos["entry_price"])

        proceeds = float(shares * exit_price)
        cost_basis = float(shares * entry_price)
        pnl = float(proceeds - cost_basis)
        ret_pct = (
            float((exit_price - entry_price) / entry_price * 100.0)
            if entry_price > 0
            else 0.0
        )

        self.state["cash"] += proceeds
        self.state["realized_pnl"] += pnl
        self.state["total_trades"] += 1
        if pnl > 0:
            self.state["winning_trades"] += 1
        else:
            self.state["losing_trades"] += 1

        now_utc = datetime.now(timezone.utc)
        date_str = now_utc.strftime("%Y-%m-%d")
        now_str = now_utc.isoformat()

        trade_record = {
            "ticker": ticker,
            "shares": shares,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "entry_date": pos["entry_date"],
            "exit_date": date_str,
            "pnl": round(pnl, 2),
            "return_pct": round(ret_pct, 2),
            "reason": reason,
        }
        self.state["closed_trades"].append(trade_record)
        del self.state["open_positions"][ticker]

        self._recalculate_metrics(date_str, now_str)
        self._save()
        logger.info(
            f"🛑 [MANUAL LIVE EXIT] Closed {ticker} @ ${exit_price:.2f} | PnL: ${pnl:+,.2f} ({ret_pct:+.2f}%)"
        )
        return {"success": True, "trade": trade_record}

    def execute_manual_scale_out(
        self,
        ticker: str,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Executes a 50% scale-out on an open position and moves stop to break-even."""
        if ticker not in self.state["open_positions"]:
            return {"success": False, "error": f"No open position for {ticker}"}

        pos = self.state["open_positions"][ticker]
        if pos.get("scaled_out", False):
            return {"success": False, "error": f"{ticker} has already been scaled out"}

        curr_price = float(
            price
            if price and price > 0
            else pos.get("current_price", pos["entry_price"])
        )
        shares = int(pos["shares"])
        entry_price = float(pos["entry_price"])
        half_shares = max(1, shares // 2)

        proceeds = float(half_shares * curr_price)
        cost_basis = float(half_shares * entry_price)
        pnl = float(proceeds - cost_basis)
        ret_pct = (
            float((curr_price - entry_price) / entry_price * 100.0)
            if entry_price > 0
            else 0.0
        )

        self.state["cash"] += proceeds
        self.state["realized_pnl"] += pnl
        pos["shares"] = shares - half_shares
        pos["scaled_out"] = True
        pos["sl_target"] = round(entry_price * 1.002, 2)

        now_utc = datetime.now(timezone.utc)
        date_str = now_utc.strftime("%Y-%m-%d")
        now_str = now_utc.isoformat()

        trade_record = {
            "ticker": ticker,
            "shares": half_shares,
            "entry_price": entry_price,
            "exit_price": curr_price,
            "entry_date": pos["entry_date"],
            "exit_date": date_str,
            "pnl": round(pnl, 2),
            "return_pct": round(ret_pct, 2),
            "reason": "MANUAL_50PCT_SCALE_OUT",
            "scale_stage": "STAGE_1_50PCT",
            "status": "RISK_FREE_RUNNER",
        }
        self.state["closed_trades"].append(trade_record)

        self._recalculate_metrics(date_str, now_str)
        self._save()
        logger.info(
            f"🎯 [MANUAL 50% SCALE-OUT] Banked 50% of {ticker} @ ${curr_price:.2f} | PnL: ${pnl:+,.2f} | SL Moved to Break-Even"
        )
        return {"success": True, "trade": trade_record}
