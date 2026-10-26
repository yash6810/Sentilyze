"""
Real 3-Ticker Walk-Forward Backtest (NVDA, SPY, WMT) with Zero Look-Ahead Bias.
- Walk-forward splits (3 non-overlapping out-of-sample test windows)
- Label at t strictly uses return at t+1
- Feature engineering strictly on data <= t (with fold-isolated StandardScaler)
- Execution at t+1 Open price
- Realistic transaction fees: 0.10% (10 bps)
- Realistic slippage: 0.05% (5 bps)
- Honest sentiment: 0.0 (zero if missing)
- Outputs results to audit_outputs/backtest_3ticker.csv
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import datetime
import duckdb
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_price_data(ticker: str) -> pd.DataFrame:
    # Try local DuckDB first
    try:
        con = duckdb.connect("data/sentilyze_market.duckdb", read_only=True)
        query = f"SELECT date, open, high, low, close, volume FROM daily_bars WHERE ticker = '{ticker}' AND date >= '2019-01-01' ORDER BY date"
        df = con.execute(query).df()
        if len(df) >= 300:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.columns = [c.capitalize() for c in df.columns]
            return df
    except Exception:
        pass

    # Direct Yahoo chart fallback
    from src.data_ingestion import _fetch_direct_yahoo_chart

    df = _fetch_direct_yahoo_chart(ticker, period="10y")
    if df is not None and len(df) >= 300:
        return df

    import yfinance as yf

    raw = yf.download(ticker, start="2019-01-01", auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    close = data["Close"]
    high = data["High"]
    low = data["Low"]

    # Technical indicators on close <= t
    data["ret_1d"] = close.pct_change(1)
    data["ret_5d"] = close.pct_change(5)
    data["ret_10d"] = close.pct_change(10)
    data["ret_20d"] = close.pct_change(20)

    data["sma_20"] = close.rolling(20).mean() / close - 1.0
    data["sma_50"] = close.rolling(50).mean() / close - 1.0
    data["sma_200"] = close.rolling(200).mean() / close - 1.0

    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    data["rsi_14"] = 100 - (100 / (1 + rs))

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data["atr_14"] = tr.rolling(14).mean()
    data["atr_pct"] = data["atr_14"] / close

    data["vol_20"] = data["ret_1d"].rolling(20).std()

    # Honest sentiment: 0.0 (no historical news fabricated)
    data["sentiment_score"] = 0.0

    # Target: 1 if next day's close > today's close, 0 otherwise
    data["target"] = (close.shift(-1) > close).astype(int)

    data.dropna(inplace=True)
    return data


def run_wfo_backtest(ticker: str):
    print(f"\n=======================================================")
    print(f"Running Honest Walk-Forward Backtest for {ticker}")
    print(f"=======================================================")
    raw_df = load_price_data(ticker)
    data = calculate_features(raw_df)

    feature_cols = [
        "ret_1d",
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "sma_20",
        "sma_50",
        "sma_200",
        "rsi_14",
        "atr_pct",
        "vol_20",
        "sentiment_score",
    ]

    total_samples = len(data)
    # Define 3 non-overlapping test windows (each ~20% of dataset)
    test_size = int(total_samples * 0.20)
    # Train windows expand to cover all prior data
    splits = [
        (int(total_samples * 0.40), int(total_samples * 0.60)),
        (int(total_samples * 0.60), int(total_samples * 0.80)),
        (int(total_samples * 0.80), total_samples),
    ]

    all_oos_dates = []
    all_oos_probs = []

    for fold_idx, (train_end, test_end) in enumerate(splits, 1):
        train_df = data.iloc[:train_end]
        test_df = data.iloc[train_end:test_end]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feature_cols])
        y_train = train_df["target"].values

        X_test = scaler.transform(test_df[feature_cols])

        clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        )
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]

        all_oos_dates.extend(test_df.index)
        all_oos_probs.extend(probs)
        print(
            f"Fold {fold_idx}: Train {len(train_df)} days -> Test {len(test_df)} days | Mean Prob Up: {probs.mean():.3f}"
        )

    oos_series = pd.Series(all_oos_probs, index=all_oos_dates)
    test_price_df = raw_df.loc[oos_series.index].copy()

    # Simulation parameters
    initial_capital = 100000.0
    cash = initial_capital
    position = 0  # shares
    entry_price = 0.0
    trades_pnl = []
    equity_curve = []
    trade_count = 0
    wins = 0

    fee_rate = 0.0010  # 0.10% transaction fee
    slippage = 0.0005  # 0.05% slippage

    position_open = False
    stop_loss = 0.0
    take_profit = 0.0

    dates = test_price_df.index
    for i in range(len(dates) - 1):
        today_date = dates[i]
        next_date = dates[i + 1]

        prob = oos_series.loc[today_date]
        today_close = test_price_df.loc[today_date, "Close"]
        next_open = test_price_df.loc[next_date, "Open"]
        next_close = test_price_df.loc[next_date, "Close"]

        # 1. Manage existing position
        if position_open:
            # Check exit conditions at next day's open
            if next_open <= stop_loss or next_open >= take_profit:
                exit_fill = next_open * (1.0 - slippage)
                revenue = position * exit_fill
                fee = revenue * fee_rate
                net_revenue = revenue - fee
                pnl = net_revenue - (position * entry_price)
                cash += net_revenue
                trades_pnl.append(pnl)
                trade_count += 1
                if pnl > 0:
                    wins += 1
                position = 0
                position_open = False

        # 2. Enter new position if flat and probability >= 0.52
        if not position_open and prob >= 0.52 and cash > 1000.0:
            entry_fill = next_open * (1.0 + slippage)
            invest_amount = cash * 0.95
            shares = int(invest_amount / entry_fill)
            if shares > 0:
                cost = shares * entry_fill
                fee = cost * fee_rate
                cash -= cost + fee
                position = shares
                entry_price = entry_fill
                position_open = True
                # Set stop-loss (3%) and take-profit (6%)
                stop_loss = entry_price * 0.97
                take_profit = entry_price * 1.06

        # Mark-to-market total equity at next close
        current_eq = cash + (position * next_close)
        equity_curve.append({"date": next_date, "equity": current_eq})

    # Close any open position at the end
    if position_open:
        final_close = test_price_df.iloc[-1]["Close"] * (1.0 - slippage)
        net_revenue = position * final_close * (1.0 - fee_rate)
        pnl = net_revenue - (position * entry_price)
        cash += net_revenue
        trades_pnl.append(pnl)
        trade_count += 1
        if pnl > 0:
            wins += 1

    eq_df = pd.DataFrame(equity_curve)
    strat_final_equity = eq_df["equity"].iloc[-1]
    strat_total_return = (strat_final_equity / initial_capital) - 1.0

    eq_series = eq_df["equity"]
    strat_dd = (eq_series / eq_series.cummax()) - 1.0
    strat_mdd = float(strat_dd.min())
    strat_pct = eq_series.pct_change().dropna()
    strat_sharpe = float((strat_pct.mean() / (strat_pct.std() + 1e-10)) * np.sqrt(252))

    # Buy and Hold benchmark
    bnh_start_price = test_price_df.iloc[0]["Open"]
    bnh_end_price = test_price_df.iloc[-1]["Close"]
    bnh_return = (bnh_end_price / bnh_start_price) - 1.0
    bnh_series = test_price_df["Close"]
    bnh_dd = (bnh_series / bnh_series.cummax()) - 1.0
    bnh_mdd = float(bnh_dd.min())
    bnh_pct = bnh_series.pct_change().dropna()
    bnh_sharpe = float((bnh_pct.mean() / (bnh_pct.std() + 1e-10)) * np.sqrt(252))

    win_rate = (wins / trade_count) if trade_count > 0 else 0.0

    res = {
        "ticker": ticker,
        "eval_days": len(eq_df),
        "trade_count": trade_count,
        "win_rate": round(win_rate * 100, 2),
        "strategy_total_return_pct": round(strat_total_return * 100, 2),
        "bnh_total_return_pct": round(bnh_return * 100, 2),
        "strategy_sharpe": round(strat_sharpe, 2),
        "bnh_sharpe": round(bnh_sharpe, 2),
        "strategy_max_dd_pct": round(strat_mdd * 100, 2),
        "bnh_max_dd_pct": round(bnh_mdd * 100, 2),
        "strategy_beats_bnh": bool(strat_total_return > bnh_return),
    }

    print(f"Results for {ticker}:")
    print(f"  Trades: {trade_count} | Win Rate: {res['win_rate']}%")
    print(
        f"  Strategy Return: {res['strategy_total_return_pct']}% vs B&H: {res['bnh_total_return_pct']}%"
    )
    print(f"  Strategy Sharpe: {res['strategy_sharpe']} vs B&H: {res['bnh_sharpe']}")
    print(
        f"  Strategy Max DD: {res['strategy_max_dd_pct']}% vs B&H: {res['bnh_max_dd_pct']}%"
    )
    print(f"  Beats Buy-and-Hold: {res['strategy_beats_bnh']}")
    return res


def main():
    tickers = ["NVDA", "SPY", "WMT"]
    results = []
    for t in tickers:
        r = run_wfo_backtest(t)
        results.append(r)

    out_df = pd.DataFrame(results)
    os.makedirs("audit_outputs", exist_ok=True)
    out_df.to_csv("audit_outputs/backtest_3ticker.csv", index=False)
    print("\nSaved 3-ticker backtest to audit_outputs/backtest_3ticker.csv")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
