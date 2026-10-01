"""
Sentilyze Model Training Monitor.
Tracks how many stocks have trained models in models/ and results/,
checks progress against stocks.txt, and displays recent training timestamps.
"""

import os
import json
from pathlib import Path
from datetime import datetime


def monitor(universe_file: str = "stocks_us_all.txt"):
    models_dir = Path("models")
    results_dir = Path("results")
    stocks_file = Path(universe_file)
    if not stocks_file.exists():
        stocks_file = Path("stocks.txt")

    # 1. Discover trained models
    trained_models = list(models_dir.glob("*_model.json"))
    trained_tickers = set(f.name.replace("_model.json", "") for f in trained_models)

    # 2. Discover results metrics
    metrics_files = list(results_dir.glob("*_metrics.json"))
    results_tickers = set(f.name.replace("_metrics.json", "") for f in metrics_files)

    # 3. Read target universe from file
    target_tickers = []
    if stocks_file.exists():
        with open(stocks_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    target_tickers.append(s)

    universe_total = len(target_tickers)
    universe_trained = [t for t in target_tickers if t in trained_tickers]
    universe_pending = [t for t in target_tickers if t not in trained_tickers]

    print("==================================================================")
    print("           SENTILYZE MODEL TRAINING MONITOR                       ")
    print("==================================================================")
    print(f"Total Trained Models in models/ : {len(trained_models):,} stocks")
    print(f"Total Metric Files in results/ : {len(metrics_files):,} stocks")
    print("------------------------------------------------------------------")

    # Retraining Status File Check
    status_file = results_dir / "retrain_status.json"
    if status_file.exists():
        try:
            st = json.loads(status_file.read_text(encoding="utf-8"))
            status_desc = "COMPLETE" if st.get("is_complete") else "IN PROGRESS"
            print(f"Batch Retrain Status           : {status_desc}")
            print(
                f"  * Batch Progress             : {st.get('completed', 0)} / {st.get('total', universe_total)} ({st.get('pct', 0.0)}%)"
            )
            print(
                f"  * Success / Failed           : {st.get('succeeded', 0)} / {st.get('failed', 0)}"
            )
            print(
                f"  * Speed / ETA                : {st.get('speed_stocks_per_sec', 0.0)} st/s | ETA: {st.get('eta_min', 0.0)} min"
            )
            print(f"  * Latest Stock Processed     : {st.get('latest_ticker', 'N/A')}")
            print("------------------------------------------------------------------")
        except Exception:
            pass

    # Recent training activity (sorted by model file modification time)
    if trained_models:
        recent_models = sorted(
            trained_models, key=lambda f: f.stat().st_mtime, reverse=True
        )[:10]
        print("Most Recently Trained Stock Models:")
        print("  Ticker     Last Modified         Accuracy   Sharpe   Return")
        print("  " + "-" * 55)
        for m in recent_models:
            tk = m.name.replace("_model.json", "")
            mtime = datetime.fromtimestamp(m.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            acc_str = "-"
            sharpe_str = "-"
            ret_str = "-"
            mf = results_dir / f"{tk}_metrics.json"
            if mf.exists():
                try:
                    data = json.loads(mf.read_text(encoding="utf-8"))
                    if "accuracy" in data:
                        acc_str = f"{data['accuracy']:.1%}"
                    if "strategy_sharpe_ratio" in data:
                        sharpe_str = f"{data['strategy_sharpe_ratio']:.2f}"
                    elif "sharpe_ratio" in data:
                        sharpe_str = f"{data['sharpe_ratio']:.2f}"
                    if "strategy_total_return" in data:
                        ret_str = f"{data['strategy_total_return']:.1%}"
                except Exception:
                    pass
            print(f"  {tk:<9}  {mtime}   {acc_str:>8}  {sharpe_str:>7}  {ret_str:>7}")

    print("==================================================================")


if __name__ == "__main__":
    import sys

    u_file = sys.argv[1] if len(sys.argv) > 1 else "stocks_us_all.txt"
    monitor(u_file)
