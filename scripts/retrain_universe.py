"""
Full Universe Retraining Engine for Sentilyze.
Retrains all stocks in stocks.txt in parallel using cached market data,
applying all 44 newly engineered quantitative features and Walk-Forward Optimization.
"""

import os
import sys
import time
import json
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from concurrent.futures import ThreadPoolExecutor, as_completed
from train import main as train_single_stock


def retrain_universe(
    stocks_file_path: str = "stocks_us_all.txt",
    max_workers: int = 6,
    leverage: float = 1.5,
):
    stocks_file = Path(stocks_file_path)
    if not stocks_file.exists():
        fallback = Path("stocks.txt")
        if fallback.exists():
            print(
                f"Warning: {stocks_file_path} not found. Falling back to stocks.txt",
                flush=True,
            )
            stocks_file = fallback
        else:
            print(f"Error: {stocks_file_path} and stocks.txt not found.", flush=True)
            return

    tickers = []
    with open(stocks_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                tickers.append(s)

    total = len(tickers)
    print(f"==================================================", flush=True)
    print(f"SENTILYZE FULL UNIVERSE RETRAINING ENGINE", flush=True)
    print(f"Universe Source:  {stocks_file.name} ({total} stocks)", flush=True)
    print(f"Parallel Workers: {max_workers} CPU threads", flush=True)
    print(f"Leverage Cap:     {leverage}x", flush=True)
    print(f"==================================================", flush=True)

    t0 = time.time()
    completed = 0
    succeeded = 0
    failed = 0
    failed_tickers = []

    def run_worker(ticker: str):
        try:
            train_single_stock(ticker, leverage=leverage, use_cache=True)
            return ticker, True, None
        except Exception as e:
            return ticker, False, str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_worker, t): t for t in tickers}
        for future in as_completed(futures):
            ticker, success, err = future.result()
            completed += 1
            if success:
                succeeded += 1
            else:
                failed += 1
                failed_tickers.append((ticker, err))

            elapsed = time.time() - t0
            rate = completed / max(1.0, elapsed)
            remaining_sec = (total - completed) / max(0.01, rate)
            pct = (completed / total) * 100.0

            # Write live status to results/retrain_status.json
            try:
                status_data = {
                    "completed": completed,
                    "total": total,
                    "pct": round(pct, 2),
                    "succeeded": succeeded,
                    "failed": failed,
                    "latest_ticker": ticker,
                    "elapsed_sec": round(elapsed, 1),
                    "eta_min": round(remaining_sec / 60.0, 1),
                    "speed_stocks_per_sec": round(rate, 2),
                    "is_complete": completed == total,
                }
                with open("results/retrain_status.json", "w", encoding="utf-8") as sf:
                    json.dump(status_data, sf, indent=2)
            except Exception:
                pass

            if completed % 10 == 0 or completed == total:
                print(
                    f"[{completed:3d}/{total}] ({pct:5.1f}%) | "
                    f"Success: {succeeded} | Failed: {failed} | "
                    f"Speed: {rate:.1f} stocks/s | ETA: {remaining_sec / 60:.1f}m | "
                    f"Latest: {ticker}",
                    flush=True,
                )

    print("==================================================", flush=True)
    print("UNIVERSE RETRAINING COMPLETE:", flush=True)
    print(f"Total Processed: {completed} / {total}", flush=True)
    print(f"Succeeded:       {succeeded}", flush=True)
    print(f"Failed:          {failed}", flush=True)
    print(f"Total Time:      {(time.time() - t0) / 60:.2f} minutes", flush=True)
    if failed_tickers:
        print(f"Failed Tickers: {[t[0] for t in failed_tickers[:10]]}", flush=True)
    print("==================================================", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Full Universe Retraining Engine for Sentilyze"
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=6,
        help="Number of parallel worker threads (default: 6)",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="stocks_us_all.txt",
        help="Stocks universe text file (default: stocks_us_all.txt)",
    )
    parser.add_argument(
        "--leverage",
        "-l",
        type=float,
        default=1.5,
        help="Leverage cap for backtest (default: 1.5)",
    )
    parser.add_argument(
        "pos_workers",
        nargs="?",
        type=int,
        default=None,
        help="Optional positional workers count",
    )
    parser.add_argument(
        "pos_file",
        nargs="?",
        type=str,
        default=None,
        help="Optional positional universe file",
    )

    args = parser.parse_args()

    workers = args.pos_workers if args.pos_workers is not None else args.workers
    stocks_file = args.pos_file if args.pos_file is not None else args.file

    retrain_universe(
        stocks_file_path=stocks_file, max_workers=workers, leverage=args.leverage
    )
