"""
Universal Parallel Multi-Core Model Training & High-Resolution Benchmark.

Functions:
- Parallel multi-process model training across the complete S&P 500 universe (538 assets).
- Automatic load balancing across CPU worker cores using concurrent.futures.ProcessPoolExecutor.
- Zero-cache mode option for fresh data or optimized pipeline throughput.
- Measures high-resolution phase latencies with microsecond precision via time.perf_counter().
"""

import os
import sys
import time
import argparse
from datetime import datetime, timezone
import concurrent.futures
from typing import List, Dict, Any
import train
from src.utils import get_logger

logger = get_logger("universe_trainer")


def load_universe_from_file(file_path: str = "stocks.txt") -> List[str]:
    """Loads cleaned ticker symbols from stocks.txt."""
    if not os.path.exists(file_path):
        return ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA"]
    with open(file_path, "r") as f:
        tickers = [
            line.strip().upper()
            for line in f
            if line.strip()
            and not line.startswith("#")
            and line.strip().upper() != "OPENAI"
        ]
    # Remove duplicates while preserving order
    seen = set()
    unique_tickers = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique_tickers.append(t)
    return unique_tickers


def train_single_asset(args_tuple) -> Dict[str, Any]:
    """Worker function to train a single asset."""
    ticker, leverage, use_cache = args_tuple
    t_start = time.perf_counter()
    try:
        train.main(ticker=ticker, leverage=leverage, use_cache=use_cache)
        t_elapsed = time.perf_counter() - t_start
        return {
            "ticker": ticker,
            "status": "SUCCESS",
            "duration_sec": round(t_elapsed, 3),
        }
    except Exception as e:
        t_elapsed = time.perf_counter() - t_start
        return {
            "ticker": ticker,
            "status": "FAILED",
            "duration_sec": round(t_elapsed, 3),
            "error": str(e),
        }


def prefetch_single_ticker(ticker: str) -> bool:
    """Pre-fetches price history and news data concurrently across fast I/O threads."""
    try:
        from src.data_ingestion import get_price_history, get_news

        get_price_history(ticker, period="5y", use_cache=True)
        get_news(ticker, use_cache=True)
        return True
    except Exception:
        return False


def prefetch_universe_data(tickers: List[str], max_workers: int = 16):
    """Pre-fetches market and news data in parallel across fast I/O threads."""
    print(
        f"\n🌐 Fast-Ingestion Pre-fetch: Concurrently pulling data for {len(tickers)} tickers ({max_workers} I/O threads)...",
        flush=True,
    )
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(prefetch_single_ticker, tickers))
    print(f"✅ Pre-fetch complete in {time.perf_counter() - t0:.2f}s!\n", flush=True)


def run_parallel_universe_training(
    tickers: List[str],
    max_workers: int = 8,
    leverage: float = 1.5,
    use_cache: bool = False,
    prefetch: bool = True,
) -> Dict[str, Any]:
    """
    Executes parallel multi-core model training across universe with real-time ETA tracking.
    """
    if prefetch:
        prefetch_universe_data(tickers, max_workers=16)

    overall_start = time.perf_counter()
    start_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print("\n" + "=" * 75)
    print("🚀 STARTING UNIVERSAL PARALLEL MODEL TRAINING (S&P 500 UNIVERSE)")
    print(f"🕒 Start Timestamp: {start_timestamp}")
    print(f"📊 Target Universe: {len(tickers)} Equities")
    print(
        f"⚡ Parallel Workers: {max_workers} CPU Cores | Cache Mode: {'ENABLED' if use_cache else 'ZERO-CACHE LIVE'}"
    )
    print(f"⚙️ Backtest Leverage: {leverage}x")
    print("=" * 75 + "\n", flush=True)

    results = []
    completed_count = 0
    total_count = len(tickers)

    task_args = [(t, leverage, use_cache) for t in tickers]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(train_single_asset, arg): arg[0] for arg in task_args
        }

        for future in concurrent.futures.as_completed(future_to_ticker):
            completed_count += 1
            res = future.result()
            results.append(res)

            elapsed_so_far = time.perf_counter() - overall_start
            avg_per_item = elapsed_so_far / completed_count
            remaining_items = total_count - completed_count
            est_remaining_sec = avg_per_item * remaining_items
            est_remaining_min = est_remaining_sec / 60.0

            pct_done = (completed_count / total_count) * 100.0
            status_icon = "✅" if res["status"] == "SUCCESS" else "❌"

            print(
                f"[{completed_count:03d}/{total_count:03d}] {pct_done:5.1f}% | "
                f"{status_icon} {res['ticker']:<5} ({res['duration_sec']:5.1f}s) | "
                f"Elapsed: {elapsed_so_far/60:4.1f}m | ETA: ~{est_remaining_min:4.1f}m",
                flush=True,
            )

    total_elapsed = time.perf_counter() - overall_start
    end_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    successes = sum(1 for r in results if r["status"] == "SUCCESS")
    failures = total_count - successes

    print("\n" + "=" * 75)
    print("🏁 FULL S&P 500 UNIVERSE TRAINING COMPLETED")
    print(f"🕒 Finish Timestamp: {end_timestamp}")
    print(
        f"⏱️ Total Wall-Clock Execution Time: {total_elapsed/60:.2f} minutes ({total_elapsed:.1f}s)"
    )
    print(
        f"📊 Completed: {successes}/{total_count} Models Successfully Optimized & Saved ({failures} failed)"
    )
    if results:
        avg_speed = sum(r["duration_sec"] for r in results) / len(results)
        print(
            f"⚡ Individual Asset Throughput: {avg_speed:.2f}s per model across {max_workers} cores"
        )
    print("=" * 75 + "\n", flush=True)

    return {
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "total_elapsed_seconds": round(total_elapsed, 2),
        "success_count": successes,
        "failure_count": failures,
    }


def main():
    parser = argparse.ArgumentParser(description="Parallel Universe Model Training")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train the entire 538-stock universe from stocks.txt",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Specific list of tickers to train",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel CPU worker processes (default: 8)",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=1.5,
        help="Maximum backtest leverage",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip tickers that already have a trained model in models/",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining of all tickers even if already trained",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Enable cache reuse for maximum throughput",
    )
    parser.add_argument(
        "--no-prefetch",
        action="store_true",
        help="Disable concurrent fast I/O data prefetching",
    )
    args = parser.parse_args()

    if args.all or not args.tickers:
        tickers = load_universe_from_file("stocks.txt")
    else:
        tickers = args.tickers

    if args.resume and not args.force:
        existing_models = set(
            f.replace("_model.json", "")
            for f in os.listdir("models")
            if f.endswith("_model.json")
        )
        remaining_tickers = [t for t in tickers if t not in existing_models]
        print(
            f"⚡ Resume Mode Active: Found {len(existing_models)} existing models. Training remaining {len(remaining_tickers)} tickers..."
        )
        tickers = remaining_tickers

    if not tickers:
        print("✅ All universe tickers are already trained and up to date!")
        return

    run_parallel_universe_training(
        tickers=tickers,
        max_workers=args.workers,
        leverage=args.leverage,
        use_cache=args.use_cache,
        prefetch=not args.no_prefetch,
    )


if __name__ == "__main__":
    main()
