"""
Batch Universe Trainer for Remaining S&P 100 Tickers.
"""

import os
import sys
import concurrent.futures
from typing import List
import train
from src.utils import get_logger

logger = get_logger("batch_trainer")


def run_single(ticker: str) -> bool:
    try:
        logger.info(f"🚀 Starting WFO Training for {ticker}...")
        train.main(ticker, leverage=1.5, use_cache=True)
        logger.info(f"✅ Successfully trained and backtested {ticker}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed training for {ticker}: {e}")
        return False


def main():
    with open("stocks.txt") as f:
        all_tickers = [
            l.strip() for l in f if l.strip() and not l.startswith("#")
        ]

    existing = set(
        f.replace("_portfolio.csv", "")
        for f in os.listdir("results")
        if f.endswith("_portfolio.csv")
    )
    remaining = [
        t for t in all_tickers if t not in existing and t != "OPENAI"
    ]

    print(
        f"Starting parallel training for {len(remaining)} remaining tickers..."
    )
    max_workers = 4

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:
        futures = {executor.submit(run_single, t): t for t in remaining}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            t = futures[future]
            completed += 1
            try:
                success = future.result()
                status = "SUCCESS" if success else "FAILED"
                print(f"[{completed}/{len(remaining)}] {t}: {status}")
            except Exception as exc:
                print(f"[{completed}/{len(remaining)}] {t}: EXCEPTION {exc}")


if __name__ == "__main__":
    main()
