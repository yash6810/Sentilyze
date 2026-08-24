import os
import pandas as pd
from datetime import datetime, timezone
from src.utils import get_logger
from src.data_ingestion import get_price_history, get_news, get_vix_data

logger = get_logger(__name__)


def sync_all_market_data(period: str = "10y") -> dict:
    """
    Nightly sync script that pre-fetches and caches 10-year OHLCV prices,
    VIX macro levels, and breaking news headlines for all universe assets.
    """
    stocks_file = "stocks.txt"
    if os.path.exists(stocks_file):
        with open(stocks_file, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]
    else:
        tickers = ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "TSLA", "AMZN"]

    logger.info(
        f"Starting nightly data lake sync for {len(tickers)} assets (Period: {period})..."
    )
    api_key = os.environ.get("NEWS_API_KEY")

    # 1. Sync Macro VIX Index
    logger.info("Syncing VIX macro index...")
    vix_df = get_vix_data(period=period, cache_duration_hours=0)
    logger.info(f"✓ VIX synced successfully: {len(vix_df)} trading bars.")

    sync_status = {}

    # 2. Sync Individual Tickers
    for ticker in tickers:
        try:
            logger.info(f"Syncing data for {ticker}...")
            news_df = get_news(ticker, api_key=api_key, cache_duration_hours=0)
            price_df = get_price_history(ticker, period=period, cache_duration_hours=0)

            sync_status[ticker] = {
                "price_bars": len(price_df),
                "news_articles": len(news_df),
                "latest_price_date": str(
                    price_df.index.max() if not price_df.empty else "N/A"
                ),
                "status": "SUCCESS",
            }
            logger.info(
                f"✓ {ticker:<6} synced: {len(price_df)} price bars (Latest: {price_df.index.max()}), "
                f"{len(news_df)} news articles."
            )
        except Exception as e:
            logger.error(f"Failed to sync {ticker}: {e}")
            sync_status[ticker] = {"status": "FAILED", "error": str(e)}

    summary = {
        "sync_time": datetime.now(timezone.utc).isoformat(),
        "assets_synced": len(tickers),
        "details": sync_status,
    }
    logger.info("Nightly data lake sync complete.")
    return summary


if __name__ == "__main__":
    sync_all_market_data()
