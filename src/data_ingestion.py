import os
import requests
import pandas as pd
from newsapi import NewsApiClient
import yfinance as yf
from src.utils import get_logger
import time
from typing import Dict

logger = get_logger(__name__)

# Get the absolute path to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# Global variable to track the last yfinance API call time
last_yfinance_call_time = 0
YFINANCE_CALL_INTERVAL = 30  # seconds


def _fetch_yfinance_news(ticker: str) -> pd.DataFrame:
    """Fetches real-time live market news headlines directly from Yahoo Finance."""
    try:
        session = _get_browser_session()
        t = yf.Ticker(ticker, session=session)
        raw_news = t.news or []
        articles = []
        for item in raw_news:
            title = item.get("title") or item.get("content", {}).get("title")
            if not title:
                continue
            pub_time = item.get("providerPublishTime") or item.get("content", {}).get(
                "pubDate"
            )
            if isinstance(pub_time, (int, float)):
                published_at = pd.to_datetime(pub_time, unit="s", utc=True)
            elif pub_time:
                published_at = pd.to_datetime(pub_time, utc=True)
            else:
                published_at = pd.to_datetime("now", utc=True)

            publisher = item.get("publisher") or item.get("content", {}).get(
                "provider", {}
            ).get("displayName", "Yahoo Finance")
            link = (
                item.get("link")
                or item.get("content", {}).get("canonicalUrl", {}).get("url", "")
            )
            summary = item.get("summary") or item.get("content", {}).get(
                "summary", ""
            )

            articles.append(
                {
                    "publishedAt": published_at,
                    "Title": title,
                    "description": summary,
                    "url": link,
                    "source": {"name": publisher},
                }
            )

        if articles:
            df = pd.DataFrame(articles)
            logger.info(
                f"Successfully fetched {len(df)} real-time live news articles for {ticker} from Yahoo Finance"
            )
            return df
    except Exception as e:
        logger.warning(f"Live yfinance news fetch failed for {ticker}: {e}")
    return pd.DataFrame()


def get_news(
    ticker: str,
    api_key: str = None,
    cache_duration_hours: int = 24,
    retries: int = 3,
    backoff_factor: float = 1,
) -> pd.DataFrame:
    """
    Fetch recent news for a given ticker from live sources (Yahoo Finance + NewsAPI), with caching.

    Args:
        ticker (str): The stock ticker to fetch news for.
        api_key (str, optional): The API key for NewsAPI.
        cache_duration_hours (int, optional): Cache freshness duration. Defaults to 24.

    Returns:
        pd.DataFrame: A DataFrame containing recent news articles, indexed by 'publishedAt'.
    """
    cache_path = os.path.join(DATA_DIR, f"{ticker}_news.csv")
    os.makedirs(DATA_DIR, exist_ok=True)

    use_cache = False
    if os.path.exists(cache_path):
        cache_age_seconds = time.time() - os.path.getmtime(cache_path)
        if cache_age_seconds < cache_duration_hours * 3600:
            use_cache = True
        else:
            logger.info(f"News cache for {ticker} is stale. Re-fetching fresh live news...")

    if use_cache:
        logger.info(f"Loading news for {ticker} from cache...")
        articles_df = pd.read_csv(cache_path)
    else:
        logger.info(f"Fetching fresh live news for {ticker}...")
        # 1. Primary: Try fetching up-to-the-minute live news from Yahoo Finance
        articles_df = _fetch_yfinance_news(ticker)

        # 2. Fallback / Augment with NewsAPI if available and yfinance was empty
        if articles_df.empty and api_key and isinstance(api_key, str):
            try:
                newsapi = NewsApiClient(api_key=api_key)
                all_articles = newsapi.get_everything(
                    q=ticker, language="en", sort_by="publishedAt", page_size=100
                )
                if all_articles.get("articles"):
                    articles_df = pd.DataFrame(all_articles["articles"])
                    articles_df.rename(columns={"title": "Title"}, inplace=True)
            except Exception as e:
                logger.warning(f"NewsAPI query failed for {ticker}: {e}")

        # 3. If still empty, fallback to cached data or dummy data
        if articles_df.empty:
            if os.path.exists(cache_path):
                logger.warning(f"Using existing cached news for {ticker} as fallback.")
                articles_df = pd.read_csv(cache_path)
            else:
                articles_df = _generate_dummy_news(ticker)

        articles_df.to_csv(cache_path, index=False)
        logger.info(f"Saved fresh news to {cache_path}")

    # Standardize the DataFrame to have a timezone-aware DatetimeIndex
    if "publishedAt" in articles_df.columns:
        articles_df["publishedAt"] = pd.to_datetime(articles_df["publishedAt"], utc=True)
        articles_df = articles_df.set_index("publishedAt").sort_index(ascending=False)

    return articles_df


def _generate_dummy_news(ticker: str) -> pd.DataFrame:
    """Generates dummy news data for showcase purposes when the API fails."""
    logger.info(f"Generating dummy news data for {ticker}...")
    import datetime
    import random

    today = datetime.datetime.now(datetime.timezone.utc)
    dummy_articles = []

    # Generate some slightly positive biased news since long-term tech is usually up
    titles = [
        f"{ticker} announces record breaking quarterly earnings",
        f"Analysts upgrade {ticker} following new product launch",
        f"Market reacts positively to {ticker}'s forward guidance",
        f"{ticker} faces supply chain concerns in upcoming quarter",
        f"CEO of {ticker} discusses future AI initiatives",
        f"{ticker} stock surges on buyout rumors",
    ]

    for i in range(15):
        days_ago = random.randint(0, 14)  # Spread over last 14 days
        pub_date = today - datetime.timedelta(days=days_ago)

        dummy_articles.append(
            {
                "publishedAt": pub_date.isoformat(),
                "Title": random.choice(titles),
                "description": f"This is a generated dummy description for {ticker} to demonstrate pipeline capabilities without an active API connection.",
                "url": "https://example.com/dummy-news",
                "source": {"name": "Portfolio Fallback Generator"},
            }
        )

    return pd.DataFrame(dummy_articles)


def _get_browser_session() -> requests.Session:
    """Creates a requests Session with modern desktop browser headers to prevent 429 scraper detection."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
            "Sec-Ch-Ua-Platform": '"Windows"',
        }
    )
    return session


def _fetch_direct_yahoo_chart(ticker: str, period: str = "10y") -> pd.DataFrame:
    """Fetches full historical price data directly from Yahoo Finance Chart API up to today."""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {
            "range": period,
            "interval": "1d",
            "events": "div,splits",
        }
        session = _get_browser_session()
        res = session.get(url, params=params, timeout=12)
        if res.status_code == 200:
            data = res.json()
            result = data["chart"]["result"][0]
            timestamps = result.get("timestamp", [])
            indicators = result.get("indicators", {})
            quotes = indicators.get("quote", [{}])[0]

            if not timestamps or not quotes.get("close"):
                return pd.DataFrame()

            import datetime

            dates = [
                datetime.datetime.fromtimestamp(
                    ts, tz=datetime.timezone.utc
                )
                for ts in timestamps
            ]
            dt_index = pd.DatetimeIndex(dates, name="Date").normalize()
            df = pd.DataFrame(
                {
                    "Open": quotes.get("open", []),
                    "High": quotes.get("high", []),
                    "Low": quotes.get("low", []),
                    "Close": quotes.get("close", []),
                    "Volume": quotes.get("volume", []),
                },
                index=dt_index,
            )
            df["Dividends"] = 0.0
            df["Stock Splits"] = 0.0

            df = df.ffill().dropna()
            logger.info(
                f"Directly fetched {len(df)} price bars for {ticker} up to {df.index[-1].strftime('%Y-%m-%d')}"
            )
            return df
    except Exception as e:
        logger.warning(f"Direct Yahoo chart fetch failed for {ticker}: {e}")
    return pd.DataFrame()


def get_price_history(
    ticker: str,
    period: str = "10y",
    cache_duration_hours: int = 24,
    retries: int = 3,
    backoff_factor: float = 2,
) -> pd.DataFrame:
    """
    Fetches historical price data up to today for a given ticker, with caching.

    Args:
        ticker (str): Stock ticker symbol.
        period (str, optional): Time range (e.g. '10y', '1y'). Defaults to '10y'.
        cache_duration_hours (int, optional): Cache freshness duration. Defaults to 24.

    Returns:
        pd.DataFrame: A DataFrame containing historical OHLCV data.
    """
    cache_path = os.path.join(DATA_DIR, f"{ticker}_price_history.csv")
    os.makedirs(DATA_DIR, exist_ok=True)

    use_cache = False
    if os.path.exists(cache_path):
        cache_age_seconds = time.time() - os.path.getmtime(cache_path)
        if cache_age_seconds < cache_duration_hours * 3600:
            use_cache = True
        else:
            logger.info(f"Price history cache for {ticker} is stale. Re-fetching...")

    if use_cache:
        logger.info(f"Loading price history for {ticker} from cache...")
        history = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        if history.index.tz is None:
            history.index = history.index.tz_localize("UTC").normalize()
        else:
            history.index = history.index.tz_convert("UTC").normalize()
    else:
        logger.info(f"Fetching fresh live price history for {ticker}...")
        # 1. Primary: Direct Yahoo Finance Chart API (immune to yfinance scraper rate limits)
        history = _fetch_direct_yahoo_chart(ticker, period=period)

        # 2. Fallback to yfinance if direct chart was empty
        if history.empty:
            try:
                session = _get_browser_session()
                stock = yf.Ticker(ticker, session=session)
                history = stock.history(period=period)
                if not history.empty:
                    if history.index.tz is None:
                        history.index = history.index.tz_localize("UTC").normalize()
                    else:
                        history.index = history.index.tz_convert("UTC").normalize()
            except Exception as e:
                logger.warning(f"yfinance fallback failed for {ticker}: {e}")

        # 3. Fallback to existing cache if API failed
        if history.empty:
            if os.path.exists(cache_path):
                logger.warning(f"Using existing cached prices for {ticker}.")
                history = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
                if history.index.tz is None:
                    history.index = history.index.tz_localize("UTC").normalize()
                else:
                    history.index = history.index.tz_convert("UTC").normalize()
            else:
                return pd.DataFrame()
        else:
            history.to_csv(cache_path)
            logger.info(f"Saved updated price history to {cache_path}")

    # Ensure required columns exist
    if "Dividends" not in history.columns:
        history["Dividends"] = 0
    if "Stock Splits" not in history.columns:
        history["Stock Splits"] = 0

    return history

    # Ensure 'Dividends' and 'Stock Splits' columns are present
    if "Dividends" not in history.columns:
        history["Dividends"] = 0
    if "Stock Splits" not in history.columns:
        history["Stock Splits"] = 0

    return history


def get_vix_data(
    period: str = "10y",
    cache_duration_hours: int = 24,
    retries: int = 5,
    backoff_factor: float = 10,
) -> pd.DataFrame:
    """
    Fetches historical data for the CBOE Volatility Index (VIX).

    Args:
        period (str): The time period for the data. Defaults to "10y".

    Returns:
        pd.DataFrame: A DataFrame containing VIX price history.
    """
    global last_yfinance_call_time
    current_time = time.time()
    elapsed_time = current_time - last_yfinance_call_time
    if elapsed_time < YFINANCE_CALL_INTERVAL:
        sleep_duration = YFINANCE_CALL_INTERVAL - elapsed_time
        logger.info(
            f"Rate limiting yfinance call for VIX. Sleeping for {sleep_duration:.2f} seconds."
        )
        time.sleep(sleep_duration)

    last_yfinance_call_time = time.time()
    cache_path = os.path.join(DATA_DIR, "vix_history.csv")
    os.makedirs(DATA_DIR, exist_ok=True)

    use_cache = False
    if os.path.exists(cache_path):
        cache_age_seconds = time.time() - os.path.getmtime(cache_path)
        if cache_age_seconds < cache_duration_hours * 3600:
            use_cache = True

    if use_cache:
        logger.info("Loading VIX data from cache...")
        history = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        if history.index.tz is None:
            history.index = history.index.tz_localize("UTC").normalize()
        else:
            history.index = history.index.tz_convert("UTC").normalize()
        return history

    else:
        logger.info("Fetching fresh live VIX data...")
        history = _fetch_direct_yahoo_chart("^VIX", period=period)

        if history.empty:
            try:
                vix = yf.Ticker("^VIX", session=_get_browser_session())
                history = vix.history(period=period)
                if not history.empty:
                    if history.index.tz is None:
                        history.index = history.index.tz_localize("UTC").normalize()
                    else:
                        history.index = history.index.tz_convert("UTC").normalize()
            except Exception as e:
                logger.warning(f"VIX yfinance fallback failed: {e}")

        if history.empty:
            if os.path.exists(cache_path):
                logger.warning("Using cached VIX data.")
                history = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
                if history.index.tz is None:
                    history.index = history.index.tz_localize("UTC").normalize()
                else:
                    history.index = history.index.tz_convert("UTC").normalize()
            else:
                return pd.DataFrame()
        else:
            history.to_csv(cache_path)
            logger.info(f"Saved fresh VIX history to {cache_path}")

        return history
