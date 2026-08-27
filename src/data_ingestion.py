import os
import requests
import pandas as pd
from newsapi import NewsApiClient
import yfinance as yf
from src.utils import get_logger, sanitize_filename, safe_path_join
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
            link = item.get("link") or item.get("content", {}).get(
                "canonicalUrl", {}
            ).get("url", "")
            summary = item.get("summary") or item.get("content", {}).get("summary", "")

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


import defusedxml.ElementTree as defused_ET
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any, List


def _fetch_google_news_rss(ticker: str) -> pd.DataFrame:
    """
    Fetches real-time financial news headlines directly from public Google News RSS.
    100% free, unlimited, and requires zero API keys.
    """
    try:
        url = f"https://news.google.com/rss/search?q={ticker}+stock+market&hl=en-US&gl=US&ceid=US:en"
        session = _get_browser_session()
        res = session.get(url, timeout=6)
        if res.status_code == 200:
            root = defused_ET.fromstring(res.content)
            articles = []
            for item in root.findall("./channel/item"):
                title_elem = item.find("title")
                pubdate_elem = item.find("pubDate")
                link_elem = item.find("link")
                source_elem = item.find("source")

                if title_elem is not None and title_elem.text:
                    title = title_elem.text
                    pub_dt = (
                        pd.to_datetime(pubdate_elem.text, utc=True)
                        if pubdate_elem is not None and pubdate_elem.text
                        else pd.to_datetime("now", utc=True)
                    )
                    link = link_elem.text if link_elem is not None else ""
                    src_name = (
                        source_elem.text
                        if source_elem is not None and source_elem.text
                        else "Google News"
                    )

                    articles.append(
                        {
                            "publishedAt": pub_dt,
                            "Title": title,
                            "description": title,
                            "url": link,
                            "source": {"name": src_name},
                        }
                    )

            if articles:
                df = pd.DataFrame(articles)
                logger.info(
                    f"Successfully fetched {len(df)} live news articles for {ticker} from Google News RSS"
                )
                return df
    except Exception as e:
        logger.debug(f"Google News RSS fetch notice for {ticker}: {e}")
    return pd.DataFrame()


def _fetch_finnhub_news(ticker: str, api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetches real-time institutional market news from Finnhub Company News API."""
    key = api_key or os.getenv("FINNHUB_API_KEY")
    if not key:
        return pd.DataFrame()
    try:
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        from_str = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_str}&to={today_str}&token={key}"
        res = requests.get(url, timeout=6)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and data:
                articles = []
                for item in data[:30]:
                    headline = item.get("headline")
                    if not headline:
                        continue
                    pub_ts = item.get("datetime", time.time())
                    articles.append(
                        {
                            "publishedAt": pd.to_datetime(pub_ts, unit="s", utc=True),
                            "Title": headline,
                            "description": item.get("summary", headline),
                            "url": item.get("url", ""),
                            "source": {"name": item.get("source", "Finnhub")},
                        }
                    )
                if articles:
                    df = pd.DataFrame(articles)
                    logger.info(
                        f"Successfully fetched {len(df)} live news articles for {ticker} from Finnhub"
                    )
                    return df
    except Exception as e:
        logger.debug(f"Finnhub news fetch notice for {ticker}: {e}")
    return pd.DataFrame()


def _fetch_marketaux_news(ticker: str, api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetches financial news from Marketaux API."""
    key = api_key or os.getenv("MARKETAUX_API_KEY")
    if not key:
        return pd.DataFrame()
    try:
        url = f"https://api.marketaux.com/v1/news/all?symbols={ticker}&language=en&api_token={key}"
        res = requests.get(url, timeout=6)
        if res.status_code == 200:
            data = res.json().get("data", [])
            articles = []
            for item in data:
                title = item.get("title")
                if not title:
                    continue
                articles.append(
                    {
                        "publishedAt": pd.to_datetime(
                            item.get("published_at", "now"), utc=True
                        ),
                        "Title": title,
                        "description": item.get("description", title),
                        "url": item.get("url", ""),
                        "source": {"name": item.get("source", "Marketaux")},
                    }
                )
            if articles:
                df = pd.DataFrame(articles)
                logger.info(
                    f"Successfully fetched {len(df)} live news articles for {ticker} from Marketaux"
                )
                return df
    except Exception as e:
        logger.debug(f"Marketaux news fetch notice for {ticker}: {e}")
    return pd.DataFrame()


def _fetch_polygon_news_feed(
    ticker: str, api_key: Optional[str] = None
) -> pd.DataFrame:
    """Fetches reference news from Polygon.io API."""
    key = api_key or os.getenv("POLYGON_API_KEY")
    if not key:
        return pd.DataFrame()
    try:
        url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&limit=20&apiKey={key}"
        res = requests.get(url, timeout=6)
        if res.status_code == 200:
            results = res.json().get("results", [])
            articles = []
            for item in results:
                title = item.get("title")
                if not title:
                    continue
                articles.append(
                    {
                        "publishedAt": pd.to_datetime(
                            item.get("published_utc", "now"), utc=True
                        ),
                        "Title": title,
                        "description": item.get("description", title),
                        "url": item.get("article_url", ""),
                        "source": {
                            "name": item.get("publisher", {}).get("name", "Polygon")
                        },
                    }
                )
            if articles:
                df = pd.DataFrame(articles)
                logger.info(
                    f"Successfully fetched {len(df)} live news articles for {ticker} from Polygon"
                )
                return df
    except Exception as e:
        logger.debug(f"Polygon news fetch notice for {ticker}: {e}")
    return pd.DataFrame()


def get_news(
    ticker: str,
    api_key: str = None,
    cache_duration_hours: int = 24,
    retries: int = 3,
    backoff_factor: float = 1,
    use_cache: bool = None,
) -> pd.DataFrame:
    """
    Enterprise Multi-Source News Router:
    Cascades through Google News RSS -> Yahoo Finance -> Finnhub -> Marketaux -> Polygon -> NewsAPI -> Local Cache.
    """
    clean_ticker = sanitize_filename(ticker)
    cache_path = safe_path_join(DATA_DIR, f"{clean_ticker}_news.csv")
    os.makedirs(DATA_DIR, exist_ok=True)

    should_load_cache = False
    if os.path.exists(cache_path):
        if use_cache is True:
            should_load_cache = True
        elif use_cache is False:
            should_load_cache = False
        else:
            cache_age_seconds = time.time() - os.path.getmtime(cache_path)
            if cache_age_seconds < cache_duration_hours * 3600:
                should_load_cache = True
            else:
                logger.info(
                    f"News cache for {ticker} is stale. Re-fetching fresh live news..."
                )

    if should_load_cache:
        logger.info(f"Loading news for {ticker} from cache...")
        articles_df = pd.read_csv(cache_path)
    else:
        logger.info(f"Routing live news aggregation for {ticker}...")
        # 1. Primary: Google News RSS (Unlimited, Real-time)
        articles_df = _fetch_google_news_rss(ticker)

        # 2. Tier 2: Yahoo Finance Live Stream
        if articles_df.empty:
            articles_df = _fetch_yfinance_news(ticker)

        # 3. Tier 3: Finnhub Company News
        if articles_df.empty:
            articles_df = _fetch_finnhub_news(ticker)

        # 4. Tier 4: Marketaux Financial News
        if articles_df.empty:
            articles_df = _fetch_marketaux_news(ticker)

        # 5. Tier 5: Polygon.io Reference News
        if articles_df.empty:
            articles_df = _fetch_polygon_news_feed(ticker)

        # 6. Tier 6: NewsAPI.org
        if articles_df.empty:
            key = api_key or os.getenv("NEWS_API_KEY")
            if key and isinstance(key, str):
                try:
                    newsapi = NewsApiClient(api_key=key)
                    all_articles = newsapi.get_everything(
                        q=ticker,
                        language="en",
                        sort_by="publishedAt",
                        page_size=100,
                    )
                    if all_articles.get("articles"):
                        articles_df = pd.DataFrame(all_articles["articles"])
                        articles_df.rename(columns={"title": "Title"}, inplace=True)
                except Exception as e:
                    logger.debug(f"NewsAPI query notice for {ticker}: {e}")

        # 7. Fallback: Existing Cache or Synthetic Generation
        if articles_df.empty:
            if os.path.exists(cache_path):
                logger.warning(f"Using existing cached news for {ticker} as fallback.")
                articles_df = pd.read_csv(cache_path)
            else:
                articles_df = _generate_dummy_news(ticker)

        articles_df.to_csv(cache_path, index=False)
        logger.info(f"Saved fresh news ({len(articles_df)} articles) to {cache_path}")

    # Standardize the DataFrame to have a timezone-aware DatetimeIndex
    if "publishedAt" in articles_df.columns:
        articles_df["publishedAt"] = pd.to_datetime(
            articles_df["publishedAt"], utc=True
        )
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
        days_ago = random.randint(0, 14)  # nosec B311
        pub_date = today - datetime.timedelta(days=days_ago)

        dummy_articles.append(
            {
                "publishedAt": pub_date.isoformat(),
                "Title": random.choice(titles),  # nosec B311
                "description": f"This is a generated dummy description for {ticker} to demonstrate pipeline capabilities without an active API connection.",
                "url": "https://example.com/dummy-news",
                "source": {"name": "Portfolio Fallback Generator"},
            }
        )

    return pd.DataFrame(dummy_articles)


def _fetch_alpaca_price_history(ticker: str, period: str = "10y") -> pd.DataFrame:
    """Fetches official US equity daily bars from Alpaca Data API v2."""
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        return pd.DataFrame()

    try:
        url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
        start_year = 2015 if "10y" in period else 2022
        params = {
            "timeframe": "1Day",
            "start": f"{start_year}-01-01T00:00:00Z",
            "limit": 10000,
            "adjustment": "all",
            "feed": "iex",
        }
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
        res = requests.get(url, params=params, headers=headers, timeout=12)
        if res.status_code == 200:
            data = res.json()
            bars = data.get("bars", [])
            if not bars:
                return pd.DataFrame()
            df = pd.DataFrame(bars)
            df.rename(
                columns={
                    "t": "Date",
                    "o": "Open",
                    "h": "High",
                    "l": "Low",
                    "c": "Close",
                    "v": "Volume",
                },
                inplace=True,
            )
            df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.normalize()
            df.set_index("Date", inplace=True)
            df["Dividends"] = 0.0
            df["Stock Splits"] = 0.0
            logger.info(
                f"[Alpaca API] Successfully fetched {len(df)} bars for {ticker} up to {df.index[-1].strftime('%Y-%m-%d')}"
            )
            return df
    except Exception as e:
        logger.warning(f"Alpaca data fetch failed for {ticker}: {e}")
    return pd.DataFrame()


def _fetch_eodhd_price_history(ticker: str, period: str = "10y") -> pd.DataFrame:
    """Fetches daily historical bars from EODHD (EOD Historical Data) API."""
    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        return pd.DataFrame()

    try:
        start_str = "2015-01-01" if "10y" in period else "2022-01-01"
        url = f"https://eodhd.com/api/eod/{ticker}.US"
        params = {
            "api_token": api_key,
            "from": start_str,
            "fmt": "json",
            "period": "d",
        }
        res = requests.get(url, params=params, timeout=12)
        if res.status_code == 200:
            data = res.json()
            if not isinstance(data, list) or not data:
                return pd.DataFrame()
            df = pd.DataFrame(data)
            df.rename(
                columns={
                    "date": "Date",
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                },
                inplace=True,
            )
            df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.normalize()
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
            df["Dividends"] = 0.0
            df["Stock Splits"] = 0.0
            logger.info(
                f"[EODHD API] Successfully fetched {len(df)} bars for {ticker} up to {df.index[-1].strftime('%Y-%m-%d')}"
            )
            return df
    except Exception as e:
        logger.warning(f"EODHD data fetch failed for {ticker}: {e}")
    return pd.DataFrame()


def _fetch_polygon_price_history(ticker: str, period: str = "10y") -> pd.DataFrame:
    """Fetches daily bars from Polygon.io API."""
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        return pd.DataFrame()

    try:
        import datetime

        today_str = datetime.date.today().strftime("%Y-%m-%d")
        start_str = "2015-01-01" if "10y" in period else "2022-01-01"
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_str}/{today_str}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": api_key,
        }
        res = requests.get(url, params=params, timeout=12)
        if res.status_code == 200:
            data = res.json()
            results = data.get("results", [])
            if not results:
                return pd.DataFrame()
            df = pd.DataFrame(results)
            df.rename(
                columns={
                    "t": "Date",
                    "o": "Open",
                    "h": "High",
                    "l": "Low",
                    "c": "Close",
                    "v": "Volume",
                },
                inplace=True,
            )
            df["Date"] = pd.to_datetime(df["Date"], unit="ms", utc=True).dt.normalize()
            df.set_index("Date", inplace=True)
            df["Dividends"] = 0.0
            df["Stock Splits"] = 0.0
            logger.info(
                f"[Polygon.io] Successfully fetched {len(df)} bars for {ticker} up to {df.index[-1].strftime('%Y-%m-%d')}"
            )
            return df
    except Exception as e:
        logger.warning(f"Polygon data fetch failed for {ticker}: {e}")
    return pd.DataFrame()


def _fetch_fmp_price_history(ticker: str, period: str = "10y") -> pd.DataFrame:
    """Fetches daily bars from Financial Modeling Prep (FMP) API."""
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        return pd.DataFrame()

    try:
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
        params = {"apikey": api_key}
        res = requests.get(url, params=params, timeout=12)
        if res.status_code == 200:
            data = res.json()
            historical = data.get("historical", [])
            if not historical:
                return pd.DataFrame()
            df = pd.DataFrame(historical)
            df.rename(
                columns={
                    "date": "Date",
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                },
                inplace=True,
            )
            df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.normalize()
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
            df["Dividends"] = 0.0
            df["Stock Splits"] = 0.0
            logger.info(
                f"[FMP API] Successfully fetched {len(df)} bars for {ticker} up to {df.index[-1].strftime('%Y-%m-%d')}"
            )
            return df
    except Exception as e:
        logger.warning(f"FMP data fetch failed for {ticker}: {e}")
    return pd.DataFrame()


def _fetch_alpaca_news(ticker: str) -> pd.DataFrame:
    """Fetches latest financial news articles from Alpaca News API."""
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        return pd.DataFrame()

    try:
        url = "https://data.alpaca.markets/v1beta1/news"
        params = {"symbols": ticker, "limit": 50, "include_content": "false"}
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
        res = requests.get(url, params=params, headers=headers, timeout=12)
        if res.status_code == 200:
            news_items = res.json().get("news", [])
            articles = []
            for item in news_items:
                articles.append(
                    {
                        "publishedAt": pd.to_datetime(item.get("created_at"), utc=True),
                        "Title": item.get("headline"),
                        "description": item.get("summary", ""),
                        "url": item.get("url", ""),
                        "source": {"name": item.get("source", "Alpaca")},
                    }
                )
            if articles:
                df = pd.DataFrame(articles)
                logger.info(
                    f"[Alpaca News] Successfully fetched {len(df)} news articles for {ticker}"
                )
                return df
    except Exception as e:
        logger.warning(f"Alpaca news fetch failed for {ticker}: {e}")
    return pd.DataFrame()


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
                datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
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
                f"[Yahoo Chart] Directly fetched {len(df)} price bars for {ticker} up to {df.index[-1].strftime('%Y-%m-%d')}"
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
    use_cache: bool = None,
) -> pd.DataFrame:
    """
    Enterprise Data Router: Fetches historical price data up to today using the best available provider.
    Priority: Alpaca Data API v2 -> Polygon.io -> FMP -> EODHD -> Yahoo Direct Chart -> yfinance -> Cache
    """
    clean_ticker = sanitize_filename(ticker)
    cache_path = safe_path_join(DATA_DIR, f"{clean_ticker}_price_history.csv")
    os.makedirs(DATA_DIR, exist_ok=True)

    should_load_cache = False
    if os.path.exists(cache_path):
        if use_cache is True:
            should_load_cache = True
        elif use_cache is False:
            should_load_cache = False
        else:
            cache_age_seconds = time.time() - os.path.getmtime(cache_path)
            if cache_age_seconds < cache_duration_hours * 3600:
                should_load_cache = True
            else:
                logger.info(
                    f"Price history cache for {ticker} is stale. Re-fetching..."
                )

    if should_load_cache:
        logger.info(f"Loading price history for {ticker} from cache...")
        history = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        if history.index.tz is None:
            history.index = history.index.tz_localize("UTC").normalize()
        else:
            history.index = history.index.tz_convert("UTC").normalize()
    else:
        logger.info(f"Routing live price history fetch for {ticker}...")
        # 1. Alpaca Markets Data API v2
        history = _fetch_alpaca_price_history(ticker, period=period)

        # 2. Polygon.io
        if history.empty:
            history = _fetch_polygon_price_history(ticker, period=period)

        # 3. Financial Modeling Prep (FMP)
        if history.empty:
            history = _fetch_fmp_price_history(ticker, period=period)

        # 4. EODHD (EOD Historical Data)
        if history.empty:
            history = _fetch_eodhd_price_history(ticker, period=period)

        # 5. Direct Yahoo Finance Chart API (zero-key fallback)
        if history.empty:
            history = _fetch_direct_yahoo_chart(ticker, period=period)

        # 6. yfinance library
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

        # 7. Fallback to existing cache if all live routes failed
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


def get_vix_data(
    period: str = "10y",
    cache_duration_hours: int = 24,
    retries: int = 5,
    backoff_factor: float = 10,
    use_cache: bool = None,
) -> pd.DataFrame:
    """
    Fetches historical data for the CBOE Volatility Index (VIX).

    Args:
        period (str): The time period for the data. Defaults to "10y".
        use_cache (bool, optional): Force cache usage if True, or force live fetch if False.

    Returns:
        pd.DataFrame: A DataFrame containing VIX price history.
    """
    cache_path = os.path.join(DATA_DIR, "vix_history.csv")
    os.makedirs(DATA_DIR, exist_ok=True)

    should_load_cache = False
    if os.path.exists(cache_path):
        if use_cache is True:
            should_load_cache = True
        elif use_cache is False:
            should_load_cache = False
        else:
            cache_age_seconds = time.time() - os.path.getmtime(cache_path)
            if cache_age_seconds < cache_duration_hours * 3600:
                should_load_cache = True

    if should_load_cache:
        logger.info("Loading VIX data from cache...")
        history = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        if history.index.tz is None:
            history.index = history.index.tz_localize("UTC").normalize()
        else:
            history.index = history.index.tz_convert("UTC").normalize()
        return history

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
