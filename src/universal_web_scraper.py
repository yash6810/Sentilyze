"""
Universal Web Scraper & Financial Intelligence Extractor for Sentilyze.
Integrates Scrapling (Adaptive anti-bot bypass), Trafilatura (Boilerplate removal),
BeautifulSoup4 (Table & DOM extraction), and FinBERT NLP sentiment scoring.
"""

from typing import Any, Dict, List, Optional, Set
import os
import re
import time
import requests
import pandas as pd
from datetime import datetime, timezone
import defusedxml.ElementTree as defused_ET
from bs4 import BeautifulSoup

from src.utils import get_logger

logger = get_logger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Known ticker list regex pattern
US_POPULAR_TICKERS = {
    "NVDA",
    "AAPL",
    "MSFT",
    "GOOGL",
    "GOOG",
    "META",
    "AMZN",
    "TSLA",
    "PLTR",
    "AMD",
    "INTC",
    "AVGO",
    "QCOM",
    "ARM",
    "SMCI",
    "NFLX",
    "DIS",
    "SPY",
    "QQQ",
    "IWM",
    "VIX",
    "COIN",
    "MSTR",
    "BABA",
    "ORCL",
    "CRM",
    "UBER",
    "LYFT",
    "SHOP",
    "SQ",
    "PYPL",
    "WMT",
    "COST",
    "HD",
    "UNH",
    "JNJ",
    "LLY",
    "NVO",
    "PFE",
    "MRK",
    "JPM",
    "BAC",
    "GS",
    "MS",
    "WFC",
    "C",
    "XOM",
    "CVX",
    "OXY",
}


def _get_browser_session() -> requests.Session:
    """Creates a configured requests session with realistic browser headers."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
        }
    )
    return session


def detect_stock_tickers(text: str) -> List[str]:
    """
    Extracts stock ticker symbols from unstructured text using cashtags ($TICKER)
    and uppercase token boundary matching against known equities.
    """
    if not text:
        return []

    found: Set[str] = set()

    # 1. Direct Cashtags: $NVDA, $AAPL, etc.
    cashtags = re.findall(r"\$([A-Z]{1,5})\b", text)
    for t in cashtags:
        found.add(t.upper())

    # 2. Known popular tickers with word boundaries
    tokens = re.findall(r"\b[A-Z]{2,5}\b", text)
    for tok in tokens:
        if tok in US_POPULAR_TICKERS:
            found.add(tok)

    return sorted(list(found))


def extract_financial_tables_from_html(html_content: str) -> List[pd.DataFrame]:
    """
    Parses all HTML <table> tags into clean pandas DataFrames.
    Filters out layout/navigation tables to return data-rich tables.
    """
    if not html_content:
        return []

    tables: List[pd.DataFrame] = []
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        for table_elem in soup.find_all("table"):
            rows = []
            for tr in table_elem.find_all("tr"):
                cells = [
                    cell.get_text(separator=" ", strip=True)
                    for cell in tr.find_all(["th", "td"])
                ]
                if any(cells):
                    rows.append(cells)

            if len(rows) >= 2 and len(rows[0]) >= 2:
                # Attempt to use first row as header if lengths match
                header = rows[0]
                data = rows[1:]
                # Normalize column lengths
                max_cols = max(len(r) for r in rows)
                header_padded = header + [
                    f"Col_{i}" for i in range(len(header), max_cols)
                ]
                padded_data = [
                    r + [""] * (max_cols - len(r))
                    for r in data
                    if len(r) == max_cols or len(r) > 0
                ]
                if padded_data:
                    df = pd.DataFrame(padded_data, columns=header_padded[:max_cols])
                    tables.append(df)
    except Exception as e:
        logger.debug(f"Table extraction notice: {e}")

    return tables


def _score_text_with_finbert(text: str) -> Dict[str, Any]:
    """Scores extracted text using FinBERT sentiment pipeline."""
    if not text or len(text.strip()) < 10:
        return {
            "sentiment_label": "neutral",
            "sentiment_score": 0.0,
            "prob_positive": 0.0,
            "prob_negative": 0.0,
            "prob_neutral": 1.0,
            "sentiment_confidence": 0.5,
        }

    try:
        from src.sentiment_analysis import clean_financial_text, _parse_analyzer_output
        from src.preprocessing import _load_sentiment_analyzer

        cleaned = clean_financial_text(text[:1500])  # FinBERT token limit
        analyzer = _load_sentiment_analyzer()
        res = analyzer(cleaned)
        return _parse_analyzer_output(res)
    except Exception as e:
        logger.debug(f"FinBERT scoring notice for scraped text: {e}")
        # Rule-based fallback
        txt_lower = text.lower()
        bull_hits = sum(
            1
            for w in [
                "surge",
                "beat",
                "record",
                "upgrade",
                "profit",
                "growth",
                "dividend",
            ]
            if w in txt_lower
        )
        bear_hits = sum(
            1
            for w in [
                "drop",
                "miss",
                "loss",
                "downgrade",
                "investigation",
                "plunge",
                "lawsuit",
            ]
            if w in txt_lower
        )
        if bull_hits > bear_hits:
            return {
                "sentiment_label": "positive",
                "sentiment_score": 0.5,
                "prob_positive": 0.7,
                "prob_negative": 0.1,
                "prob_neutral": 0.2,
                "sentiment_confidence": 0.7,
            }
        elif bear_hits > bull_hits:
            return {
                "sentiment_label": "negative",
                "sentiment_score": -0.5,
                "prob_positive": 0.1,
                "prob_negative": 0.7,
                "prob_neutral": 0.2,
                "sentiment_confidence": 0.7,
            }
        return {
            "sentiment_label": "neutral",
            "sentiment_score": 0.0,
            "prob_positive": 0.2,
            "prob_negative": 0.2,
            "prob_neutral": 0.6,
            "sentiment_confidence": 0.6,
        }


def _extract_summary_sentences(text: str, max_sentences: int = 3) -> str:
    """Extracts top lead narrative sentences as an executive summary."""
    if not text:
        return ""
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 40]
    if not paragraphs:
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 30]
        return ". ".join(sentences[:max_sentences]) + ("." if sentences else "")
    return " ".join(paragraphs[:max_sentences])


class UniversalWebScraper:
    """
    Universal Adaptive Web Scraper supporting Scrapling, Trafilatura, and BeautifulSoup4.
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = _get_browser_session()

    def scrape(
        self,
        url: str,
        extract_tables: bool = True,
        score_sentiment: bool = True,
    ) -> Dict[str, Any]:
        """
        Scrapes any URL, cleans DOM noise, extracts structured text, tables, tickers,
        and computes FinBERT sentiment score.
        """
        if not url or not (url.startswith("http://") or url.startswith("https://")):
            return {
                "success": False,
                "error": "Invalid URL. URL must start with http:// or https://",
                "url": url,
            }

        html_content = ""
        engine_used = "Standard"
        title = ""
        clean_text = ""
        published_at = ""
        author = ""
        meta_desc = ""

        # 1. Attempt Scrapling Fetcher
        try:
            from scrapling.fetchers import Fetcher

            fetcher = Fetcher()
            response = fetcher.get(url, timeout=self.timeout)
            if response and response.status == 200:
                html_content = (
                    response.text
                    if hasattr(response, "text")
                    else response.content.decode("utf-8", errors="ignore")
                )
                engine_used = "Scrapling (Adaptive)"
        except Exception as e:
            logger.debug(f"Scrapling fetch fallback to requests: {e}")

        # 2. Fallback to requests with browser session
        if not html_content:
            try:
                res = self.session.get(url, timeout=self.timeout)
                if res.status_code == 200:
                    html_content = res.text
                    engine_used = "Requests + BeautifulSoup"
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {res.status_code} Error while fetching {url}",
                        "url": url,
                    }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Connection error: {str(e)}",
                    "url": url,
                }

        # 3. Extract Text & Metadata using Trafilatura
        try:
            import trafilatura

            extracted = trafilatura.extract(
                html_content,
                url=url,
                include_tables=True,
                include_links=True,
                include_comments=False,
                output_format="txt",
            )
            metadata = trafilatura.extract_metadata(html_content)

            if extracted:
                clean_text = extracted
                engine_used = f"{engine_used} + Trafilatura"

            if metadata:
                title = metadata.title or ""
                author = metadata.author or ""
                published_at = metadata.date or ""
                meta_desc = metadata.description or ""
        except Exception as e:
            logger.debug(f"Trafilatura extraction fallback: {e}")

        # 4. Fallback DOM Parsing with BeautifulSoup if Trafilatura missed metadata or text
        soup = BeautifulSoup(html_content, "html.parser")

        if not title:
            title_tag = soup.find("title") or soup.find("h1")
            title = title_tag.get_text(strip=True) if title_tag else "No Title Found"

        if not meta_desc:
            meta_tag = soup.find("meta", attrs={"name": "description"}) or soup.find(
                "meta", attrs={"property": "og:description"}
            )
            meta_desc = meta_tag.get("content", "").strip() if meta_tag else ""

        if not clean_text:
            # Strip noise elements
            for tag in soup(
                [
                    "script",
                    "style",
                    "nav",
                    "footer",
                    "header",
                    "aside",
                    "noscript",
                    "svg",
                    "form",
                ]
            ):
                tag.decompose()
            paragraphs = [
                p.get_text(strip=True)
                for p in soup.find_all(["p", "h1", "h2", "h3", "h4", "li"])
            ]
            clean_text = "\n\n".join([p for p in paragraphs if len(p) > 25])

        # Extract Tables
        tables = (
            extract_financial_tables_from_html(html_content) if extract_tables else []
        )

        # Detect Tickers
        detected_tickers = detect_stock_tickers(f"{title} {clean_text}")

        # Compute Word Count
        words = clean_text.split()
        word_count = len(words)
        reading_time_mins = round(max(1, word_count / 200), 1)

        # Generate Executive Summary
        summary = _extract_summary_sentences(clean_text, max_sentences=3)

        # Sentiment Analysis
        sentiment = (
            _score_text_with_finbert(f"{title}\n{clean_text}")
            if score_sentiment
            else {}
        )

        # Links extraction
        outbound_links = []
        for a in soup.find_all("a", href=True)[:25]:
            href = a["href"]
            txt = a.get_text(strip=True)
            if txt and (href.startswith("http://") or href.startswith("https://")):
                outbound_links.append({"text": txt[:60], "url": href})

        return {
            "success": True,
            "url": url,
            "title": title,
            "author": author or "Editorial Desk / Official Filing",
            "published_at": published_at
            or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "meta_description": meta_desc,
            "clean_text": clean_text,
            "summary": summary,
            "word_count": word_count,
            "reading_time_mins": reading_time_mins,
            "tables_count": len(tables),
            "tables": tables,
            "detected_tickers": detected_tickers,
            "outbound_links": outbound_links,
            "sentiment": sentiment,
            "engine_used": engine_used,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        }


# Global singleton instance
_scraper_instance = UniversalWebScraper()


def scrape_url(
    url: str, extract_tables: bool = True, score_sentiment: bool = True
) -> Dict[str, Any]:
    """Convenience functional wrapper to scrape any website URL."""
    return _scraper_instance.scrape(
        url, extract_tables=extract_tables, score_sentiment=score_sentiment
    )


def batch_scrape_urls(urls: List[str]) -> List[Dict[str, Any]]:
    """Scrapes a list of URLs concurrently or sequentially."""
    results = []
    for u in urls:
        results.append(scrape_url(u))
    return results


def scrape_ticker_news_from_web(ticker: str, limit: int = 20) -> pd.DataFrame:
    """
    Multi-hub web scraper that retrieves real-time financial news headlines,
    SEC 8-K material event disclosures, and PR releases directly from financial
    portals (Finviz, SEC EDGAR, and Google News RSS).
    Used as an automatic fallback whenever conventional API keys or feeds fail.
    """
    clean_tk = ticker.strip().upper()
    articles: List[Dict[str, Any]] = []
    session = _get_browser_session()

    # 1. Finviz Live Financial News Table
    try:
        finviz_url = f"https://finviz.com/quote.ashx?t={clean_tk}"
        res = session.get(finviz_url, timeout=6)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            news_table = soup.find("table", id="news-table")
            if news_table:
                current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                for row in news_table.find_all("tr")[:limit]:
                    tds = row.find_all("td")
                    if len(tds) >= 2:
                        date_text = tds[0].get_text(strip=True)
                        link_elem = tds[1].find("a")
                        src_elem = tds[1].find("span")

                        if link_elem:
                            title = link_elem.get_text(strip=True)
                            url = link_elem.get("href", "")
                            source_name = (
                                src_elem.get_text(strip=True)
                                .replace("(", "")
                                .replace(")", "")
                                if src_elem
                                else "Finviz Wire"
                            )

                            # Parse date_text: might be "Sep-08-26 09:30AM" or just "09:30AM"
                            pub_dt = datetime.now(timezone.utc)
                            if " " in date_text:
                                parts = date_text.split()
                                current_date = parts[0]
                                try:
                                    pub_dt = pd.to_datetime(
                                        f"{parts[0]} {parts[1]}", utc=True
                                    )
                                except Exception:
                                    pub_dt = datetime.now(timezone.utc)
                            else:
                                try:
                                    pub_dt = pd.to_datetime(
                                        f"{current_date} {date_text}", utc=True
                                    )
                                except Exception:
                                    pub_dt = datetime.now(timezone.utc)

                            articles.append(
                                {
                                    "publishedAt": pub_dt,
                                    "Title": title,
                                    "description": title,
                                    "url": url,
                                    "source": {"name": source_name},
                                }
                            )
    except Exception as e:
        logger.debug(f"Finviz news scrape notice for {clean_tk}: {e}")

    # 2. SEC EDGAR 8-K Current Material Events
    try:
        sec_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={clean_tk}&type=8-K&dateb=&owner=include&count=6&output=atom"
        sec_session = requests.Session()
        sec_session.headers.update(
            {"User-Agent": "SentilyzeQuantResearch bot@sentilyze.ai"}
        )
        sec_res = sec_session.get(sec_url, timeout=6)
        if sec_res.status_code == 200:
            root = defused_ET.fromstring(sec_res.content)
            for entry in root.findall("{http://www.w3.org/2005/Atom}entry")[:5]:
                t_elem = entry.find("{http://www.w3.org/2005/Atom}title")
                u_elem = entry.find("{http://www.w3.org/2005/Atom}updated")
                l_elem = entry.find("{http://www.w3.org/2005/Atom}link")

                if t_elem is not None and t_elem.text:
                    filing_title = f"[SEC 8-K Filing] {clean_tk} Material Current Report: {t_elem.text}"
                    filing_date = (
                        pd.to_datetime(u_elem.text, utc=True)
                        if u_elem is not None and u_elem.text
                        else datetime.now(timezone.utc)
                    )
                    filing_url = (
                        l_elem.attrib.get("href", "") if l_elem is not None else ""
                    )

                    articles.append(
                        {
                            "publishedAt": filing_date,
                            "Title": filing_title,
                            "description": filing_title,
                            "url": filing_url,
                            "source": {"name": "SEC EDGAR 8-K"},
                        }
                    )
    except Exception as e:
        logger.debug(f"SEC EDGAR scrape notice for {clean_tk}: {e}")

    if articles:
        df = pd.DataFrame(articles)
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], utc=True)
        logger.info(
            f"Successfully scraped {len(df)} real-time web headlines for {clean_tk} from web portals & SEC EDGAR"
        )
        return df

    return pd.DataFrame()
