import os
import re
import html
import pandas as pd
from typing import Any, Dict, List
from src.utils import get_logger

os.environ["TRANSFORMERS_BACKEND"] = "pytorch"
logger = get_logger(__name__)

# Define the path for processed (cached) data
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def clean_financial_text(text: str) -> str:
    """
    Cleans raw financial headlines/descriptions by stripping boilerplate publisher tags,
    HTML artifacts, ticker suffixes, and normalizes whitespace for high-precision NLP.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Unescape HTML entities (e.g. &amp; -> &)
    cleaned = html.unescape(text)

    # Strip HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    # Remove common boilerplate financial publisher footers & prefixes
    patterns = [
        r"\(Reuters\)\s*-\s*",
        r"\(Bloomberg\)\s*-\s*",
        r"\(AP\)\s*-\s*",
        r"\(PR\s*Newswire\)\s*-\s*",
        r"\(Business\s*Wire\)\s*-\s*",
        r"\(GlobeNewswire\)\s*-\s*",
        r"via\s+Zacks\s+Investment\s+Research",
        r"\|\s*The\s*Motley\s*Fool",
        r"\|\s*Seeking\s*Alpha",
        r"\|\s*Benzinga",
        r"\|\s*Investopedia",
        r"https?://\S+",
        r"www\.\S+",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _parse_analyzer_output(res: Any) -> Dict[str, Any]:
    """
    Normalizes pipeline output whether given full multi-class probabilities (list of dicts)
    or single top-label dictionary (legacy format).

    Returns a standardized dictionary with:
      - sentiment_label: 'positive' | 'negative' | 'neutral'
      - sentiment_score: signed polarity score in [-1.0, +1.0] (P_pos - P_neg)
      - prob_positive: float in [0.0, 1.0]
      - prob_negative: float in [0.0, 1.0]
      - prob_neutral: float in [0.0, 1.0]
      - sentiment_confidence: max probability
    """
    prob_pos = 0.0
    prob_neg = 0.0
    prob_neu = 0.0

    if isinstance(res, list):
        # Multi-class output: [{'label': 'Positive', 'score': 0.9}, {'label': 'Negative', 'score': ...}]
        for item in res:
            lbl = str(item.get("label", "")).strip().lower()
            score = float(item.get("score", 0.0))
            if "pos" in lbl:
                prob_pos = score
            elif "neg" in lbl:
                prob_neg = score
            elif "neu" in lbl:
                prob_neu = score
    elif isinstance(res, dict):
        # Single label dictionary
        lbl = str(res.get("label", "")).strip().lower()
        score = float(res.get("score", 0.5))
        if "pos" in lbl:
            prob_pos = score
            prob_neu = max(0.0, 1.0 - score)
        elif "neg" in lbl:
            prob_neg = score
            prob_neu = max(0.0, 1.0 - score)
        else:
            prob_neu = score
            prob_pos = max(0.0, (1.0 - score) / 2.0)
            prob_neg = max(0.0, (1.0 - score) / 2.0)

    # Determine winning label
    scores = {"positive": prob_pos, "negative": prob_neg, "neutral": prob_neu}
    best_label = max(scores, key=scores.get)
    max_prob = scores[best_label]

    # Signed polarity score: +1.0 (strongly bullish) to -1.0 (strongly bearish)
    # If neutral dominates, score smoothly converges towards 0.0
    signed_score = round(prob_pos - prob_neg, 4)

    return {
        "sentiment_label": best_label,
        "sentiment_score": signed_score,
        "prob_positive": round(prob_pos, 4),
        "prob_negative": round(prob_neg, 4),
        "prob_neutral": round(prob_neu, 4),
        "sentiment_confidence": round(max_prob, 4),
    }


def get_sentiment(
    articles: pd.DataFrame,
    sentiment_analyzer: Any,
    ticker: str | None = None,
    cache_duration_hours: int = 24,
) -> pd.DataFrame:
    """
    Analyzes the sentiment of news articles using high-precision FinBERT pipeline,
    extracting full multi-class probability distributions, signed polarity scores,
    and optional file-based caching.
    """
    if ticker:
        cache_path = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_sentiment.csv")
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        if os.path.exists(cache_path):
            import time

            cache_age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600.0
            if cache_age_hours < cache_duration_hours:
                try:
                    logger.info(
                        f"Loading cached sentiment data for {ticker} from {cache_path}"
                    )
                    sentiment_df = pd.read_csv(
                        cache_path, index_col="publishedAt", parse_dates=True
                    )
                    return sentiment_df
                except Exception as e:
                    logger.warning(
                        f"Corrupted cache file found for {ticker} ({e}). Re-running sentiment analysis."
                    )
                    try:
                        os.remove(cache_path)
                    except OSError:
                        pass
            else:
                logger.info(
                    f"Sentiment cache for {ticker} is stale ({cache_age_hours:.1f}h old). Re-analyzing fresh news..."
                )

    if articles is None or articles.empty:
        return pd.DataFrame()

    logger.info(
        f"Running FinBERT sentiment analysis (caching {'enabled' if ticker else 'disabled'})."
    )

    from tqdm import tqdm

    # Determine available text columns
    text_columns = []
    if "Title" in articles.columns:
        text_columns.append("Title")
    if "description" in articles.columns:
        text_columns.append("description")

    if not text_columns:
        logger.warning(
            "No 'Title' or 'description' columns found for sentiment analysis. Skipping."
        )
        return articles

    articles_for_sentiment = articles[text_columns].dropna(subset=text_columns).copy()

    # Apply specialized financial text cleaning
    clean_titles = (
        articles_for_sentiment["Title"].apply(clean_financial_text)
        if "Title" in text_columns
        else None
    )
    clean_descs = (
        articles_for_sentiment["description"].apply(clean_financial_text)
        if "description" in text_columns
        else None
    )

    if clean_titles is not None and clean_descs is not None:
        articles_for_sentiment["text"] = clean_titles + ". " + clean_descs
    elif clean_titles is not None:
        articles_for_sentiment["text"] = clean_titles
    elif clean_descs is not None:
        articles_for_sentiment["text"] = clean_descs

    # Filter out empty texts
    articles_for_sentiment["text"] = articles_for_sentiment["text"].str.strip()
    articles_for_sentiment = articles_for_sentiment[
        articles_for_sentiment["text"] != ""
    ]

    if articles_for_sentiment.empty:
        return articles

    results = []
    text_list = articles_for_sentiment["text"].tolist()
    chunk_size = 32

    for i in tqdm(
        range(0, len(text_list), chunk_size), desc="Analyzing sentiment (FinBERT)"
    ):
        chunk = text_list[i : i + chunk_size]
        raw_res = sentiment_analyzer(chunk)
        results.extend(raw_res)

    parsed_metrics: List[Dict[str, Any]] = [_parse_analyzer_output(r) for r in results]
    parsed_df = pd.DataFrame(parsed_metrics, index=articles_for_sentiment.index)

    # Join metrics back to original articles DataFrame
    cols_to_join = [
        "sentiment_label",
        "sentiment_score",
        "prob_positive",
        "prob_negative",
        "prob_neutral",
        "sentiment_confidence",
    ]
    for c in cols_to_join:
        if c in articles.columns:
            articles.drop(columns=[c], inplace=True)

    enriched_articles = articles.join(parsed_df[cols_to_join])

    enriched_articles["sentiment_label"] = enriched_articles["sentiment_label"].fillna(
        "neutral"
    )
    enriched_articles["sentiment_score"] = enriched_articles["sentiment_score"].fillna(
        0.0
    )
    enriched_articles["prob_positive"] = enriched_articles["prob_positive"].fillna(0.0)
    enriched_articles["prob_negative"] = enriched_articles["prob_negative"].fillna(0.0)
    enriched_articles["prob_neutral"] = enriched_articles["prob_neutral"].fillna(1.0)
    enriched_articles["sentiment_confidence"] = enriched_articles[
        "sentiment_confidence"
    ].fillna(0.5)

    if ticker:
        enriched_articles.to_csv(cache_path, index=True)
        logger.info(f"Saved enhanced FinBERT sentiment data to {cache_path}")

    return enriched_articles


def analyze_sentiment(
    articles: pd.DataFrame,
    ticker: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper for sentiment scoring using the cached FinBERT pipeline."""
    if articles is None or articles.empty:
        return pd.DataFrame()
    from src.preprocessing import _load_sentiment_analyzer

    analyzer = _load_sentiment_analyzer()
    cache_dur = 24 if use_cache else 0
    return get_sentiment(
        articles,
        sentiment_analyzer=analyzer,
        ticker=ticker,
        cache_duration_hours=cache_dur,
    )
