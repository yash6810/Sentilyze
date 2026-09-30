import os
import re
import html
import pandas as pd
from typing import Any, Dict, List
from src.utils import get_logger, sanitize_filename, safe_path_join

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
    cached_df = None
    if ticker:
        clean_ticker = sanitize_filename(ticker)
        cache_path = safe_path_join(PROCESSED_DATA_DIR, f"{clean_ticker}_sentiment.csv")
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        if os.path.exists(cache_path):
            import time

            cache_age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600.0
            try:
                cached_df = pd.read_csv(
                    cache_path, index_col="publishedAt", parse_dates=True
                )
                if (
                    cache_duration_hours > 0
                    and cache_age_hours < cache_duration_hours
                    and not cached_df.empty
                ):
                    logger.info(
                        f"Loading cached sentiment data for {ticker} from {cache_path}"
                    )
                    return cached_df
            except Exception as e:
                logger.warning(
                    f"Corrupted cache file found for {ticker} ({e}). Re-running full sentiment analysis."
                )
                cached_df = None
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

    if articles is None or articles.empty:
        return (
            cached_df
            if (cached_df is not None and not cached_df.empty)
            else pd.DataFrame()
        )

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

    # Determine which articles are truly new / unscored (Delta Increment)
    articles_to_score = articles.copy()
    if cached_df is not None and not cached_df.empty:
        # Match against existing cached headlines by normalized Title
        if "Title" in articles.columns and "Title" in cached_df.columns:
            existing_titles = set(
                cached_df["Title"].dropna().astype(str).str.strip().str.lower()
            )
            new_mask = (
                ~articles["Title"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(existing_titles)
            )
            articles_to_score = articles[new_mask].copy()

    if articles_to_score.empty and cached_df is not None and not cached_df.empty:
        logger.info(
            f"⚡ Delta Cache Hit: All {len(articles)} headlines for {ticker} already scored. Fast-skipping FinBERT."
        )
        try:
            # Touch cache file to update mtime
            os.utime(cache_path, None)
        except OSError:
            pass
        return cached_df

    delta_count = len(articles_to_score)
    total_count = len(articles)
    if delta_count < total_count:
        logger.info(
            f"⚡ Incremental Sentiment: Scoring {delta_count} fresh headlines for {ticker} (skipping {total_count - delta_count} cached)..."
        )
    else:
        logger.info(
            f"Running FinBERT sentiment analysis on {total_count} headlines (caching {'enabled' if ticker else 'disabled'})."
        )

    from tqdm import tqdm

    articles_for_sentiment = (
        articles_to_score[text_columns].dropna(subset=text_columns).copy()
    )

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

    cols_to_join = [
        "sentiment_label",
        "sentiment_score",
        "prob_positive",
        "prob_negative",
        "prob_neutral",
        "sentiment_confidence",
    ]

    if not articles_for_sentiment.empty:
        if sentiment_analyzer is None:
            from src.preprocessing import _load_sentiment_analyzer

            sentiment_analyzer = _load_sentiment_analyzer()
        results = []
        text_list = articles_for_sentiment["text"].tolist()
        chunk_size = 32

        for i in tqdm(
            range(0, len(text_list), chunk_size), desc=f"FinBERT ({ticker or 'NLP'})"
        ):
            chunk = text_list[i : i + chunk_size]
            raw_res = sentiment_analyzer(chunk)
            results.extend(raw_res)

        parsed_metrics: List[Dict[str, Any]] = [
            _parse_analyzer_output(r) for r in results
        ]
        parsed_df = pd.DataFrame(parsed_metrics, index=articles_for_sentiment.index)

        # Drop old sentiment columns if present
        for c in cols_to_join:
            if c in articles_to_score.columns:
                articles_to_score.drop(columns=[c], inplace=True)

        scored_delta = articles_to_score.join(parsed_df[cols_to_join])
    else:
        scored_delta = articles_to_score.copy()
        for c in cols_to_join:
            if c not in scored_delta.columns:
                scored_delta[c] = None

    # Merge delta with cached_df if available
    if cached_df is not None and not cached_df.empty:
        combined_df = pd.concat([cached_df, scored_delta])
        # Drop duplicates if any by Title or index
        if "Title" in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=["Title"], keep="last")
        else:
            combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
        enriched_articles = combined_df.sort_index()
    else:
        enriched_articles = scored_delta

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

    # ⚡ Swift Priority Weighting & Time-Decay Recency
    try:
        from src.news_filter import assign_source_weight

        s_weights = []
        s_tiers = []
        for _, row in enriched_articles.iterrows():
            title_str = str(row.get("Title", ""))
            src_val = row.get("source", "")
            src_str = (
                src_val.get("name", "") if isinstance(src_val, dict) else str(src_val)
            )
            w, tier = assign_source_weight(title_str, src_str)
            s_weights.append(w)
            s_tiers.append(tier)

        enriched_articles["source_weight"] = s_weights
        enriched_articles["source_tier"] = s_tiers

        # Compute Recency Decay Factor
        now_utc = pd.to_datetime("now", utc=True)
        recency_multipliers = []
        if isinstance(enriched_articles.index, pd.DatetimeIndex):
            pub_dates = enriched_articles.index
        elif "publishedAt" in enriched_articles.columns:
            pub_dates = pd.to_datetime(
                enriched_articles["publishedAt"], utc=True, errors="coerce"
            )
        else:
            pub_dates = None

        for i, row in enriched_articles.iterrows():
            rec_factor = 1.0
            if pub_dates is not None:
                try:
                    dt = (
                        pub_dates[i]
                        if not isinstance(pub_dates, pd.DatetimeIndex)
                        else i
                    )
                    if pd.notnull(dt):
                        dt_utc = pd.to_datetime(dt, utc=True)
                        hours_old = max(
                            0.0, (now_utc - dt_utc).total_seconds() / 3600.0
                        )
                        if hours_old <= 4.0:
                            rec_factor = 2.0  # Breaking catalyst boost
                        elif hours_old <= 12.0:
                            rec_factor = 1.5
                        elif hours_old <= 24.0:
                            rec_factor = 1.0
                        else:
                            rec_factor = max(0.4, float(np.exp(-0.02 * hours_old)))
                except Exception:
                    rec_factor = 1.0
            recency_multipliers.append(round(rec_factor, 3))

        enriched_articles["recency_factor"] = recency_multipliers
        enriched_articles["effective_weight"] = (
            enriched_articles["source_weight"]
            * enriched_articles["sentiment_confidence"]
            * enriched_articles["recency_factor"]
        ).round(4)
    except Exception as sw_err:
        logger.debug(f"Swift source weighting notice: {sw_err}")
        enriched_articles["source_weight"] = 1.0
        enriched_articles["source_tier"] = "TIER3_GENERAL"
        enriched_articles["effective_weight"] = enriched_articles[
            "sentiment_confidence"
        ]

    if ticker:
        enriched_articles.to_csv(cache_path, index=True)
        logger.info(f"Saved enhanced FinBERT sentiment data to {cache_path}")

    return enriched_articles


def calculate_swift_sentiment_summary(
    enriched_articles: pd.DataFrame, ticker: str | None = None
) -> Dict[str, Any]:
    """
    Computes an institutional, Swift-Weighted summary of sentiment data,
    separating Tier 1 catalysts (SEC/Benzinga/DarkPool) from general news noise.
    """
    if enriched_articles is None or enriched_articles.empty:
        return {
            "weighted_sentiment_score": 0.0,
            "raw_mean_sentiment": 0.0,
            "sentiment_dispersion": 0.0,
            "tier1_catalyst_count": 0,
            "tier1_headlines": [],
            "top_label": "neutral",
            "confidence": 0.5,
        }

    scores = enriched_articles.get("sentiment_score", pd.Series([0.0]))
    weights = enriched_articles.get("effective_weight", pd.Series([1.0]))

    total_weight = float(weights.sum())
    if total_weight > 0:
        weighted_score = float((scores * weights).sum() / total_weight)
    else:
        weighted_score = float(scores.mean())

    raw_mean = float(scores.mean())
    dispersion = float(scores.std()) if len(scores) > 1 else 0.0

    # Extract Tier 1 Catalysts
    t1_mask = enriched_articles.get("source_weight", pd.Series([1.0])) >= 2.5
    t1_articles = enriched_articles[t1_mask] if t1_mask.any() else pd.DataFrame()
    t1_headlines = (
        t1_articles["Title"].tolist() if "Title" in t1_articles.columns else []
    )

    top_label = (
        "positive"
        if weighted_score > 0.15
        else ("negative" if weighted_score < -0.15 else "neutral")
    )

    return {
        "weighted_sentiment_score": round(weighted_score, 4),
        "raw_mean_sentiment": round(raw_mean, 4),
        "sentiment_dispersion": round(dispersion, 4),
        "tier1_catalyst_count": len(t1_headlines),
        "tier1_headlines": t1_headlines[:5],
        "top_label": top_label,
        "confidence": round(
            float(
                enriched_articles.get("sentiment_confidence", pd.Series([0.5])).mean()
            ),
            4,
        ),
    }


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
