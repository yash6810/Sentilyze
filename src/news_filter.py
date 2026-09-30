"""
Institutional News Filter, Semantic Deduplicator & Catalyst Classifier for Sentilyze.
====================================================================================
Filters noise, eliminates duplicate syndication spam, scores entity relevance,
and assigns quantitative source authority weights:
- Tier 1 (3.5x - 3.0x): SEC Form 8-K Filings, Benzinga Analyst Ratings & Dark Pool Relays
- Tier 2 (1.8x): Reddit 8-Station Premarket Intelligence & Options Flow
- Tier 3 (1.0x): Standard Google News RSS & Aggregator Wire Feeds
"""

import re
import html
import unicodedata
from typing import Dict, Any, List, Set, Tuple
import pandas as pd
from datetime import datetime, timezone

from src.utils import get_logger

logger = get_logger("news_filter")

# Source Authority Weight Mapping
SOURCE_WEIGHT_MAP = {
    "sec edgar": 3.5,
    "sec 8-k": 3.5,
    "benzinga": 3.0,
    "darkpooldiver": 2.8,
    "marianarelay": 2.8,
    "ephemeral_key": 2.8,
    "_0xwhale": 2.8,
    "reddit": 1.8,
    "wallstreetbets": 1.8,
    "unusual_whales": 1.8,
    "options": 1.8,
    "bloomberg": 2.2,
    "reuters": 2.2,
    "dow jones": 2.2,
    "wsj": 2.0,
    "financial times": 2.0,
    "google news": 1.0,
    "yahoo finance": 1.1,
}

# High-impact corporate catalyst keywords
TIER1_CATALYST_KEYWORDS = {
    "earnings",
    "results",
    "beat",
    "miss",
    "guidance",
    "fda approval",
    "phase 3",
    "clinical trial",
    "sec 8-k",
    "acquisition",
    "merger",
    "takeover",
    "restructuring",
    "investigation",
    "subpoena",
    "delisting",
    "ceo",
    "cfo",
    "resigns",
    "appointed",
    "contract",
    "partnership",
    "patent",
    "buyout",
    "upgrade",
    "downgrade",
    "target price",
    "rating",
}


def normalize_headline_tokens(text: str) -> Set[str]:
    """Extracts clean alphanumeric tokens for fast semantic similarity."""
    if not isinstance(text, str):
        return set()
    cleaned = html.unescape(text).lower()
    cleaned = re.sub(r"\[.*?\]", " ", cleaned)  # remove tags like [Benzinga]
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    tokens = {t for t in cleaned.split() if len(t) > 2 and not t.isdigit()}
    return tokens


def calculate_jaccard_similarity(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    """Calculates Jaccard token overlap between two headlines."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    return float(intersection / max(1, union))


def assign_source_weight(title: str, source_name: str) -> Tuple[float, str]:
    """
    Assigns institutional source weight and priority tier based on publisher authority.
    """
    t_lower = (title or "").lower()
    s_lower = (source_name or "").lower()

    # 1. Check for SEC 8-K
    if "[sec 8-k]" in t_lower or "sec edgar" in s_lower:
        return 3.5, "TIER1_REGULATORY"

    # 2. Check for Benzinga Analyst Wire
    if "[benzinga]" in t_lower or "benzinga" in s_lower:
        return 3.0, "TIER1_CATALYST_WIRE"

    # 3. Check for Dark Pool & Whale relays
    for dp_key in ["darkpooldiver", "marianarelay", "ephemeral_key", "_0xwhale"]:
        if dp_key in t_lower or dp_key in s_lower:
            return 2.8, "TIER1_DARK_POOL"

    # 4. Check for Reddit 8-Station Social Intel
    if "[reddit" in t_lower or "reddit" in s_lower or "wallstreetbets" in s_lower:
        return 1.8, "TIER2_SOCIAL_RADAR"

    # 5. Check other sources
    for k, w in SOURCE_WEIGHT_MAP.items():
        if k in s_lower:
            return w, "TIER2_NEWS_WIRE" if w >= 2.0 else "TIER3_GENERAL"

    return 1.0, "TIER3_GENERAL"


def deduplicate_news_stream(
    df: pd.DataFrame, sim_threshold: float = 0.65
) -> pd.DataFrame:
    """
    Fast Jaccard-based semantic deduplication that eliminates syndicated press releases
    and repeated wire stories, keeping the version with the highest source authority.
    """
    if df is None or df.empty or "Title" not in df.columns:
        return df if df is not None else pd.DataFrame()

    records = df.copy()
    if "source" in records.columns:
        records["source_str"] = records["source"].apply(
            lambda x: x.get("name", "") if isinstance(x, dict) else str(x)
        )
    else:
        records["source_str"] = ""

    # Compute source weights
    weights_tiers = [
        assign_source_weight(str(r["Title"]), str(r["source_str"]))
        for _, r in records.iterrows()
    ]
    records["source_weight"] = [wt[0] for wt in weights_tiers]
    records["source_tier"] = [wt[1] for wt in weights_tiers]

    # Pre-tokenize all titles
    token_list = [normalize_headline_tokens(str(t)) for t in records["Title"]]
    records["_tokens"] = token_list

    # Sort descending by source_weight and recency
    sort_cols = ["source_weight"]
    if "publishedAt" in records.columns:
        records["_pub"] = pd.to_datetime(
            records["publishedAt"], utc=True, errors="coerce"
        )
        sort_cols = ["source_weight", "_pub"]

    records = records.sort_values(by=sort_cols, ascending=False).reset_index(drop=True)

    kept_indices = []
    seen_token_sets: List[Set[str]] = []

    for idx, row in records.iterrows():
        tokens = row["_tokens"]
        if not tokens:
            kept_indices.append(idx)
            continue

        is_duplicate = False
        for prev_tokens in seen_token_sets:
            similarity = calculate_jaccard_similarity(tokens, prev_tokens)
            if similarity >= sim_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept_indices.append(idx)
            seen_token_sets.append(tokens)

    deduped_df = records.iloc[kept_indices].copy()
    drop_temp = [
        c for c in ["_tokens", "_pub", "source_str"] if c in deduped_df.columns
    ]
    deduped_df.drop(columns=drop_temp, inplace=True)

    logger.info(
        f"⚡ [NEWS FILTER] Semantic deduplication reduced {len(df)} headlines to {len(deduped_df)} unique catalysts."
    )
    return deduped_df
