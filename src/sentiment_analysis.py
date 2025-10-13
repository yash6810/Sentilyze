import os
import pandas as pd
from typing import Any

os.environ["TRANSFORMERS_BACKEND"] = "pytorch"


def get_sentiment(articles: pd.DataFrame, sentiment_analyzer: Any) -> pd.DataFrame:
    """
    Analyzes the sentiment of news articles.

    Args:
        articles (pd.DataFrame): A DataFrame containing news articles with 'title' and 'description' columns.
        sentiment_analyzer (Any): A sentiment analysis pipeline object.

    Returns:
        pd.DataFrame: The input DataFrame with added 'sentiment_label' and 'sentiment_score' columns.
    """
    articles = articles.dropna(subset=["title", "description"]).copy()
    articles["text"] = articles["title"] + ". " + articles["description"]

    # Process all articles
    sentiments = sentiment_analyzer(list(articles["text"]))

    articles[["sentiment_label", "sentiment_score"]] = [
        [s["label"], s["score"]] for s in sentiments
    ]

    return articles
