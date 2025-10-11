import os
os.environ["TRANSFORMERS_BACKEND"] = "pytorch"

def get_sentiment(articles, sentiment_analyzer):
    articles = articles.dropna(subset=['title', 'description']).copy()
    articles['text'] = articles['title'] + '. ' + articles['description']
    
    # Process all articles
    sentiments = sentiment_analyzer(list(articles['text']))
    
    articles[['sentiment_label', 'sentiment_score']] = [[s['label'], s['score']] for s in sentiments]
    
    return articles
