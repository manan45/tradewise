from transformers import pipeline

def analyze_sentiment(news_articles):
    """
    Analyze sentiment of news articles using a pre-trained language model.

    :param news_articles: List of news articles.
    :return: List of sentiment scores.
    """
    sentiment_pipeline = pipeline("sentiment-analysis")
    sentiments = [sentiment_pipeline(article)[0] for article in news_articles]
    return sentiments
