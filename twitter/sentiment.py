# https://github.com/cjhutto/vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def analyse(tweet):
    return analyzer.polarity_scores(tweet)
