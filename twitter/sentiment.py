# https://github.com/cjhutto/vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def analyse(tweet):
    return analyzer.polarity_scores(tweet)


def analyse_per_language(tweet, lan):
    if lan == 'en':
        return analyse(tweet)
    else:
        return {'compound': 0.5}

