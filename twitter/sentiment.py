# https://github.com/cjhutto/vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def analyse(tweet):
    return analyzer.polarity_scores(tweet)


def analyse_per_language(tweet, lan):
    """
    returns the sentiment of a tweet.
    If there is not a trained classifier return o neutral value (0.5)
    :param tweet:
    :param lan:
    :return:
    """
    if lan == 'en':
        return analyse(tweet)
    else:
        return {'compound': 0.5}

