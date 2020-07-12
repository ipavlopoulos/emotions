# https://github.com/cjhutto/vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import RobertaTokenizer
import torch
from bert import score, process_tweet,  model_path, EMOTIONS, BERTClass

tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
analyzer = SentimentIntensityAnalyzer()

model = BERTClass(num_of_cols=len(EMOTIONS))
model.load_state_dict(torch.load(model_path))


def analyse(tweet):
    return analyzer.polarity_scores(tweet)


def analyse_with_roberta(tweet):
    processed_tweet = process_tweet(tweet_text=tweet, tokenizer=tokenizer, max_len=100)
    scores = score(model, processed_tweet)
    return scores


def analyse_per_language(tweet, lan):
    """
    returns the sentiment of a tweet.
    If there is not a trained classifier return o neutral value (0.5)
    :param tweet:
    :param lan: 'en' for english 'el' (or any) for greek
    :return:
    """
    if lan == 'en':
        return analyse_with_roberta(tweet)
    else:
        return {x: 0.5 for x in EMOTIONS}

