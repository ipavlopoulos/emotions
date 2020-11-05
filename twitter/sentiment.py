# https://github.com/cjhutto/vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import RobertaTokenizer, AutoTokenizer
import torch
from bert import score, process_tweet,  model_path, EMOTIONS, BERTClass, TinyBert
from torch import cuda

DEBUG = True
device = 'cuda' if cuda.is_available() else 'cpu'

if DEBUG:
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    model = TinyBert(num_of_cols=len(EMOTIONS)).to(device)
else:
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    model = BERTClass(num_of_cols=len(EMOTIONS)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

analyzer = SentimentIntensityAnalyzer()


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

