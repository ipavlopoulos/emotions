import torch


EMOTIONS = ('anger', 'anticipation',
            'disgust', 'fear', 'joy',
            'love', 'optimism', 'pessimism',
            'sadness', 'surprise', 'trust',
            'positive', 'negative')
emotion_dict = {w: i for i, w in enumerate(EMOTIONS)}
model_path = "pytorch_models/roberta.mdl"


def load_torch_model(path):
    return torch.load(path)


def process_tweet(tweet_text, tokenizer, max_len):
    tweet_text = " ".join(tweet_text.split())

    inputs = tokenizer.encode_plus(
        tweet_text,
        None,
        add_special_tokens=True,
        max_length=max_len,
        pad_to_max_length=True,
        return_token_type_ids=True,
        truncation=True)
    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    token_type_ids = inputs["token_type_ids"]
    return {'ids': ids, 'mask': mask, 'token_type_ids': token_type_ids}


def score(model, processed_tweet, emotions=EMOTIONS):
    scores = model(processed_tweet['ids'].unsqueeze(0),
                   processed_tweet['mask'].unsqueeze(0),
                   processed_tweet['token_type_ids'].unsqueeze(0))[0]
    scores = scores.cpu().numpy().detach()
    return {em: sc for em, sc in zip(emotions, scores)}



