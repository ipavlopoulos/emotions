from nltk.tokenize import TweetTokenizer
from tqdm import tqdm
import numpy as np
import torch
from time import time
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report
from torch.autograd import Variable
import yaml

tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)


def tokenize(tweet):
    return tweet_tokenizer.tokenize(tweet)


def get_embeddings(path):
    embeddings_dict = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in tqdm(f):
            values = line.strip().split(" ")
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_dict[values[0]] = coefs
    return embeddings_dict


def create_freq_vocabulary(tokenized_texts):
    token_dict = {}
    for text in tokenized_texts:
        for token in text:
            try:
                token_dict[token] += 1
            except KeyError:
                token_dict[token] = 1
    return token_dict


def get_top_freq_words(token_dict, min_freq):
    return [x for x in token_dict if token_dict[x] >= min_freq]


def get_unique_tokens(tokenized_texts, min_freq):
    voc = create_freq_vocabulary(tokenized_texts)
    print("tokens found in training data set:", len(voc))
    freq_words = get_top_freq_words(voc, min_freq)
    print("tokens with frequency >= %d: %d" % (min_freq, len(freq_words)))
    return freq_words


def create_final_dictionary(freq_words, embeddings_dict, unk_token, pad_token):
    words = list(set(freq_words).intersection(embeddings_dict.keys()))
    print("embedded tokens: %d" % (len(words)))
    words = [pad_token, unk_token] + words
    return {w: i for i, w in enumerate(words)}


def get_embeddings_matrix(word_dict, embeddings_dict, size):
    embs = np.zeros(shape=(len(word_dict), size))
    for word in tqdm(word_dict):
        try:
            embs[word_dict[word]] = embeddings_dict[word]
        except KeyError:
            print('no embedding for: ', word)
    embs[1] = np.mean(embs[2:])
    return embs


def get_indexed_value(w2i, word, unk_token):
    try:
        return w2i[word]
    except KeyError:
        return w2i[unk_token]


def get_indexed_text(w2i, words, unk_token):
    return [get_indexed_value(w2i, word, unk_token) for word in words]


def pad_text(tokenized_text, maxlen, pad_tkn):
    if len(tokenized_text) < maxlen:
        return [pad_tkn] * (maxlen - len(tokenized_text)) + tokenized_text
    else:
        return tokenized_text[len(tokenized_text) - maxlen:]


def create_batches(df, batch_size, w2i, pad_token, unk_token, target_list):
    batches = []
    offset = 0
    while offset < len(df):
        upper_limit = min(len(df), offset + batch_size)
        batch_df = df.iloc[offset: upper_limit]
        maxlen = batch_df['lengths'].values[-1]

        batch_df['x'] = batch_df['tokens'].map(
            lambda x: get_indexed_text(w2i, pad_text(x, maxlen, pad_token), unk_token))
        batches.append({'x': np.array([x for x in batch_df['x']], dtype=np.int32),
                        'y': np.array(batch_df[target_list], dtype=np.float32)})
        offset = upper_limit
    return batches


MODELS_DIR = ""


def save_model(model):
    torch.save(model.state_dict(), MODELS_DIR + model.name + '.pkl')


def load_model(model):
    model.load_state_dict(torch.load(MODELS_DIR + model.name + '.pkl'))
    return model


def train(model, train_batches, test_batches, optimizer, criterion,
          epochs, init_patience, target_list, cuda=True):
    patience = init_patience
    best_auc = 0.0
    for i in range(1, epochs + 1):
        start = time()
        auc = run_epoch(model, train_batches, test_batches, optimizer, criterion, target_list,
                        cuda)
        end = time()
        print('epoch %d, auc: %2.3f  Time: %d minutes, %d seconds'
              % (i, 100 * auc, (end - start) / 60, (end - start) % 60))
        if best_auc < auc:
            best_auc = auc
            patience = init_patience
            save_model(model)
            if i > 1:
                print('best epoch so far')
        else:
            patience -= 1
        if patience == 0:
            break
    return best_auc


def run_epoch(model, train_batches, test_batches, optimizer, criterion, target_list,  cuda):
    model.train(True)
    perm = np.random.permutation(len(train_batches))
    for i in tqdm(perm):
        batch = train_batches[i]
        inner_perm = np.random.permutation(len(batch['x']))
        data = []
        if cuda:
            data.append(Variable(torch.from_numpy(batch['x'][inner_perm]).long().cuda()))
        else:
            data.append(Variable(torch.from_numpy(batch['x'][inner_perm]).long()))
        if cuda:
            y = Variable(torch.from_numpy(batch['y'][inner_perm]).cuda())
        else:
            y = Variable(torch.from_numpy(batch['y'][inner_perm]))
        outputs = model(*data)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return evaluate(model, test_batches, target_list)


def evaluate(model, test_batches, y_list):
    model.train(False)
    results = get_scores(model, test_batches, y_list)
    auc_scores = []
    for k in results:
        auc = roc_auc_score(results[k]['labels'], np.asarray(results[k]['scores'], dtype='float32'))
        #         print("{} - auc:{}".format(y_list[k], auc))
        auc_scores.append(auc)
    return np.mean(auc_scores)


def get_scores(model, test_batches, y_list):
    results = {y: {'scores': [], 'labels': []} for y in range(len(y_list))}
    for batch in test_batches:
        batch_scores = model(torch.from_numpy(batch['x']).long())
        for i in range(len(y_list)):
            results[i]['scores'].extend(batch_scores[:, i].detach().numpy())
            results[i]['labels'].extend(batch['y'][:, i])
    return results


def best_thr(labels, scores):
    best_thr = 0.05
    best_f1 = 0.0
    for thr in np.arange(0.01, 0.99, 0.01):
        scr = f1_score(labels, [x > thr for x in scores])
        if scr > best_f1:
            best_f1 = scr
            best_thr = thr
    return best_thr


def full_classification_report(model, test_batches, y_list):
    model.train(False)
    results = get_scores(model, test_batches, y_list)
    print("\tEmotion\tAUC\tAccuracy")
    best_thresholds = {}
    for k, emotion in enumerate(y_list):
        best_thresholds[emotion] = best_thr(results[k]['labels'], np.asarray(results[k]['scores']))
        auc = roc_auc_score(results[k]['labels'], np.asarray(results[k]['scores'], dtype='float32'))

        acc = accuracy_score(results[k]['labels'],
                             np.asarray([x > best_thresholds[emotion] for x in results[k]['scores']], dtype='float32'))
        print("\t{:.5s}\t{:.4f}\t{:.4f}".format(emotion, auc, acc))
    full_predictions = np.zeros(shape=(len(results[0]['scores']), len(y_list)))
    full_targets = np.zeros(shape=(len(results[0]['scores']), len(y_list)))
    for i in range(len(y_list)):
        full_predictions[:, i] = [x > best_thresholds[y_list[i]] for x in results[i]['scores']]
        full_targets[:, i] = results[i]['labels']
    print(classification_report(full_targets, full_predictions, target_names=y_list))


def load_yaml(path):
    with open(path, 'r') as f:
        config = yaml.load(f)
    return config