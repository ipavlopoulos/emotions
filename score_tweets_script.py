import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, AutoTokenizer
from tqdm import tqdm
from torch import cuda
from bert import BERTClass,TinyBert, model_path
import click
import datetime
import os
import time


cli = click.Group()

device = 'cuda' if cuda.is_available() else 'cpu'


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment_text
        self.targets = self.data.targets
        self.max_len = max_len
        self.dummy_id = dataframe.dummy_id

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True, truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float),
            'dummy_id': torch.tensor(self.dummy_id[index], dtype=torch.int32)
        }


def prepare_df(df, text_col, label_cols=None):
    df.comment_text = df[text_col]
    df.targets = df[label_cols].values.tolist()
    df.dummy_id = df['dummy_id']
    return df


def score_data(model, data_loader, label_cols):
    model.eval()
    results = {x: [] for x in label_cols}
    results['dummy_id'] = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids)
            outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()

            dummy_id = data['dummy_id'].to(device, dtype = torch.float)
            results['dummy_id'].extend([int(x) for x in dummy_id.cpu().detach().numpy().tolist()])
            for i, lbl in enumerate(label_cols):
                results[lbl].extend([x[i] for x in outputs])
    return results


def run(input_dir, result_dir, debug, only_located, batch_size):
    previous_date = str(datetime.date.today() - datetime.timedelta(1))
    print(previous_date)
    label_cols = ['anger', 'anticipation', 'disgust',
                  'fear', 'joy', 'love',
                  'optimism', 'pessimism', 'sadness',
                  'surprise', 'trust']
    df = pd.read_csv(os.path.join(input_dir, previous_date+".csv"))
    if only_located:
        df = df.dropna(subset=['full_name']).reset_index()
    df['dummy_id'] = range(len(df))

    for col in label_cols:
        df[col] = 0.0
    bert_path = "prajjwal1/bert-tiny" if debug else 'roberta-large'
    max_len = 50
    if debug:
        tokenizer = AutoTokenizer.from_pretrained(bert_path)
    else:
        tokenizer = RobertaTokenizer.from_pretrained(bert_path)

    df_test = prepare_df(df, 'text', label_cols=label_cols)

    df = df.drop(columns=label_cols)

    testing_set = CustomDataset(df_test, tokenizer, max_len)

    test_params = {'batch_size': batch_size,
                   'shuffle': False,
                   'num_workers': 0
                   }

    testing_loader = DataLoader(testing_set, **test_params)

    if debug:
        model = TinyBert(num_of_cols=len(label_cols))

    else:
        model = BERTClass(num_of_cols=len(label_cols))
        model.to(device)
        print("model loaded.")
        model.load_state_dict(torch.load(model_path))
    results = score_data(model, testing_loader, label_cols=label_cols)

    df2 = pd.DataFrame.from_dict(results)

    result = df.merge(df2, on='dummy_id')

    result = result.drop(columns=['dummy_id'])

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result.to_csv(os.path.join(result_dir, previous_date), index=False)


@cli.command()
@click.option('--input_dir', default="../tweets/en/")
@click.option('--result_dir', default="../tweets/scored_en")
@click.option('--debug',  default=True)
@click.option('--only_located',  default=True)
@click.option('--batch_size',  default=8)
def loop(input_dir, result_dir, debug, only_located, batch_size):
    while True:
        current_time = time.time()
        run(input_dir, result_dir, debug, only_located, batch_size)
        duration = time.time() - current_time
        time.sleep(24 * 60 * 60 - duration)


if __name__ == "__main__":
    loop()




