import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, AutoTokenizer
from tqdm import tqdm
from torch import cuda
from bert import BERTClass,TinyBert, model_path
import click
import datetime

cli = click.Group()

device = 'cuda' if cuda.is_available() else 'cpu'


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment_text
        self.targets = self.data.targets
        self.max_len = max_len
        self.dummy_id  = dataframe.dummy_id

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


@click.option('--input_dir')
@click.option('--result_dir')
@click.option('--debug',  default=False)
@click.option('--only_located',  default=True)
def run(input_dir, result_dir, debug, only_located):
    datetime.date.today() - datetime.timedelta(1)
    label_cols = ['anger', 'anticipation', 'disgust',
                  'fear', 'joy', 'love',
                  'optimism', 'pessimism', 'sadness',
                  'surprise', 'trust']
    df = pd.read_csv(input_dir)
    if only_located:
        df = df.dropna(subset=['full_name']).reset_index()
    df['dummy_id'] = range(len(df))
    bert_path =  '' if debug else 'roberta-large'
    max_len = 50
    valid_batch_size = 8
    if debug:
        tokenizer = AutoTokenizer.from_pretrained(bert_path)
    else:
        tokenizer = RobertaTokenizer.from_pretrained(bert_path)

    df_test = prepare_df(df, 'full_text', label_cols=label_cols)

    testing_set = CustomDataset(df_test, tokenizer, max_len)

    test_params = {'batch_size': valid_batch_size,
                   'shuffle': False,
                   'num_workers': 0
                   }

    testing_loader = DataLoader(testing_set, **test_params)

    if debug:
        model = BERTClass(num_of_cols=len(label_cols))
        model.to(device)
        print("model loaded.")
        model.load_state_dict(torch.load(model_path))
    else:
        model = TinyBert(num_of_cols=len(label_cols))
    results = score_data(model, testing_loader, label_cols=label_cols)

    df2 = pd.DataFrame.from_dict(results)

    result = df.merge(df2, on='dummy_id')

    result = result.drop(columns=['dummy_id'])

    result.to_csv(result_dir, index=False)







