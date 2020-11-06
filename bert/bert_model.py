import torch
from transformers import RobertaModel


class BERTClass(torch.nn.Module):
    def __init__(self, num_of_cols, path='roberta-large', do=0.3):
        super(BERTClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained(path)
        self.l2 = torch.nn.Dropout(do)
        self.l3 = torch.nn.Linear(1024, num_of_cols)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

