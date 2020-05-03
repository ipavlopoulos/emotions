from torch.nn import Module, ModuleList
from modeling.modules import PretrainedEmbeddingLayer, MLP, CellLayer, SequentialModel, LastState
import torch
from modeling.activation_factory import ActivationFactory


class LastStateRNN(Module):
    def __init__(self, embeddings,
                 embeddings_dropout=0.0,
                 is_gru=True,
                 cell_hidden_size=128,
                 stacked_layers=1,
                 bidirectional=False,
                 top_mlp_layers=2,
                 top_mlp_activation="relu",
                 top_mlp_outer_activation=None, targets=11,
                 top_mlp_dropout=0.0):
        super(LastStateRNN, self).__init__()
        self.name = "LastStateRNN"

        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings, dropout=embeddings_dropout, trainable=False)

        self.cell = CellLayer(is_gru, self.word_embedding_layer.get_output_size(),
                              cell_hidden_size, bidirectional, stacked_layers)
        self.decision_layer = MLP(num_of_layers=top_mlp_layers,
                                  init_size=self.cell.get_output_size(),
                                  out_size=targets,
                                  dropout=top_mlp_dropout,
                                  inner_activation=ActivationFactory.get_activation(top_mlp_activation),
                                  outer_activation=ActivationFactory.get_activation(top_mlp_outer_activation))
        self.state = LastState(self.cell.get_output_size())
        self.seq = SequentialModel([self.word_embedding_layer, self.cell, self.state, self.decision_layer])
        self.trainable_weights = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x):
        out = self.seq(x)
        return out
