from torch.nn import Module, ModuleList
from torch.nn.functional import relu
from modeling.modules import PretrainedEmbeddingLayer, MLP, CellLayer, SequentialModel, AttendedState
import torch
from modeling.activation_factory import ActivationFactory


class MultiAttentionRNN(Module):
    def __init__(self, embeddings,
                 embeddings_dropout,
                 is_gru,
                 cell_hidden_size,
                 stacked_layers,
                 bidirectional,
                 att_mlp_layers,
                 att_mlp_dropout,
                 top_mlp_layers,
                 top_mlp_activation,
                 top_mlp_outer_activation,
                 targets,
                 top_mlp_dropout):
        super(MultiAttentionRNN, self).__init__()
        self.name = "MultiAttentionRNN"

        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings, dropout=embeddings_dropout, trainable=False)

        self.cell = CellLayer(is_gru, self.word_embedding_layer.get_output_size(),
                              cell_hidden_size, bidirectional, stacked_layers)
        large_size = cell_hidden_size * 2 if bidirectional else cell_hidden_size
        decision_layers = [MLP(num_of_layers=top_mlp_layers,
                               init_size=large_size,
                               out_size=1,
                               dropout=top_mlp_dropout,
                               inner_activation=ActivationFactory.get_activation(top_mlp_activation),
                               outer_activation=ActivationFactory.get_activation(top_mlp_outer_activation)) for _ in range(targets)]
        self.decision_layers = ModuleList(decision_layers)
        self.attentions = ModuleList(
            [AttendedState(att_mlp_layers, large_size, att_mlp_dropout, relu) for _ in range(targets)])
        self.seq = SequentialModel([self.word_embedding_layer, self.cell])
        self.trainable_weights = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x):
        encoder = self.seq(x)
        states = [decision(attention(encoder)) for attention, decision in zip(self.attentions, self.decision_layers)]
        out = torch.cat(states, dim=1)
        return out
