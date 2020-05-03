from torch.nn import Module
from torch.nn.functional import relu
from modeling.modules import PretrainedEmbeddingLayer, MLP, CellLayer, SequentialModel, AttendedState
from modeling.activation_factory import ActivationFactory


class SingleAttentionRNN(Module):
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
        super(SingleAttentionRNN, self).__init__()
        self.name = "SingleAttentionRNN"

        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings, dropout=embeddings_dropout, trainable=False)

        self.cell = CellLayer(is_gru, self.word_embedding_layer.get_output_size(),
                              cell_hidden_size, bidirectional, stacked_layers)
        self.decision_layer = MLP(num_of_layers=top_mlp_layers,
                                  init_size=self.cell.get_output_size(),
                                  out_size=targets,
                                  dropout=top_mlp_dropout,
                                  inner_activation=ActivationFactory.get_activation(top_mlp_activation),
                                  outer_activation=ActivationFactory.get_activation(top_mlp_outer_activation))
        self.attention = AttendedState(att_mlp_layers, self.cell.get_output_size(), att_mlp_dropout, relu)
        self.seq = SequentialModel([self.word_embedding_layer, self.cell])
        self.trainable_weights = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x):
        encoder = self.seq(x)
        out = self.decision_layer(self.attention(encoder))
        return out
