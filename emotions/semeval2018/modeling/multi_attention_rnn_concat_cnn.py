from torch.nn import Module, ModuleList
from torch.nn.functional import relu
from modeling.modules import PretrainedEmbeddingLayer, MLP, CellLayer, AttendedState, ConvBlock
import torch
from modeling.activation_factory import ActivationFactory


class MultiAttentionRNNConcatCNN(Module):
    def __init__(self, embeddings,
                 embeddings_dropout,
                 is_gru,
                 filters,
                 min_kernel,
                 max_kernel,
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
        super(MultiAttentionRNNConcatCNN, self).__init__()
        self.name = "MultiAttentionRNNConcatCNN"
        self.targets = targets

        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings, dropout=embeddings_dropout, trainable=False)
        conv_blocks = [ConvBlock(in_channels=self.word_embedding_layer.get_output_size(),
                                 filters=filters, window=i) for i in range(min_kernel, max_kernel+1)]
        self.conv_blocks = ModuleList(conv_blocks)
        self.cell = CellLayer(is_gru, self.word_embedding_layer.get_output_size(),
                              cell_hidden_size, bidirectional, stacked_layers)
        concat_size = self.cell.get_output_size() + filters * len(conv_blocks)
        decision_layers = [MLP(num_of_layers=top_mlp_layers,
                               init_size=concat_size,
                               out_size=1,
                               dropout=top_mlp_dropout,
                               inner_activation=ActivationFactory.get_activation(top_mlp_activation),
                               outer_activation=ActivationFactory.get_activation(top_mlp_outer_activation)) for _ in range(targets)]
        self.decision_layers = ModuleList(decision_layers)
        attentions = [AttendedState(att_mlp_layers, self.cell.get_output_size(), att_mlp_dropout, relu)
                      for _ in range(targets)]
        self.attentions = ModuleList(attentions)
        self.trainable_weights = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x):
        embeddings = self.word_embedding_layer(x)
        n_grams = [block(embeddings) for block in self.conv_blocks]
        n_grams_concat = torch.cat(n_grams, dim=1)
        sequence_encoder = self.cell(embeddings)
        states = [attention(sequence_encoder) for attention in self.attentions]
        states = [torch.cat((state, n_grams_concat), dim=1) for state in states]
        decisions = [decision(state) for decision, state in zip(self.decision_layers, states)]
        out = torch.cat(decisions, dim=1)
        return out
