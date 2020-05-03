from torch.nn import Module, ModuleList
from torch.nn.functional import relu
from modeling.modules import PretrainedEmbeddingLayer, MLP, AttendedState
import torch
from modeling.activation_factory import ActivationFactory


class MLPWithAttentionModel(Module):
    def __init__(self, embeddings,
                 embeddings_dropout,
                 att_mlp_layers,
                 att_mlp_dropout,
                 top_mlp_layers,
                 top_mlp_activation,
                 top_mlp_outer_activation,
                 targets,
                 top_mlp_dropout):

        super(MLPWithAttentionModel, self).__init__()
        self.name = "MLPWithAttentionModel"
        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings,
                                                             dropout=embeddings_dropout,
                                                             trainable=False)
        fs_size = self.word_embedding_layer.get_output_size()
        decision_layers = [MLP(num_of_layers=top_mlp_layers,
                               init_size=fs_size,
                               out_size=1,
                               dropout=top_mlp_dropout,
                               inner_activation=ActivationFactory.get_activation(top_mlp_activation),
                               outer_activation=ActivationFactory.get_activation(top_mlp_outer_activation))
                           for _ in range(targets)]
        self.decision_layers = ModuleList(decision_layers)
        attentions = [AttendedState(att_mlp_layers, fs_size, att_mlp_dropout, relu) for _ in range(targets)]
        self.attentions = ModuleList(attentions)
        self.trainable_weights = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x):
        encoder = self.word_embedding_layer(x)
        states = [decision(attention(encoder)) for attention, decision in zip(self.attentions, self.decision_layers)]
        out = torch.cat(states, dim=1)
        return out
