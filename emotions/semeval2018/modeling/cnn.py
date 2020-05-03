from torch.nn import Module, ModuleList
from modeling.modules import PretrainedEmbeddingLayer, MLP, ConvBlock
import torch
from modeling.activation_factory import ActivationFactory


class CNN(Module):
    def __init__(self, embeddings,
                 embeddings_dropout,
                 filters,
                 min_kernel,
                 max_kernel,
                 top_mlp_layers,
                 top_mlp_activation,
                 top_mlp_outer_activation, targets,
                 top_mlp_dropout):
        super(CNN, self).__init__()
        self.name = "CNN"

        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings,
                                                             dropout=embeddings_dropout,
                                                             trainable=False)

        conv_blocks = [ConvBlock(in_channels=self.word_embedding_layer.get_output_size(),
                                 filters=filters, window=i) for i in range(min_kernel, max_kernel+1)]

        self.conv_blocks = ModuleList(conv_blocks)

        concat_size = len(conv_blocks) * filters
        self.decision_layer = MLP(num_of_layers=top_mlp_layers,
                                  init_size=concat_size,
                                  out_size=targets,
                                  dropout=top_mlp_dropout,
                                  inner_activation=ActivationFactory.get_activation(top_mlp_activation),
                                  outer_activation=ActivationFactory.get_activation(top_mlp_outer_activation))
        self.trainable_weights = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x):
        embeddings = self.word_embedding_layer(x)
        n_gram_representations = [block(embeddings) for block in self.conv_blocks]
        encoder = torch.cat(n_gram_representations, dim=1)
        out = self.decision_layer(encoder)
        return out
