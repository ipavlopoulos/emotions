from torch.nn import Module
from modeling.modules import PretrainedEmbeddingLayer, MLP, AvgPoolingState, SequentialModel
from modeling.activation_factory import ActivationFactory


class MLPModel(Module):
    def __init__(self, embeddings,
                 embeddings_dropout,
                 top_mlp_layers,
                 top_mlp_activation,
                 top_mlp_outer_activation,
                 targets,
                 top_mlp_dropout):

        super(MLPModel, self).__init__()
        self.name = "MLPModel"

        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings,
                                                             dropout=embeddings_dropout,
                                                             trainable=False)
        fs_size = self.word_embedding_layer.get_output_size()
        self.decision_layer = MLP(num_of_layers=top_mlp_layers,
                                  init_size=fs_size,
                                  out_size=targets,
                                  dropout=top_mlp_dropout,
                                  inner_activation=ActivationFactory.get_activation(top_mlp_activation),
                                  outer_activation=ActivationFactory.get_activation(top_mlp_outer_activation))
        self.state = AvgPoolingState(input_size=fs_size)
        self.seq = SequentialModel([self.word_embedding_layer, self.state, self.decision_layer])
        self.trainable_weights = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x):
        out = self.seq(x)
        return out
