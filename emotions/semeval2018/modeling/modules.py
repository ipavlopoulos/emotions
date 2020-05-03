from torch.nn import Module, Dropout2d, Embedding, Conv1d, LSTM, GRU, Dropout, Linear, ModuleList
from torch.nn.functional import softmax
import torch


class SpatialDropout(Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class Layer(Module):
    def __init__(self):
        super(Layer, self).__init__()

    def get_output_size(self):
        raise NotImplementedError

    def get_input_size(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class PretrainedEmbeddingLayer(Layer):
    def __init__(self, embeddings, dropout=0.0, trainable=True):
        """
        :param embeddings: a numpy array with the embeddings
        :param trainable: if false the embeddings will be frozen
        """
        super(PretrainedEmbeddingLayer, self).__init__()
        self._input_size = embeddings.shape[0]
        self._output_size = embeddings.shape[1]
        self.dropout = SpatialDropout(dropout)
        self.embed = Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embed.weight.data.copy_(torch.from_numpy(embeddings))
        if not trainable:
            self.embed.weight.requires_grad = False

    def forward(self, x):
        return self.dropout(self.embed(x))

    def get_output_size(self):
        return self._output_size

    def get_input_size(self):
        return self._input_size


class ConvBlock(Layer):
    def __init__(self, in_channels, filters, window=2):
        """
        :param in_channels: the input tensor's size
        :param filters: the output tensor's size
        :param window: the n-gram size
        """
        super(ConvBlock, self).__init__()
        self._input_size = in_channels
        self._output_size = filters
        self.conv = Conv1d(in_channels=in_channels, out_channels=filters, kernel_size=window, padding=(window // 2))

    def forward(self, x):
        return torch.max(self.conv(x.permute(0, 2, 1)), 2)[0]

    def get_output_size(self):
        return self._output_size

    def get_input_size(self):
        return self._input_size


class CellLayer(Layer):
    def __init__(self, is_gru, input_size, hidden_size, bidirectional, stacked_layers):
        """
        :param is_gru: GRU cell type if true, otherwise LSTM
        :param input_size: the size of the tensors that will be used as input (embeddings or projected embeddings)
        :param hidden_size: the size of the cell
        :param bidirectional: boolean
        :param stacked_layers: the number of stacked layers
        """
        super(CellLayer, self).__init__()
        if is_gru:
            self.cell = GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True,
                            bidirectional=bidirectional, num_layers=stacked_layers)

        else:
            self.cell = LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True,
                             bidirectional=bidirectional, num_layers=stacked_layers)

        self._output_size = hidden_size * 2 if bidirectional else hidden_size
        self._input_size = input_size

    def forward(self, x):
        return self.cell(x)[0]

    def get_output_size(self):
        return self._output_size

    def get_input_size(self):
        return self._input_size


class MLP(Layer):
    def __init__(self, num_of_layers, init_size, out_size, dropout=0.0,
                 inner_activation=None, outer_activation=None):
        """
        :param num_of_layers: the total number of layers
        :param init_size: unit size of hidden layers
        :param out_size: output size
        :param inner_activation: the activation function for the inner layers
        :param outer_activation: the activation function for the last layer
        """
        super(MLP, self).__init__()
        self.num_of_layers = num_of_layers
        self._input_size = init_size
        self._output_size = out_size
        self.dropout = Dropout(dropout)
        if self.num_of_layers > 0:
            self.layers = ModuleList(
                [Linear(init_size, init_size) for _ in range(num_of_layers - 1)] + [Linear(init_size, out_size)])
            self.activation_list = [inner_activation for _ in range(num_of_layers - 1)] + [outer_activation]

    def forward(self, x):
        if self.num_of_layers > 0:
            for layer, activation in zip(self.layers, self.activation_list):
                if activation is None:
                    x = self.dropout(layer(x))
                else:
                    x = self.dropout(activation(layer(x)))
        return x

    def get_output_size(self):
        return self._output_size

    def get_input_size(self):
        return self._input_size


class LastState(Layer):
    def __init__(self, input_size):
        super(LastState, self).__init__()
        self._input_size = input_size
        self._output_size = input_size

    def forward(self, x):
        return x[:, -1, :]

    def get_input_size(self):
        return self._input_size

    def get_output_size(self):
        return self._output_size


class AttendedState(Layer):
    def __init__(self, num_of_layers, hidden_size, dropout=0.0, activation=None):
        super(AttendedState, self).__init__()
        self._input_size = hidden_size
        self._output_size = hidden_size
        self.mlp = MLP(num_of_layers=num_of_layers,
                       init_size=hidden_size, out_size=hidden_size,
                       dropout=dropout,
                       inner_activation=activation,
                       outer_activation=activation)

        self.attention = Linear(hidden_size, 1)

    def forward(self, x):
        states_mlp = self.mlp(x)
        att_sc_dist = self.attention(states_mlp).squeeze(-1)
        att_weights = softmax(att_sc_dist, dim=1).unsqueeze(2)
        out_attended = torch.sum(torch.mul(att_weights, x), dim=1)
        return out_attended

    def get_input_size(self):
        return self._input_size

    def get_output_size(self):
        return self._output_size


class AvgPoolingState(Layer):
    def __init__(self, input_size):
        super(AvgPoolingState, self).__init__()
        self._input_size = input_size
        self._output_size = input_size

    def forward(self, x):
        return torch.mean(x, 1)

    def get_input_size(self):
        return self._input_size

    def get_output_size(self):
        return self._output_size


class SequentialModel(Layer):
    def __init__(self, layers):
        super(Layer, self).__init__()
        for i in range(len(layers) - 1):
            assert (layers[i].get_output_size() == layers[i + 1].get_input_size())
        self.layers = ModuleList(layers)
        self._output_size = self.layers[-1].get_output_size()
        self._input_size = self.layers[0].get_input_size()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_input_size(self):
        return self._input_size

    def get_output_size(self):
        return self._output_size

    def add_layer(self, layer):
        assert (layer.get_input_size() == self._input_size)
        self.layers.append(layer)
        self._output_size = layer.get_output_size()