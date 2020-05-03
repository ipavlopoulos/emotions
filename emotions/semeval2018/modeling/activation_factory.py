from torch.nn.functional import relu, tanh


class ActivationFactory:
    @staticmethod
    def get_activation(str_activation):
        if str_activation == 'relu':
            return relu
        elif str_activation == 'tanh':
            return tanh
        else:
            return None
