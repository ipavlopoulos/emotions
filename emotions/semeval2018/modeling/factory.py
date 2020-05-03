from .cnn import CNN
from .last_state_rnn import LastStateRNN
from .mlp import MLPModel
from .mlp_with_multiple_attention import MLPWithAttentionModel
from .multi_attention_rnn import MultiAttentionRNN
from .multi_attention_rnn_concat_cnn import MultiAttentionRNNConcatCNN
from .projected_multi_attention_rnn import ProjectedMultiAttentionRNN
from .single_attention_rnn import SingleAttentionRNN


class ModelFactory:
    @staticmethod
    def get_model(model_name, model_params):
        if model_name == "cnn":
            return CNN(**model_params)
        elif model_name == "last_state_rnn":
            return LastStateRNN(**model_params)
        elif model_name == "mlp":
            return MLPModel(**model_params)
        elif model_name == "mlp_with_multiple_attention":
            return MLPWithAttentionModel(**model_params)
        elif model_name == "multi_attention_rnn":
            return MultiAttentionRNN(**model_params)
        elif model_name == "multi_attention_rnn_concat_cnn":
            return MultiAttentionRNNConcatCNN(**model_params)
        elif model_name == "projected_multi_attention_rnn":
            return ProjectedMultiAttentionRNN(**model_params)
        elif model_name == "single_attention_rnn":
            return SingleAttentionRNN(**model_params)
        else:
            raise NotImplementedError
