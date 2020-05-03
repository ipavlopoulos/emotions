import pandas as pd
from utils import tokenize, create_freq_vocabulary, get_top_freq_words, get_embeddings, load_yaml
from utils import create_batches, create_final_dictionary, get_embeddings_matrix, train, full_classification_report, load_model
from modeling import ModelFactory
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
import click
cli = click.Group()


@cli.command()
@click.option('--model_name', default='cnn')
@click.option('--config', default="config/config.yaml")
def run(config, model_name):
    config = load_yaml(config)
    if model_name not in config['model']:
        raise NotImplementedError("{} is not implemented. ".format(model_name))
    preprocessing_params = config['preprocessing']
    training_params = config['training']
    model_params = config['model'][model_name]
    train_df = pd.read_csv(preprocessing_params['train_path'], sep='\t')
    test_df = pd.read_csv(preprocessing_params['test_path'], sep='\t')
    t_list = preprocessing_params['target_list']
    model_params['targets'] = len(t_list)

    train_df['tokens'] = train_df['Tweet'].map(lambda x: tokenize(x))
    test_df['tokens'] = test_df['Tweet'].map(lambda x: tokenize(x))
    train_df['lengths'] = train_df['tokens'].map(lambda x: len(x))
    test_df['lengths'] = test_df['tokens'].map(lambda x: len(x))

    word_freq_dict = create_freq_vocabulary(list(train_df['tokens']) + list(test_df['tokens']))

    tokens = get_top_freq_words(word_freq_dict, 1)

    train_df = train_df.sort_values(by="lengths")
    test_df = test_df.sort_values(by="lengths")
    embeddings = get_embeddings(path=preprocessing_params['embeddings_path'])
    w2i = create_final_dictionary(tokens, embeddings,
                                  unk_token=preprocessing_params['unk_token'],
                                  pad_token=preprocessing_params['pad_token'])
    emb_matrix = get_embeddings_matrix(w2i, embeddings, preprocessing_params['embedding_size'])

    model_params['embeddings'] = emb_matrix

    train_batches = create_batches(train_df, training_params['batch_size'], w2i=w2i,
                                   pad_token=preprocessing_params['pad_token'],
                                   unk_token=preprocessing_params['unk_token'], target_list=t_list)
    test_batches = create_batches(test_df, training_params['batch_size'], w2i=w2i,
                                  pad_token=preprocessing_params['pad_token'],
                                  unk_token=preprocessing_params['unk_token'], target_list=t_list)

    model = ModelFactory.get_model(model_name, model_params)
    optimizer = Adam(model.trainable_weights, training_params['lr'])
    criterion = BCEWithLogitsLoss()
    train(model, train_batches, test_batches, optimizer, criterion, epochs=training_params['epochs'],
          init_patience=training_params['patience'], cuda=False, target_list=t_list)
    model = load_model(model)
    full_classification_report(model, test_batches, t_list)


if __name__ == "__main__":
    run()





