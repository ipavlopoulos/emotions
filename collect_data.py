from utils import load_yaml
from twitter import DataHandler, StreamExecutor, GlobalStreamListener
import click


cli = click.Group()


@cli.command()
@click.option('--lan', default='en')
@click.option('--config', default="configs/configuration.yaml")
@click.option('--twitter_config', default="configs/twitter.yaml")
def execute_data_collection(lan, config, twitter_config):
    config = load_yaml(config)[lan]
    twitter_config = load_yaml(twitter_config)['twitter_credentials']
    handler = DataHandler(directory=config['path'])
    listener = GlobalStreamListener(lan=lan, handler=handler,
                                    update_data_size=config['update_data_size'], stream_all=True)
    executor = StreamExecutor(listener, twitter_config=twitter_config)
    executor.loop(terms=config['terms'])


if __name__ == "__main__":
    execute_data_collection()

