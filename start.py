from utils import load_yaml
from twitter import DataHandler, StreamExecutor, GlobalStreamListener


def execute_data_collection(lan):
    config = load_yaml("configs/configuration.yaml")[lan]
    twitter_config = load_yaml("configs/twitter.yaml")['twitter_credentials']
    handler = DataHandler(directory=config['path'], max_capacity_per_file=config['csv_size'])
    listener = GlobalStreamListener(lan=lan, handler=handler,
                                    update_data_size=config['update_data_size'], stream_all=True)
    executor = StreamExecutor(listener, twitter_config=twitter_config)
    executor.loop(terms=config['terms'])


if __name__ == "__main__":
    execute_data_collection("en")

