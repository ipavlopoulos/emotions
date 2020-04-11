from twitter.stream import StreamExecutor, GlobalStreamListener
from twitter.datahandler import DataHandler
# from config import PATH, TERMS, CSV_SIZE, UPDATE_DATA_SIZE
import yaml


def load_yaml(path):
    with open(path, 'r') as f:
        config = yaml.load(f)
    return config


def run(lan):
    config = load_yaml("../configuration.yaml")[lan]

    handler = DataHandler(directory=config['path'].format(lan), max_capacity_per_file=config['csv_size'])
    listener = GlobalStreamListener(lan=lan, handler=handler,
                                    update_data_size=config['update_data_size'], stream_all=True)
    executor = StreamExecutor(listener)
    executor.loop(terms=config['terms'])


if __name__ == "__main__":
    run("el")

