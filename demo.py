from twitter.stream import StreamExecutor, GlobalStreamListener
from twitter.datahandler import DataHandler
from config import PATH, TERMS, CSV_SIZE, UPDATE_DATA_SIZE


def run(lan):
    handler = DataHandler(directory=PATH.format(lan), max_capacity_per_file=CSV_SIZE)
    listener = GlobalStreamListener(lan=lan, handler=handler, update_data_size=UPDATE_DATA_SIZE)
    executor = StreamExecutor(listener)
    executor.loop(terms=TERMS[lan])


if __name__ == "__main__":
    run("el")

