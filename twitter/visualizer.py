from .geo import setup as geo_setup, gpd, plot_loc_with_sentiment
from .datahandler import DataHandler
import matplotlib.pyplot as plt
import time


class Visualizer:
    """
    This class serves to visualize tweets store in a data handler
    """
    def __init__(self, mapping: str, handler: DataHandler):
        """
        :param mapping: the map
        :param handler: a DataHandler
        """
        self.handler = handler
        self.geocode = geo_setup()
        self.map_image = gpd.read_file(gpd.datasets.get_path(mapping))
        self.ax = self.map_image.plot(figsize=(20, 40))

    def pin(self, num_of_data: int, image_path: str):
        """
        pin the data at the map
        :param num_of_data: how many data to pin
        :param image_path: the resulted image
        :return:
        """
        df = self.handler.load_all_data()
        df = df[df['location'].notnull()]
        _, _ = plot_loc_with_sentiment(self.ax, df.location[num_of_data:], df.sentiment[num_of_data:], self.geocode)
        plt.savefig(image_path)

    def loop(self, num_of_data: int,  image_path: str, reload: int = 10):
        """
        pins data in loop mode
        :param num_of_data: how many data to pin
        :param image_path: image
        :param reload: time in seconds to wait in order to collect and pin new data
        :return:
        """
        while True:
            try:
                time.sleep(reload)
                self.pin(num_of_data, image_path)
            except Exception as ex:
                print(ex)
