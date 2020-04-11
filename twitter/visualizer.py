from .geo import setup as geo_setup, gpd, plot_loc_with_sentiment
from .datahandler import DataHandler
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, mapping: str, handler: DataHandler):
        self.handler = handler

        self.geocode = geo_setup()
        self.map_image = gpd.read_file(gpd.datasets.get_path(mapping))
        self.ax = self.map_image.plot(figsize=(20, 40))

    def pin(self):
        df = self.handler.load_all_data()
        df = df.filter(lambda x: x.location is not None, axis=1)
        _, _ = plot_loc_with_sentiment(self.ax, df.locations, df.sentiments, self.geocode)
        plt.savefig("smsearch.png")
