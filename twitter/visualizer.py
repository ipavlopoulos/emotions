from .geo import setup as geo_setup, gpd, plot_loc_with_sentiment
from .datahandler import DataHandler
from config import MAPPING
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, lan: str, handler: DataHandler):
        self.lan = lan
        self.handler = handler

        self.geocode = geo_setup()
        self.map_image = gpd.read_file(gpd.datasets.get_path(MAPPING[lan]))
        self.ax = self.map_image.plot(figsize=(20, 40))

    def pin(self):
        df = self.handler.load_all_data()
        pl_ax, countries = plot_loc_with_sentiment(self.ax, df.locations, df.sentiments, self.geocode)
        # print(Counter(countries))
        print(len(countries), len(df.sentiments), len(df.locations))
        # neg, neu, pos --> -0.05 < score, -0.05<score<0.05, score<0.05
        # print("Overall sentiment (>0.05 means 'positive'):", np.mean(self.sentiments))
        # stats = pd.DataFrame({"country": countries, "sentiment": self.sentiments})
        # print("Sentiment/Country:", stats.groupby(["country"]).mean())
        plt.savefig("smsearch.png")
