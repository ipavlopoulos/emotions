
import tweepy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect
from .twitter_config import config
from .sentiment import analyse
from collections import Counter
from .geo import setup as geo_setup, gpd, plot_loc_with_sentiment

import urllib3

mapping = {'en': 'naturalearth_lowres',
           'gr': 'naturalearth_lowres'}


class GlobalStreamListener(tweepy.StreamListener):

    def __init__(self, lan):
        super(GlobalStreamListener, self).__init__()
        self.lan = lan
        self.texts = []
        self.sentiments = []
        self.locations = []
        self.sent_per_country = {}
        self.geocode = geo_setup()
        self.map_image = gpd.read_file(gpd.datasets.get_path(mapping[lan]))
        self.ax = self.map_image.plot(figsize=(20, 40))

    def on_status(self, status):
        sts = status._json
        txt = sts["text"]
        user_location = sts["user"]["location"]  # loc = sts["location"]
        if user_location is not None:
            try:
                lang = detect(txt)
                if lang == self.lan:
                    self.locations.append(user_location)
                    self.sentiments.append(analyse(txt)["compound"])

                    # print(txt, "\n", sent, "\n")
            except:
                print(f"Could not detect the language for: {txt}")
                #todo: add to logger

        if len(self.locations) % 10 == 0:
            # pl_ax, countries = plot_location(self.ax, self.locations, self.geocode)
            pl_ax, countries = plot_loc_with_sentiment(self.ax, self.locations, self.sentiments, self.geocode)
            print(Counter(countries))
            print(len(countries), len(self.sentiments), len(self.locations))
            # neg, neu, pos --> -0.05 < score, -0.05<score<0.05, score<0.05
            print("Overall sentiment (>0.05 means 'positive'):", np.mean(self.sentiments))
            stats = pd.DataFrame({"country": countries, "sentiment": self.sentiments})
            print("Sentiment/Country:", stats.groupby(["country"]).mean())
            plt.savefig("smsearch.png")

    def get_size_of_data(self):
        return len(self.texts)

    def get_last_results(self, num_of_results=10):
        return {'sentiment': self.sentiments[-num_of_results:],
                'text': self.texts[-num_of_results:],
                'location': self.locations[-num_of_results:]}

    def dump_data(self):
        buffered_data = self.get_last_results(self.get_size_of_data())
        df = pd.DataFrame.from_dict(buffered_data)
        df.to_csv("results.csv", index=False)


class StreamExecutor:
    def __init__(self, listener: GlobalStreamListener) -> None:
        self.auth = tweepy.OAuthHandler(config['api_key'], config['api_secret_key'])
        self.auth.set_access_token(config['access_token'], config['access_token_secret'])
        self.api = tweepy.API(self.auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        self.listener = listener
        self.stream = None

    def setup(self, terms=('covid-19', 'coronavirus')):
        self.stream = tweepy.Stream(auth=self.api.auth, listener=self.listener)
        self.stream.filter(track=terms)

    def set_up_with_exception_handling(self, terms=('covid-19', 'coronavirus')):
        try:
            self.setup(terms)
        except urllib3.exceptions.ProtocolError:
            print("urllib3.exceptions.ProtocolError ")

    def loop(self, terms=('covid-19', 'coronavirus')):
        while True:
            self.set_up_with_exception_handling(terms=terms)

    def get_last_results(self, num_of_results=10):
        return self.listener.get_last_results(num_of_results)

    def dump_data(self):
        self.listener.dump_data()
