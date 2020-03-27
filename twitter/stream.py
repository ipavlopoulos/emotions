
import tweepy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect
from .twitter_config import config
from .sentiment import analyse
from collections import Counter
from .geo import setup as geo_setup, plot_loc as plot_location, loc_to_latlon, gpd

auth = tweepy.OAuthHandler(config['api_key'], config['api_secret_key'])
auth.set_access_token(config['access_token'], config['access_token_secret'])

api = tweepy.API(auth,
                 wait_on_rate_limit=True,
                 wait_on_rate_limit_notify=True)

locations = []
sentiments = []
sent_per_country = {}

geocode = geo_setup()
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ax = world.plot(figsize=(20,40))

class EnTermsStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        sts = status._json
        txt = sts["text"]
        uloc = sts["user"]["location"]  # loc = sts["location"]
        if uloc is not None:
            try:
                lang = detect(txt)
                if lang == "en":
                    locations.append(uloc)
                    sentiments.append(analyse(txt)["compound"])
                    # print(txt, "\n", sent, "\n")
            except:
                print(f"Could not detect the language for: {txt}")
                #todo: add to logger

        if len(locations) % 10 == 0:
            pl_ax, countries = plot_location(ax, locations, geocode)
            print(Counter(countries))
            print(len(countries), len(sentiments), len(locations))
            # neg, neu, pos --> -0.05 < score, -0.05<score<0.05, score<0.05
            print("Overall sentiment (>0.05 means 'positive'):", np.mean(sentiments))
            stats = pd.DataFrame({"country": countries, "sentiment": sentiments})
            print("Sentiment/Country:", stats.groupby(["country"]).mean())
            plt.savefig("smsearch.png")


def set_up(terms = ['covid-19', 'coronavirus']):
    stream = tweepy.Stream(auth=api.auth, listener=EnTermsStreamListener())
    stream.filter(track=terms)