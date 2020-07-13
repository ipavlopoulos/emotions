from utils import load_yaml
import pandas as pd
import click
from datetime import datetime, timedelta
import numpy as np

cli = click.Group()


@cli.command()
@click.option('--lan', default='en')
@click.option('--config', default="configs/configuration.yaml")
def dump(lan, config, country_code):
    # load the tweets of the requested language
    config = load_yaml(config)[lan]
    data = pd.read_csv(f"{config['path']}tweets_id_0.csv")
    tweets = data[data.is_retweet == False]
    # fetch only tweets from yesterday
    tweets.set_index(pd.to_datetime(tweets.created_at, format='%a %b %d %H:%M:%S +0000 %Y'), inplace=True)
    yesterday = datetime.now() - timedelta(1)
    # filter past ones (the cron should run at 00:00:01)
    tweets = tweets[tweets.index >= yesterday]
    tweets.to_csv(f"{config['path']}.{str(yesterday)[:10]}.{country_code}.csv", index=False)


@cli.command()
@click.option('--lan', default='en')
@click.option('--config', default="configs/configuration.yaml")
@click.option('--days', default=7)
@click.option('--country_code', default="US")
def aggregate_n_dump(lan, config, days, country_code):
    # load the tweets of the requested language
    config = load_yaml(config)[lan]
    data = pd.read_csv(f"{config['path']}tweets_id_0.csv")
    try:
        data = pd.concat([data, pd.read_csv(f"{config['path']}tweets_id_1.csv")])
    except:
        print("ERROR: No other file saved so far")
    tweets = data[data.is_retweet == False]
    tweets['day'] = pd.to_datetime(tweets.created_at, format='%a %b %d %H:%M:%S +0000 %Y').dt.strftime('%Y-%m-%d')
    # fetch only tweets from yesterday
    tweets.set_index(pd.to_datetime(tweets.created_at, format='%a %b %d %H:%M:%S +0000 %Y'), inplace=True)
    past = datetime.now() - timedelta(days)
    # filter past ones (the cron should run at 00:00:01)
    tweets = tweets[tweets.index >= past]
    tweets = tweets[tweets.country_code == country_code]
    tweets = tweets[tweets.full_name.notna()]
    tweets["state"] = tweets.full_name.apply(find_us_state)
    places = pd.DataFrame()
    places["sentiment"] = tweets.groupby(["day", "state"]).sentiment.apply(lambda x: np.mean([float(s) for s in x]))
    places["size"] = tweets.groupby(["day", "state"]).sentiment.apply(lambda x: len(x))
    for abbr in state_map:
        state = state_map[abbr]
        if state in places.index.get_level_values(1):
            places.xs(state, level=1).reset_index().to_csv(f"docs/DATA/{state}.csv", index=False)


state_map = {"NV": "Nevada",
             "TX": "Texas",
             "FL": "Florida",
             "CA":"California",
             "CO": "Colorado",
             "IL":"Ilinois",
             "NY":"New York",
             "OH":"Ohio",
             "WA": "Washington",
             "MA": "Massachusetts",
             "MI": "Michigan",
             "NJ": "New Jersey",
             "GA": "Georgia",
             "NC": "North Carolina",
             "PA": "Pensilvania",
             "OK": "Oklahoma",
             "MN":"Minnesota",
             "IN": "Indiana",
             "VA": "Virginia",
             "TN": "Tennessee",
             "CT": "Connecticut",
             "WI": "Wisconsin",
             "AZ": "Arizona",
             "HI": "Hawaii",
             "OR": "Oregon",
             "IA": "Iowa",
             "MO": "Missouri",
             "NH": "New Hampshire",
             "AR": "Arkansas",
             "AL": "Alabama",
             "WI": "Wisconsin",
             "RI": "Rhode Island",
             "LA": "Louisiana"}


def find_us_state(place, state_map=state_map):
    if place.endswith('USA'):
        return place.split(",")[0]
    else:
        try:
            return state_map[place.split(",")[1].strip()]
        except:
            return "UNKNOWN_STATE"


if __name__ == "__main__":
    aggregate_n_dump()


