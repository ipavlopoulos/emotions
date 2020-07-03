from utils import load_yaml
import pandas as pd
import click


cli = click.Group()


@cli.command()
@click.option('--lan', default='en')
@click.option('--config', default="configs/configuration.yaml")
def dump(lan, config):
    # load the tweets of the requested language
    config = load_yaml(config)[lan]
    data = pd.read_csv(f"{config['path']}tweets_id_0.csv")
    tweets = data[data.is_retweet == False]
    # fetch only tweets from yesterday
    tweets.set_index(pd.to_datetime(tweets.created_at, format='%a %b %d %H:%M:%S +0000 %Y'))
    yesterday = tweets.groupby(pd.Grouper(freq='d')).apply(list).index[-2]
    # dump
    tweets.loc[yesterday].to_csv(config['path']+str(yesterday)[:10] + ".csv", index=False)


if __name__ == "__main__":
    dump()


