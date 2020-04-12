import tweepy
import pandas as pd
from langdetect import detect
from .sentiment import analyse_per_language
from .datahandler import DataHandler
import gc


class GlobalStreamListener(tweepy.StreamListener):
    """
    Twitter listener. collects tweets and stores it to a data-handler
    """
    def __init__(self, lan: str,
                 handler: DataHandler,
                 update_data_size: int,
                 max_size: int = 100000,
                 stream_all: bool = False):
        """
        :param lan: the language of the tweets to be collected
        :param handler: a data-handler to store the data
        :param update_data_size: after how many data to dump on the data-handler
        :param max_size: when achieves it, it must empty all lists
        :param stream_all: whether to store all tweets or only those with geo-location
        """
        super(GlobalStreamListener, self).__init__()
        self.lan = lan
        self.texts = []
        self.sentiments = []
        self.locations = []
        self.created_at = []
        self.handler = handler
        self.update_data_size = update_data_size
        self.max_size = max_size
        self.stream_all = stream_all

    def on_status(self, status):
        sts = status._json
        txt = sts["text"]
        user_location = sts["user"]["location"]
        created_at = sts['created_at']
        if user_location is not None or self.stream_all:
            try:
                lang = detect(txt)
                if lang == self.lan and txt not in self.texts:
                    self.locations.append(user_location)
                    self.sentiments.append(analyse_per_language(txt, self.lan)["compound"])
                    self.created_at.append(created_at)
                    self.texts.append(txt)
            except:
                print(f"Could not detect the language for: {txt}")
                #todo: add to logger

        if len(self.locations) % self.update_data_size == 0:
            self.dump_data()

    def get_size_of_data(self):
        return len(self.texts)

    def get_last_results(self, num_of_results=10):
        return {'sentiment': self.sentiments[-num_of_results:],
                'text': self.texts[-num_of_results:],
                'location': self.locations[-num_of_results:],
                'created_at': self.created_at[-num_of_results:]}

    def dump_data(self):
        buffered_data = self.get_last_results(num_of_results=self.update_data_size)
        df = pd.DataFrame.from_dict(buffered_data)
        self.handler.store_new_data(df)
        if self.get_size_of_data() % self.max_size == 0:
            self.empty_lists()

    def init_lists(self):
        """
        re-initialize the lists
        :return:
        """
        self.texts = []
        self.sentiments = []
        self.locations = []
        self.created_at = []

    def empty_lists(self):
        """
        empties the lists, calls the garbage collector and re-initialize the lists
        :return:
        """
        del self.texts, self.sentiments, self. locations, self.created_at
        gc.collect()
        self.init_lists()


class StreamExecutor:
    """
    steams tweets by using a listener
    """
    def __init__(self, listener: GlobalStreamListener,
                 twitter_config: dict) -> None:
        """

        :param listener: a GlobalStreamListener as defined above
        :param twitter_config: a dict containing all credentials needed for the twitter API
        """
        self.auth = tweepy.OAuthHandler(twitter_config['api_key'], twitter_config['api_secret_key'])
        self.auth.set_access_token(twitter_config['access_token'], twitter_config['access_token_secret'])
        self.listener = listener
        self.stream = None

    def setup_and_run(self, terms: tuple):
        """
        the actual streaming call
        :param terms: stream only tweets with hash-tags one on these terms
        :return:
        """
        api = tweepy.API(self.auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        self.stream = tweepy.Stream(auth=api.auth, listener=self.listener)
        self.stream.filter(track=terms)

    def set_up_with_exception_handling(self, terms: tuple):
        """
        executes self.setup_and_run with exception handling
        :param terms: stream only tweets with hash-tags one on these terms
        :return:
        """
        try:
            self.setup_and_run(terms)
        except Exception as ex:
            print(str(ex))

    def loop(self, terms: tuple):
        """
        executes constantly the streaming method
        :param terms: stream only tweets with hash-tags one on these terms
        :return:
        """
        while True:
            self.set_up_with_exception_handling(terms=terms)
