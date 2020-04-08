from flask import Flask
from twitter.stream import loop
app = Flask(__name__)


@app.route('/')
def collect_tweets():
    # loop(lan='en')
    return "hello"


if __name__ == '__main__':
   app.run()