# smsearch
* Search Social Media (e.g., Twitter) about specific topics (e.g., COVID-19).
* Analyse the retrieved posts (e.g., doing Sentiment Analysis).
<hr/>

<img src="https://github.com/ipavlopoulos/smsearch/blob/master/smsearch.png" width="200"/>

## Dependencies
* [geopy](https://pypi.org/project/geopy/)
* [tweepy](https://pypi.org/project/tweepy/)
* [langdetect](https://pypi.org/project/langdetect/)
* [descartes](https://pypi.org/project/descartes/)
* [vaderSentiment](https://pypi.org/project/vaderSentiment/)
* Also, you need to sign up for Twitter Dev API credentials and update ``twitter_config.py``

## Under development
* Add logger.
* Add sampling to perform sentiment analysis (for fairness), and do not show all countries (ones with too few instances should be excluded).
* Show sentiments on map.
* Add or store time based estimations; e.g., to see the sentiment over time.
