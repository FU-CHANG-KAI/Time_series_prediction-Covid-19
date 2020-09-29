import os

basePath = os.path.dirname(os.path.abspath(__file__))

TRAIN_FILE = basePath + '/data/time_series_covid19_confirmed_US.csv'
TWEETS_FILE_TXT = basePath +'./data/tweet-ids-001.txt'
TWEETS_FILE_CSV = basePath + '/data/tweets-ids-%s.csv'
TWEETS_FILE_CSV_MERGE = basePath + '/data/tweets-ids-merge-filter.csv'
GLOVE_FILE = basePath + '/misc/glove.twitter.27B.%sd.txt'
GLOVE_WV_FILE = basePath + '/misc/glove_%s.wv'