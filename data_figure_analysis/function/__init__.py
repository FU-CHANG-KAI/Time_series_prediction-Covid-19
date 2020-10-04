import os

parent_path = os.path.abspath(os.pardir)

TRAIN_FILE = parent_path + '/data_figure_analysis/data/time_series_covid19_confirmed_US.csv'
TWEETS_FILE_CSV = parent_path + '/data_figure_analysis/data/tweets-ids-%s.csv'
TWEETS_FILE_CSV_MERGE = parent_path + '/data_figure_analysis/data/tweets-ids-merge-filter.csv'