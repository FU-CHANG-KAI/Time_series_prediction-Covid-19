import pandas
import numpy
import os
import json
from nltk.corpus import stopwords
from dateutil.parser import parse
import os

basePath = os.path.dirname(os.path.abspath(__file__))

TRAIN_FILE = basePath + '/data/time_series_covid19_confirmed_US.csv'
TWEETS_FILE_TXT = basePath +'./data/tweet-ids-001.txt'
TWEETS_FILE_CSV = basePath + '/data/tweets-ids-%s.csv'
TWEETS_FILE_CSV_MERGE = basePath + '/data/tweets-ids-merge-filter.csv'
GLOVE_FILE = basePath + '/misc/glove.twitter.27B.%sd.txt'
GLOVE_WV_FILE = basePath + '/misc/glove_%s.wv'

def _get_training_data_from_csv():
    # Transform to a specific fotmat fits RNN models
    df = pandas.read_csv(TRAIN_FILE)
    dates = df.columns[11:]
    states = list(set(df.Province_State.values))

    total_confirm_sum= []
    for date in dates:
        date_confirm_sum= []
        for state in states:
            state_cases = sum(df[df.Province_State == state][date].values)
            date_confirm_sum.append(state_cases)
        total_confirm_sum.append(date_confirm_sum)
    confirm_states = pandas.DataFrame(total_confirm_sum, columns = states)

    # Sequence according to the columns(States name alphabet)
    confirm_states = confirm_states.reindex(sorted(confirm_states.columns), axis=1)
    
    # Convert daily confirmed total number to daily increase
    daily_cases = confirm_states.astype(numpy.int64)
    # Check the confirmed cases at first day- 21 Jan 2020 which was not zero, Washington
    # daily_cases.iloc[0]
    daily_cases['Washington'][0] = 1
    daily_cases = daily_cases.diff().fillna(0)
    daily_cases['usa'] = daily_cases.apply(lambda x: x.sum(),axis=1)

    # return dataFrame with a period of time 
    return daily_cases

def _merge_tweets_from_txt(): # Deal with split tweets data 
    walk_dir = basePath + '\data\covid-id'

    print('walk_dir = ' + walk_dir)
    print('walk_dir (absolute) = ' + os.path.abspath(walk_dir))

    for root, subdirs, files in os.walk(walk_dir):
        print(root)
        path, book_name = os.path.split(root)
        stored_path = "D:\Project\Msc project\Time_series_Covid-19\data\covid-id-merge"
        if len(files) < 10:
            continue
        stored_filename = files[0][:-7]
        container = []
        error_occur = []
        for filename in files:
            if filename[:-7] != stored_filename:
                print("Store in ",stored_filename)
                f = open(stored_path + '/' + stored_filename + '.txt', 'w')
                for id in container:
                    f.write('%s' % id) 
                container = []
                stored_filename = filename[:-7]
            try: 
                fp = open(root + '/' + filename, 'r')
                lines = fp.readlines()
                container += lines
            except:
                print("===================Format error occurs at {}=================".format(filename))
                error_occur.append(filename[:-5])
                continue
            # Store the last .txt file
        print("Store in ", stored_filename)
        f = open(stored_path + '/' + stored_filename + '.txt', 'w')
        for id in container:
            f.write('%s' % id)

    print("===============Merge complete===============")

# Can be deleted
def _catch_error_tweetes():
    PATH = basePath + '/data/tweets-id/tweet-ids-001.txt'
    container = []
    fp = open(PATH, 'r')
    lines = fp.readlines()
    n = len(lines)
    count_true = 0
    count_false = 0
    for line in lines:
        try:
            float(line)
            container += lines
            count_true += 1
            rate = count_true/n * 100
            print("Successfully process {:.5f}% tweets data".format(rate))
        except KeyError:
            count_false += 0
            print("===================Format error occurs=================")
            print("False count = {}".format(count_false))


    f = open(basePath + '/data/tweets-id/tweet-id_filter.txt', 'w')
    for id in container:
        f.write('%s' % id)

def _merge_and_save_csv():
    PATH = TWEETS_FILE_CSV
    df_list = []
    for i in range(11):
        df_list.append(pandas.read_csv(PATH%i))
    df = pandas.concat(df_list)
    df.to_csv(TWEETS_FILE_CSV_MERGE)
    print("The merged files is successfully stored")

def _get_usa_tweets_from_csv(df, states_full, states_abb):
    # Tweets location is "place": 
    # https://developer.twitter.com/en/docs/tutorials/filtering-tweets-by-location
    # 1. dropna NaN at feature "place"
    print("1. dropna NaN at feature [place]")
    df = df.dropna(subset = ["place"])

    # 2. Keep tweets in USA when a new feature "place_filter" is True
    print("2. Keep tweets in USA when a new feature [place_filter] is True")
    df["place_filter"] = df["place"].apply(lambda x: None if (x[:-6] not in states_full and x[-2:] not in states_abb) else True)
    
    df = df.dropna(subset = ["place_filter"])

    # 3. Geo-tag the location type i.e. Los angelas, CA or Californua, USA using "place_filter"
    # full:1, abbreviation:2  
    df["place_filter"] = df["place"].apply(lambda x: 1 if x[:-6] in states_full else 2)
    
    # 4. Transform the geolocation to state level at "place"
    for i in df["place_filter"].index:
        if df["place_filter"][i] == 1:
            df["place"][i] = states_full[states_full.index(df["place"][i][:-6])]
        else:
            df["place"][i] = states_full[states_abb.index(df["place"][i][-2:])]
    
    # 5. keep tweets with lang == "en"
    df = df[df.lang == 'en']

    # 6. Sequence according to the datetime using dateutil.parser
    # Datetime for Jan 22 - The first case occurs in USA
    Date_first_case = parse("Jan 22 01:53:18 +0000 2020") 
    df['time'] = df['created_at'].apply(lambda x: parse(x))
    df['time']  = df['time'].apply(lambda x:(x-Date_first_case).days)

    df = df[['time','place','text']]

    return df

def _tweets_usa_case_time(df):
    time_stamp = set(df.time)
    usa_count = []
    for t in time_stamp:
        daily_count = df[df.time == t].count()[0]
        usa_count.append(daily_count)
    
    usa_count_df = pandas.DataFrame(usa_count, columns = ['daily_confirm'], index = set(df.time))
    return usa_count_df['daily_confirm'] # Return a DataFrame index = "time"
    #return usa_count # directly return a list which is suitable for drawing

def _tweets_state_case_time(df, state):
    time_stamp = set(df.time)

    #state_count = {}
    #for state in states_full:
    #    state_count[state] = []
    state_count = []
    for t in time_stamp:
        daily_state_count = len(df[(df.time == t) & (df.place == state)]['time'])
        state_count.append(daily_state_count)

    state_count_df = pandas.DataFrame(state_count, columns = ['daily_confirm'], index = set(df.time) )
    return state_count_df['daily_confirm'] # Return a Series with index = "time"
    #return state_count

def _save_to_txt(id_list, index):
    PATH = basePath + '/data/tweets-ids-{}'.format(index)
    with open(PATH,'w') as output:
        for id in id_list:
            output.write('%s' % id)
    print('Finish saving No.{} .txt file'.format(index))

def _get_stop_words(strip_handles=False, strip_rt=False):
    ''' Returns stopwords '''
    stop_words = (stopwords.words('english'))
    if strip_rt: stop_words += ['rt']
    # TODO: if strip_handles
    return set(stop_words)