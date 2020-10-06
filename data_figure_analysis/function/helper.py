import pandas
import numpy
import os
import math
from nltk.corpus import stopwords
from dateutil.parser import parse

import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1 import host_subplot 
from function import TWEETS_FILE_CSV_MERGE, TWEETS_FILE_CSV, TRAIN_FILE

parent_path = os.path.abspath(os.pardir)

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

# Convert the merged twitter dataset to country level daily confirmed cases (100 x 1)
def _tweets_usa_case_time(df):
    time_stamp = set(df.time)
    usa_count = []
    for t in time_stamp:
        daily_count = df[df.time == t].count()[0]
        usa_count.append(daily_count)
    
    usa_count_df = pandas.DataFrame(usa_count, columns = ['daily_confirm'], index = set(df.time))
    return usa_count_df['daily_confirm'] # Return a DataFrame index = "time"

# Convert the merged twitter dataset to regional level daily confirmed cases (100 x 58)
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
    PATH = parent_path + '/data_figure_analysis/data/tweets/tweets-ids-{}'.format(index)
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


def _convert_to_log(daily_case_series):
    index =daily_case_series.index
    case_log = []
    for case in daily_case_series:
        if case > 0:
            case = math.log10(case)
        case_log.append(case)
    case_log_df = pandas.DataFrame(case_log, columns = ['daily_confirm_log'], index = index )
    return case_log_df['daily_confirm_log']

def _twin_axis_drawing(location, daily_case_series, tweets_count_series):
    host = host_subplot(111, axes_class=AA.Axes) 
    plt.subplots_adjust(right=1.0) 

    par1 = host.twinx() 

    new_fixed_axis = par1.get_grid_helper().new_fixed_axis 
    par1.axis["right"] = new_fixed_axis(loc="right", axes=par1) 
    par1.axis["right"].toggle(all=True) 

    host.set_xlim(0, 180) 
    #host.set_ylim(0, 90000) 

    host.set_xlabel("Days") 
    host.set_ylabel("Daily New cases") 
    par1.set_ylabel("Daily tweets volume") 

    p1, = host.plot(daily_case_series, label="Daily New cases") 
    p2, = par1.plot(tweets_count_series, label="Daily tweets volume") 

    #par1.set_ylim(0, 10000)  

    host.legend() 

    host.axis["left"].label.set_color(p1.get_color()) 
    par1.axis["right"].label.set_color(p2.get_color()) 

    plt.draw()  
    plt.savefig(parent_path + "/data_figure_analysis/output/tweets \
and new daily cases mapping - {}.png".format(location), bbox_inches = "tight")
    plt.clf()

def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

def pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)

def pearson_drawing(location, period, daily_cases, tweets_count):
    tweets_count = tweets_count.tolist()
    day_index = [i for i in range(0, period+1)]
    pearson_list = []
    for i in day_index:
        daily_cases_list = daily_cases[41+i:141+i].tolist()
        pearson = pearson_def(daily_cases_list, tweets_count)
        #print(pearson_def(CA_daily_cases, CA_count))
        pearson_list.append(pearson)
    pearson_df = pandas.DataFrame(pearson_list, index = day_index)
    plt.plot(pearson_df)
    plt.ylabel("Pearson's Correlation Coefficient")
    plt.xlabel("Shifting days forward (Days)")
    plt.savefig(parent_path + "/data_figure_analysis/output/pearson_vs_time - {}.png".format(location)
     , bbox_inches = "tight")
    plt.clf()
