from mpl_toolkits.axes_grid1 import host_subplot 
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import pandas
import math
import helper
import os

basePath = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = basePath + '/data/time_series_covid19_confirmed_US.csv'
TWEETS_FILE_TXT = basePath +'./data/tweet-ids-001.txt'
TWEETS_FILE_CSV = basePath + '/data/tweets-ids-%s.csv'
TWEETS_FILE_CSV_MERGE = basePath + '/data/tweets-ids-merge-filter.csv'
GLOVE_FILE = basePath + '/misc/glove.twitter.27B.%sd.txt'
GLOVE_WV_FILE = basePath + '/misc/glove_%s.wv'



states_full = ['Alabama','Alaska','American Samoa','Arizona','Arkansas','California',
 'Colorado','Connecticut','Delaware','Diamond Princess','District of Columbia','Florida','Georgia',
 'Grand Princess','Guam','Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana',
 'Maine','Maryland','Massachusetts','Michigan','Minnesota','Mississippi','Missouri',
 'Montana','Nebraska','Nevada','New Hampshire','New Jersey','New Mexico','New York',
 'North Carolina','North Dakota','Northern Mariana Islands','Ohio','Oklahoma','Oregon','Pennsylvania','Puerto Rico',
 'Rhode Island','South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont',
 'Virgin Islands','Virginia','Washington','West Virginia','Wisconsin','Wyoming']
 # DI = 'Diamond Princess', GP = 'Grand Princess', 
 # GU = 'Guam', MP = 'Northern Mariana Islands'

states_abb = ["AL", "AK", "AS", "AZ", "AR", "CA", "CO", "CT", "DE", "DI", "DC", "FL", 
            "GA", "GP", "GU", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", 
            "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", 
            "NY", "NC", "ND", "MP", "OH", "OK", "OR", "PA", "PR", "RI", "SC", "SD", 
            "TN", "TX", "UT", "VT", "VI", "VA", "WA", "WV", "WI", "WY"]

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
    plt.savefig(basePath + "/output/tweets and new daily cases mapping - {}.png".format(location)
    , bbox_inches = "tight")
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
    plt.savefig(basePath + "/output/pearson_vs_time - {}.png".format(location)
     , bbox_inches = "tight")
    plt.clf()


df_list = []
try:
    df_usa = pandas.read_csv(TWEETS_FILE_CSV_MERGE)
except:
    print("The file does not exists, will merge it now")
if not os.path.exists(TWEETS_FILE_CSV_MERGE):
    for i in range(11):
        PATH = TWEETS_FILE_CSV%i
        df = pandas.read_csv(PATH)
        df_usa_partition = _get_usa_tweets_from_csv(df, states_full, states_abb)
        n_partition= df_usa_partition.count()[0]
        print("In the process to merge the {}/11 file with length = {}".format(i, n_partition))
        
        df_list.append(df_usa_partition)
        df_usa = pandas.concat(df_list)

        df_usa.sort_values("time",inplace=True)
        # Reset index
        #df_usa = df_usa.reset_index()
        df_usa.to_csv(TWEETS_FILE_CSV_MERGE)
    n = df_usa.count()[0]
    print("Successfully merge files with total length = {}".format(n))

period = 30
# Country level plot
usa_tweets_count = helper._tweets_usa_case_time(df_usa)
usa_daily_cases = helper._get_training_data_from_csv()['usa']
_twin_axis_drawing('USA National', usa_daily_cases, usa_tweets_count )

# Log10 plot comparison
usa_tweets_count_log = _convert_to_log(usa_tweets_count)
usa_daily_cases_log =  _convert_to_log(usa_daily_cases)
_twin_axis_drawing('USA National - log', usa_daily_cases_log, usa_tweets_count_log)

pearson_drawing('USA', period, usa_daily_cases, usa_tweets_count)

# Reference: https://www.nytimes.com/interactive/2020/us/coronavirus-us-cases.html
# state level plot
states = ['New York', 'New Jersey', 'Pennsylvania', 'Michigan', 'Connecticut', 'Massachusetts', 
          'Maryland', 'Rhode Island', 'Illinois', 'District of Columbia', 'Vermont', 
          'California', 'North Carolina', 'Utah', 'Arizona', 'Alabama']
# Normal and log10 volumne plot
for state in states:
    state_tweets_count = helper._tweets_state_case_time(df_usa, state)
    state_daily_cases = helper._get_training_data_from_csv()[state]
    _twin_axis_drawing(state, state_daily_cases, state_tweets_count)

    state_tweets_count_log = _convert_to_log(state_tweets_count)
    state_daily_cases_log =  _convert_to_log(state_daily_cases)
    _twin_axis_drawing(state + ' - log', state_daily_cases_log, state_tweets_count_log)

    pearson_drawing(state, period, state_daily_cases, state_tweets_count)
    #pearson_drawing(state, period, state_daily_cases_log, state_tweets_count_log)