import pandas
import numpy as np
import os
from function import helper
from function import TWEETS_FILE_CSV_MERGE, TWEETS_FILE_CSV

parent_path = os.path.abspath(os.pardir)

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


# Preproces COVID-19 daily confirmed cases 
confirm = pandas.read_csv('./data/time_series_covid19_confirmed_US.csv')
dates = confirm.columns[11:]
total_confirm_sum= []
for date in dates:
    date_confirm_sum= []
    for state in states_full:
        state_cases = sum(confirm[confirm.Province_State == state][date].values)
        date_confirm_sum.append(state_cases)
    total_confirm_sum.append(date_confirm_sum)
confirm_states = pandas.DataFrame(total_confirm_sum, columns = states_full)
confirm_states.index

daily_cases = confirm_states.astype(np.int64)
daily_cases = daily_cases.diff().fillna(0)
### DataFrame to .txt file without header and idnex
daily_cases.to_csv(parent_path + '/time_series_models/data/daily_cases.txt', sep=',', index=False, header = 0)


# Preprocess twitter data
df_list = []
try:
    df_usa = pandas.read_csv(TWEETS_FILE_CSV_MERGE)
except:
    print("The file does not exists, will merge it now")
if not os.path.exists(TWEETS_FILE_CSV_MERGE):
    for i in range(12):
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
helper._twin_axis_drawing('USA National', usa_daily_cases, usa_tweets_count )

# Log10 plot comparison
usa_tweets_count_log = helper._convert_to_log(usa_tweets_count)
usa_daily_cases_log =  helper._convert_to_log(usa_daily_cases)
helper._twin_axis_drawing('USA National - log', usa_daily_cases_log, usa_tweets_count_log)

helper.pearson_drawing('USA', period, usa_daily_cases, usa_tweets_count)

# Reference: https://www.nytimes.com/interactive/2020/us/coronavirus-us-cases.html
# state level plot
states = ['New York', 'New Jersey', 'Pennsylvania', 'Michigan', 'Connecticut', 'Massachusetts', 
          'Maryland', 'Rhode Island', 'Illinois', 'District of Columbia', 'Vermont', 
          'California', 'North Carolina', 'Utah', 'Arizona', 'Alabama']
# Normal and log10 volumne plot
for state in states:
    state_tweets_count = helper._tweets_state_case_time(df_usa, state)
    state_daily_cases = helper._get_training_data_from_csv()[state]
    helper._twin_axis_drawing(state, state_daily_cases, state_tweets_count)

    state_tweets_count_log = helper._convert_to_log(state_tweets_count)
    state_daily_cases_log =  helper._convert_to_log(state_daily_cases)
    helper._twin_axis_drawing(state + ' - log', state_daily_cases_log, state_tweets_count_log)

    helper.pearson_drawing(state, period, state_daily_cases, state_tweets_count)
    #pearson_drawing(state, period, state_daily_cases_log, state_tweets_count_log)