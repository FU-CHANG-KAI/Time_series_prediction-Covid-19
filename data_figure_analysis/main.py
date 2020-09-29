import pandas
import os
from tfn.helper import _save_to_txt, _get_usa_tweets_from_csv,_tweets_usa_case_time, _tweets_state_case_time
from tfn import TWEETS_FILE_TXT

states_full = ['Alabama','Alaska','American Samoa','Arizona','Arkansas','California',
 'Colorado','Connecticut','Delaware','Diamond Princess','District of Columbia','Florida','Georgia',
 'Grand Princess','Guam','Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana',
 'Maine','Maryland','Massachusetts','Michigan','Minnesota','Mississippi','Missouri',
 'Montana','Nebraska','Nevada','New Hampshire','New Jersey','New Mexico','New York',
 'North Carolina','North Dakota','Northern Mariana Islands','Ohio','Oklahoma','Oregon','Pennsylvania','Puerto Rico',
 'Rhode Island','South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont',
 'Virgin Islands','Virginia','Washington','West Virginia','Wisconsin','Wyoming']

states_abb = ["AL", "AK", "AS", "AZ", "AR", "CA", "CO", "CT", "DE", "DI", "DC", "FL", 
            "GA", "GP", "GU", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", 
            "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", 
            "NY", "NC", "ND", "MP", "OH", "OK", "OR", "PA", "PR", "RI", "SC", "SD", 
            "TN", "TX", "UT", "VT", "VI", "VA", "WA", "WV", "WI", "WY"]
 # DI = 'Diamond Princess', GP = 'Grand Princess', 
 # GU = 'Guam', MP = 'Northern Mariana Islands'


basePath = os.path.dirname(os.path.abspath(__file__))
# Split tweets_id.txt file to smaller files with less size
PATH = TWEETS_FILE_TXT
print(PATH)
with open(PATH ,'r') as fp:
    id_all = fp.readlines()

n = len(id_all)
print("Total number of Covid-19 ids = {}".format(n))
partition = n//200000 + 1

for i in range(partition): 
    # Split index from the last part to the prior part 5,4,3,2,1
    start_index = (partition-i-1) * 200000
    id_partition = id_all[start_index:]
    n_partition = len(id_partition)
    _save_to_txt(id_partition, i)
    print("Complete saving the {}th dataset with length = {}".format(i, n_partition))
    id_all = id_all[:start_index]