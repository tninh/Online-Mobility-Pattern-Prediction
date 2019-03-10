#!/usr/bin/env python
# coding: utf-8

# In[27]:


# from hmmlearn import hmm
import numpy as np
import pandas as pd
from collections import defaultdict
import datetime
import math
# import pykovy-master/ 
#from pykovy.src.pykovy import chain
import matplotlib.pyplot as plt


# In[28]:


df = pd.read_csv('data_sample/all_data.csv')
df = df[['Client Username', 'Association Time', 'Map Location', 'Session Duration']]


# In[29]:


# re-code userid
df['Client Username'] = pd.Categorical(df['Client Username'])
df['Client Username'] = df['Client Username'].cat.codes
# df = df.drop(['Client Username'], axis=1)

# recode map location
map_dict = {}
count = 0
for i in df['Map Location'].unique():
    map_dict[i] = count
    count += 1
df['Map Location'] = df['Map Location'].map(map_dict)


# In[30]:


df


# In[31]:


result = {}
result["client_username"] = [-1] * 3
result["client_username"][2] = 1
print(result)
print(24 % 24)


# In[32]:


#from datetime import datetime
def convertMonth(month):
    if month == 'Jan': return 1
    elif month == 'Feb': return 2
    elif month == 'Mar': return 3
    elif month == 'Apr': return 4
    elif month == 'May': return 5
    elif month == 'Jun': return 6
    elif month == 'Jul': return 7
    elif month == 'Aug': return 8
    elif month == 'Sep': return 9
    elif month == 'Oct': return 10
    elif month == 'Nov': return 11
    elif month == 'Dec': return 12
    else: print('error input is not correct')
        
def convert_to_timeslot(duration):
    temp = duration.split()
    if len(temp) == 2: sess_duration = int(temp[0])
    elif len(temp) == 4: sess_duration = int(temp[0]) * 60 + int(temp[2])
    elif len(temp) == 6: sess_duration = (int(temp[0]) * 60 + int(temp[2])) * 60 + int(temp[4])
    else: print('error')
    return datetime.timedelta(seconds=sess_duration)

def convert_duration_to_seconds(duration):
    temp = duration.split()
    if len(temp) == 2: sess_duration = int(temp[0])
    elif len(temp) == 4: sess_duration = int(temp[0]) * 60 + int(temp[2])
    elif len(temp) == 6: sess_duration = (int(temp[0]) * 60 + int(temp[2])) * 60 + int(temp[4])
    else: print('error')
    return int(sess_duration)

def convert_to_datetime(time):
    temp = time.split()
    d_time = datetime.datetime(year=int(temp[5]), month=convertMonth(temp[1]), day=int(temp[2]), hour=int(temp[3].split(':')[0]), minute=int(temp[3].split(':')[1]), second=int(temp[3].split(':')[2]))
    return d_time

print(convert_to_datetime('Sat Feb 24 13:42:56 PST 2018'))
print(convert_duration_to_seconds('7 hrs 25 min 57 sec'))


# In[33]:


#from datetime import datetime
# convert a time format
# Sat Feb 24 13:42:56 PST 2018
# to timestamp value
def convert_time_to_timestamp(input_time):
    datetime_format = convert_to_datetime(input_time)
    dt_obj = datetime.datetime.strptime(str(datetime_format),
                                        '%Y-%m-%d %H:%M:%S')
    return int(dt_obj.timestamp())
print(convert_time_to_timestamp('Sat Feb 24 13:42:56 PST 2018'))


# In[34]:


def convert_datetime_to_timestamp(input_time):
    return int(input_time.timestamp())
print(convert_datetime_to_timestamp(datetime.datetime(2018, 2, 24, 12, 2, 49)))


# In[35]:


def convert_timedelta_to_seconds(input_time):
    return int(input_time.total_seconds())
print(convert_timedelta_to_seconds(datetime.timedelta(seconds=1927)))


# In[36]:


def convert_timestamp_to_index(input_ts, start_ts, window_in_minutes):
    index = math.floor((input_ts - start_ts) / (window_in_minutes * 60))
    return index
print(convert_timestamp_to_index(1519508576, 1519462800, 60))


# In[37]:


# data_list has same number of row as df.
# Each row [userid, map_loc, associate_time, associate_time+sess_dur, sess_dur].
def create_data_list(df):
    data_list = []
    c=0
    for i in range(df.shape[0]):
        #if c%500000 == 0:
        #    print(c)
        sess = convert_to_timeslot(df.iloc[i,3])
        s_time = convert_to_datetime(df.iloc[i,1])
        e_time = s_time + sess
        data_list.append([df.iloc[i,0], df.iloc[i,2], s_time, e_time, sess])
        c+=1
        
    data_list = sorted(data_list, key=lambda element: (element[0], element[2]))
    return data_list
data_list = create_data_list(df)
#data_list


# In[38]:


# add up the session duration when the previous row has the same userid and location. 
# (this happen when you disconnect from the ap and reconnect again or your wifi session expire, etc.)
def preprocess_data_list(df, data_list):
    for i in reversed(range(1, df.shape[0])):
        if data_list[i][0] == data_list[i-1][0] and data_list[i][1] == data_list[i-1][1]: #and data_list[i][2] - data_list[i-1][3] <= timedelta(hours=5):
            data_list[i-1][4] += data_list[i][4]

    # loop each row, remove the next concecutive rows has the same userid and map_loc with the current row.
    # Because they are useless now
    final_data_list = []
    i = 0
    while i < df.shape[0]-1:
        if data_list[i][0] == data_list[i+1][0] and data_list[i][1] == data_list[i+1][1]:
            final_data_list.append(data_list[i])
            c = 2
            while i+c <= df.shape[0]-1 and data_list[i][0] == data_list[i+c][0] and data_list[i][1] == data_list[i+c][1]:
                c += 1
            i += c

        else:
            final_data_list.append(data_list[i])
            i += 1

    if not (final_data_list[-1][0] == data_list[-1][0] and final_data_list[-1][1] == data_list[-1][1]):
        final_data_list.append(data_list[-1])

    return final_data_list

final_data_list = preprocess_data_list(df, data_list)
#final_data_list


# In[39]:


# user_dict's key are each userid and value is a list of all rows of each user in final_data_list
def create_user_dict(final_data_list):
    user_dict = defaultdict(list)
    for row in final_data_list:
        user_dict[row[0]].append([row[2], row[1], row[4]])
        
#     for usr in user_dict:
#         user_dict[usr].sort()

    return user_dict
#user_dict = create_user_dict(final_data_list)
#user_dict


# In[40]:


def remove_jitter(final_data_list):
    pass


# In[41]:


def preprocess_and_create_user_dict(df):
    data_list = create_data_list(df)
    final_data_list = preprocess_data_list(df, data_list)
    #print(len(final_data_list), len(data_list))
#     final_data_list = remove_jitter(final_data_list)
    #print(len(final_data_list), len(data_list))
    user_dict = create_user_dict(final_data_list)
    return user_dict
#user_dict = preprocess_and_create_user_dict(df)
#user_dict


# In[42]:

'''
for usr in user_dict:
    print(usr)
    for i in range(0, len(user_dict[usr])):
        print(user_dict[usr][i][0])
        print(convert_datetime_to_timestamp(user_dict[usr][i][0]))
        print(user_dict[usr][i][1])
        print(convert_timedelta_to_seconds(user_dict[usr][i][2]))
'''

# In[18]:


# create a time slot 2d array each row is a user and each cols is a timeslot.
# for a particular user if there is no information on a particular time slot append(-1)
# this function is not finished
# Input:
# - df: dataframe
# - userdict: list of user id
# - start_timestamp: start time timestamp
# - end_timestamp: end time timestamp
# - window_in_minute: time window, use 60 (1 hour) first for simplicity
# 
# Output:
# - result: a hashmap (dictionary) where each map item is a 1D array where array 
#           indexes are indexed timeslot and values are "Map Location" (integer value)
#           -1 if there's no information for that timeslot
def create_time_slot_data(df, user_dict, start_timestamp, end_timestamp, window_in_minutes):
    # calculate number of index in each array
    # Formula:
    # number_of_time_entry = (diff of timestamp in second) / (number of seconds in that time window)
    number_of_time_entry = int((end_timestamp - start_timestamp) / (window_in_minutes * 60))
    
    # Addumption: start date at 0:0:0, end date at 23:59:59
    # 86400 = seconds in a day
    number_of_days = int((end_timestamp - start_timestamp) / 86400)
    
    number_of_time_entry_per_day = int(86400 / (window_in_minutes * 60))
    #print(number_of_time_entry_per_day)
    result = {}
    #for i in range(df.shape[0]):
    for usr in user_dict:
        client_username = usr
        for i in range(0, len(user_dict[usr])):
        #print(user_dict[usr][i][0])
        #print(convert_datetime_to_timestamp(user_dict[usr][i][0]))
        #print(user_dict[usr][i][1])
        #print(convert_timedelta_to_seconds(user_dict[usr][i][2]))
        #client_username = df.iloc[i,0]
            association_time = convert_datetime_to_timestamp(user_dict[usr][i][0])
            map_location = user_dict[usr][i][1]
            session_duration = convert_timedelta_to_seconds(user_dict[usr][i][2])
        
            # 1/ If there's no entry for this user in result dict
            #    create a new 1d array with all -1 
            #    then change values base on association time and session duration
            # 2/ If user already in the result dict
            #    then change values base on association time and session duration
            # Caveats: there's maybe duplication, ignore it for now
            if client_username not in result:
                #time_entry_ts = start_timestamp
                #while time_entry_ts < end_timestamp:
                result[client_username] = [-1] * number_of_time_entry
                    #time_entry_ts += 86400
            # convert association time to timestamp
            # and calculate the total sesstion in timestamp
            association_time_ts = association_time
            end_association_time_ts = association_time_ts + session_duration
            if (association_time_ts > end_timestamp): 
                continue
            # TODO:
            # Need to handle the case where association timestamp < start_timestamp
            # in that case, some array entry might change
            start_index = convert_timestamp_to_index(association_time_ts, start_timestamp, window_in_minutes)
            end_index = convert_timestamp_to_index(end_association_time_ts, start_timestamp, window_in_minutes)
            # mark all the index from start index to end index to map location
            for i in range(start_index, end_index):
                result[client_username][i] = map_location
       
    #print(result)
    # Now we have a result dictionary without date
    # Do some more calculation to convert this to a dict with
    # {userid, date, list of timeslot}
    output = {}
    for user in result:
        output[user] = {}
        date_ts = start_timestamp
        output[user][date_ts] = [-1] * number_of_time_entry_per_day
        for i in range(0, number_of_time_entry):
            if (i != 0) and (i % number_of_time_entry_per_day == 0):
                date_ts += 86400
                output[user][date_ts] = [-1] * number_of_time_entry_per_day
            output[user][date_ts][i % number_of_time_entry_per_day] = result[user][i]
                
    return output

#output = create_time_slot_data(df, user_dict, 1519376400, 1519545599, 60)
#print(pd.DataFrame(output))


# In[19]:


final_data_list[0][1]


# In[20]:


# helper function to lookup a user map location 
# on a given time
def lookup_map_location_by_time(data, client_username, input_time, start_time, window_in_minutes):
    time_ts = convert_time_to_timestamp(input_time)
    start_ts = convert_time_to_timestamp(start_time)
    index = convert_timestamp_to_index(time_ts, start_ts, window_in_minutes)
    return data[client_username][index]

#print(lookup_map_location_by_time(output, 133, 'Sat Feb 24 01:20:38 PST 2018',
#                                 'Sat Feb 23 01:00:00 PST 2018', 60))


# In[ ]:


# create a training data from final_data_list
# where each data is an array of 199 entry,
# 198 entry represent the weight on last X history 
# of each location that the user on
# and entry 199th is the prediction
NO_AP_LOCATION_VALUE = 199
def create_training_data(final_data_list, number_of_history):
    output = []
    prev_user_1 = -1
    prev_user_2 = -1
    prev_user_3 = -1
    # list is already sorted by user
    for idx, val in enumerate(final_data_list):
        location_id = val[1]
        time_in_s = convert_timedelta_to_seconds(val[4])
        user = val[0]
        entry = [0] * 199
        if idx == 0:
            # first index
            # location: time spent as weight
            # prediction: -1
            #entry[location_id] = time_in_s
            entry[198] = -1
            prev_user_1 = user
        elif idx == 1:
            # second index
            # use this index as prediction for the last index if same user
            if prev_user_1 != user:
                #entry = [0] * 199
                entry[198] = -1
            else:
                print(idx)
                print(val[idx-1])
                prev_loc_1 = final_data_list[idx-1][1]
                entry[prev_loc_1] = time_in_s
                entry[198] = -1
            prev_user_2 = prev_user_1
            prev_user_1 = user
        elif idx == 2:
            if prev_user_1 != user:
                #entry = [0] * 199
                entry[198] = -1
            else:
                prev_loc_1 = final_data_list[idx-1][1]
                entry[198] = -1
                if prev_user_2 != user:
                    prev_loc_2 = 0
                else:
                    total_time = convert_timedelta_to_seconds(final_data_list[idx-2][4]) + convert_timedelta_to_seconds(final_data_list[idx-1][4])
                    prev_loc_2 = final_data_list[idx-2][1]
                    entry[prev_loc_2] = convert_timedelta_to_seconds(final_data_list[idx-2][4])
                    entry[prev_loc_1] = entry[prev_loc_2] + convert_timedelta_to_seconds(final_data_list[idx-2][4])/total_time
            prev_user_3 = prev_user_2
            prev_user_2 = prev_user_1
            prev_user_1 = user
        else:
            if prev_user_1 != user:
                #entry = [0] * 199
                entry[198] = -1
            elif prev_user_2 != user:
                # only 1 entry
                prev_loc_1 = final_data_list[idx-1][1]
                entry[prev_loc_1] = time_in_s
                entry[198] = -1
            elif prev_user_3 != user:
                # only 2 entries
                prev_loc_1 = final_data_list[idx-1][1]
                prev_loc_2 = final_data_list[idx-2][1]
                total_time = convert_timedelta_to_seconds(final_data_list[idx-2][4]) + convert_timedelta_to_seconds(final_data_list[idx-1][4])
                entry[prev_loc_2] = convert_timedelta_to_seconds(final_data_list[idx-2][4])
                entry[prev_loc_1] = entry[prev_loc_2] + convert_timedelta_to_seconds(final_data_list[idx-2][4])/total_time
                entry[198] = -1
            else:
                # have all 3 entries
                prev_loc_1 = final_data_list[idx-1][1]
                prev_loc_2 = final_data_list[idx-2][1]
                prev_loc_3 = final_data_list[idx-3][1]
                total_time = convert_timedelta_to_seconds(final_data_list[idx-3][4]) +                             convert_timedelta_to_seconds(final_data_list[idx-2][4]) +                             convert_timedelta_to_seconds(final_data_list[idx-1][4])
                entry[prev_loc_3] = convert_timedelta_to_seconds(final_data_list[idx-3][4])
                entry[prev_loc_2] = entry[prev_loc_3] + convert_timedelta_to_seconds(final_data_list[idx-3][4])/total_time
                entry[prev_loc_1] = entry[prev_loc_2] + (convert_timedelta_to_seconds(final_data_list[idx-3][4])/total_time) +                                     (convert_timedelta_to_seconds(final_data_list[idx-2][4])/total_time)
                entry[198] = location_id
            prev_user_3 = prev_user_2
            prev_user_2 = prev_user_1
            prev_user_1 = user
        output.append(entry)
    return output

#output = create_training_data(final_data_list, 1)

#print(output)


# In[43]:


# create a training data from final_data_list
# where each data is an array of 199 entry,
# 198 entry represent the weight on last X history 
# of each location that the user on
# and entry 199th is the prediction
NO_AP_LOCATION_VALUE = 199
def create_training_data_2(final_data_list, number_of_history):
    output = {"x":[], "y":[]}
    prev_user_1 = -1
    prev_user_2 = -1
    prev_user_3 = -1
    # list is already sorted by user
    for idx, val in enumerate(final_data_list):
        location_id = val[1]
        time_in_s = convert_timedelta_to_seconds(val[4])
        user = val[0]
        entry = [0] * 199
        if idx == 0:
            # first index
            # location: time spent as weight
            # prediction: -1
            #entry[location_id] = time_in_s
            entry[198] = -1
            prev_user_1 = user
        elif idx == 1:
            # second index
            # use this index as prediction for the last index if same user
            if prev_user_1 != user:
                #entry = [0] * 199
                entry[198] = -1
            else:
                print(idx)
                print(val[idx-1])
                prev_loc_1 = final_data_list[idx-1][1]
                entry[prev_loc_1] = time_in_s
                entry[198] = -1
            prev_user_2 = prev_user_1
            prev_user_1 = user
        elif idx == 2:
            if prev_user_1 != user:
                #entry = [0] * 199
                entry[198] = -1
            else:
                prev_loc_1 = final_data_list[idx-1][1]
                entry[198] = -1
                if prev_user_2 != user:
                    prev_loc_2 = 0
                else:
                    total_time = convert_timedelta_to_seconds(final_data_list[idx-2][4]) + convert_timedelta_to_seconds(final_data_list[idx-1][4])
                    prev_loc_2 = final_data_list[idx-2][1]
                    entry[prev_loc_2] = convert_timedelta_to_seconds(final_data_list[idx-2][4])
                    entry[prev_loc_1] = entry[prev_loc_2] + convert_timedelta_to_seconds(final_data_list[idx-2][4])/total_time
            prev_user_3 = prev_user_2
            prev_user_2 = prev_user_1
            prev_user_1 = user
        else:
            if prev_user_1 != user:
                #entry = [0] * 199
                entry[198] = -1
            elif prev_user_2 != user:
                # only 1 entry
                prev_loc_1 = final_data_list[idx-1][1]
                entry[prev_loc_1] = time_in_s
                entry[198] = -1
            elif prev_user_3 != user:
                # only 2 entries
                prev_loc_1 = final_data_list[idx-1][1]
                prev_loc_2 = final_data_list[idx-2][1]
                total_time = convert_timedelta_to_seconds(final_data_list[idx-2][4]) + convert_timedelta_to_seconds(final_data_list[idx-1][4])
                entry[prev_loc_2] = convert_timedelta_to_seconds(final_data_list[idx-2][4])
                entry[prev_loc_1] = entry[prev_loc_2] + convert_timedelta_to_seconds(final_data_list[idx-2][4])/total_time
                entry[198] = -1
            else:
                # have all 3 entries
                prev_loc_1 = final_data_list[idx-1][1]
                prev_loc_2 = final_data_list[idx-2][1]
                prev_loc_3 = final_data_list[idx-3][1]
                total_time = convert_timedelta_to_seconds(final_data_list[idx-3][4]) +                             convert_timedelta_to_seconds(final_data_list[idx-2][4]) +                             convert_timedelta_to_seconds(final_data_list[idx-1][4])
                entry[prev_loc_3] = convert_timedelta_to_seconds(final_data_list[idx-3][4])
                entry[prev_loc_2] = entry[prev_loc_3] + convert_timedelta_to_seconds(final_data_list[idx-3][4])/total_time
                entry[prev_loc_1] = entry[prev_loc_2] + (convert_timedelta_to_seconds(final_data_list[idx-3][4])/total_time) +                                     (convert_timedelta_to_seconds(final_data_list[idx-2][4])/total_time)
                entry[198] = location_id
            prev_user_3 = prev_user_2
            prev_user_2 = prev_user_1
            prev_user_1 = user
        output["x"].append(entry[0:197])
        output["y"].append(entry[198])
    return output

output = create_training_data_2(final_data_list, 1)

#Copy output to file
import csv
with open("output.csv", "w") as f:
    #writer = csv.writer(f, delimiter=',')
    for item in range(0, len(output["x"])):
        f.write(str(output["x"][item]) + "," + str(output["y"][item]) + '\n')
        #writer.writerows([output["x"][item], output["y"][item]])



#print(output["x"])
#print(output["y"])
#print(output)
#print("Len of x is " + str(len(output["x"])))
#print("Len of y is " + str(len(output["y"])))


# In[ ]:




