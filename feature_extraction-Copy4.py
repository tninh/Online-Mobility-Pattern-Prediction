
# coding: utf-8

# In[7]:



# from hmmlearn import hmm
# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from collections import defaultdict
import datetime
from time import time
import math

import tensorflow as tf

# from tf.keras.models import Sequential  # This does not work!
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, LSTM
from tensorflow.python.keras.optimizers import Adam
# from tensorflow.python.keras.preprocessing.text import Tokenizer
# from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import to_categorical

config = tf.ConfigProto()
config.gpu_options.allow_growth = True




# In[11]:


df = pd.read_csv('../all_data.csv')
df = df[['Client Username', 'Association Time', 'Map Location', 'Session Duration', 'Client MAC Address']]
df = df.assign(id=(df['Client Username'] + '_' + df['Client MAC Address']).astype('category').cat.codes)


# In[ ]:





# In[12]:


df = df.drop(['Client Username', 'Client MAC Address'], axis=1)
df.rename(columns={'id':'Client Username'}, inplace=True)
df = df[['Client Username', 'Association Time', 'Map Location', 'Session Duration']]


# In[13]:


# recode map location
map_dict = {}
count = 0
for i in df['Map Location'].unique():
    map_dict[i] = count
    count += 1
df['Map Location'] = df['Map Location'].map(map_dict)


# In[14]:


# df[df['Map Location']==64]


# In[ ]:





# In[15]:


def timer(func):
    # this part is for timer ignore
    def wrapper(*arg):
        before = time()
        rv = func(*arg)
        after = time()
        print('Elapsed: ', after-before)
        return rv
    return wrapper

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

def convert_to_datetime(time):
    temp = time.split()
    d_time = datetime.datetime(year=int(temp[5]), month=convertMonth(temp[1]), day=int(temp[2]), hour=int(temp[3].split(':')[0]), minute=int(temp[3].split(':')[1]), second=int(temp[3].split(':')[2]))
    return d_time

# data_list has same number of row as df.
# Each row [userid, map_loc, associate_time, associate_time+sess_dur, sess_dur].
def create_data_list(df):
    data_list = []
    c=0
    for i in range(df.shape[0]):
        if c%500000 == 0:
            print(c)
        sess = convert_to_timeslot(df.iloc[i,3])
        s_time = convert_to_datetime(df.iloc[i,1])
        e_time = s_time + sess
        data_list.append([df.iloc[i,0], df.iloc[i,2], s_time, e_time, sess])
        c+=1
        
    data_list = sorted(data_list, key=lambda element: (element[0], element[2]))
    data_list1 = sorted(data_list, key=lambda element: (element[2]))
    first = data_list1[0][2]
    last = data_list1[-1][2]
    return data_list, first, last

# add up the session duration when the previous row has the same userid and location. 
# (this happen when you disconnect from the ap and reconnect again or your wifi session expire, etc.)
def preprocess_data_list(df, data_list):
    for i in reversed(range(1, df.shape[0])):
        if data_list[i][0] == data_list[i-1][0] and data_list[i][1] == data_list[i-1][1]:
#             if data_list[i][2] - data_list[i-1][3] <= datetime.timedelta(hours=2):
            data_list[i-1][4] += data_list[i][4]
#             else: print(i)
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


# user_dict's key are each userid and value is a list of all rows of each user in final_data_list
def create_user_dict(final_data_list):
    user_dict = defaultdict(list)
    for row in final_data_list:
        user_dict[row[0]].append([row[2], row[1], row[4]])
        
#     for usr in user_dict:
#         user_dict[usr].sort()

    return user_dict

# 
def remove_jitter(final_data_list):
    res = [final_data_list[0]]
    i = 1
    mem = 0
    while i < len(final_data_list):
        cur_usrid = final_data_list[i][0]
        cur_loc = final_data_list[i][1]
        cur_sess = final_data_list[i][4]
        cur_end_time = final_data_list[i][2] + final_data_list[i][4]
        cont_time = final_data_list[i][2] - (res[-1][2] + res[-1][4]) <= datetime.timedelta(minutes=1)
        if cur_usrid == res[-1][0] and cur_sess <= datetime.timedelta(minutes=1) and cont_time:


            if cur_loc != res[-1][1] and mem == 0:
                mem = i

            elif cur_loc == res[-1][1]:
                
#                 print(i, cur_loc, cur_end_time, res[-1][1], res[-1][2])

#                 print(cur_end_time - res[-1][2])
                res[-1][4] = cur_end_time - res[-1][2]
#                 print(res[-1][4])
                mem=0
                
            else: pass
        elif cur_usrid == res[-1][0] and cur_sess <= datetime.timedelta(minutes=1) and not cont_time:
            if cur_loc != res[-1][1] and mem == 0:
                res.append(final_data_list[i])

            elif cur_loc == res[-1][1]:
                res[-1][4] = cur_end_time - res[-1][2]
                mem=0
                
            else:                 
                for j in range(mem, i):
                    res.append(final_data_list[j])
                res.append(final_data_list[i])
                mem=0
        else:
            if mem != 0:
                for j in range(mem, i):
                    res.append(final_data_list[j])
                res.append(final_data_list[i])
                mem=0
            else: res.append(final_data_list[i])
        
        i+=1

    return res

@timer
def preprocess_and_create_user_dict(df):
    data_list, start_ts, end_ts= create_data_list(df)
    final_data_list = preprocess_data_list(df, data_list)
    print(len(final_data_list), len(data_list))
    final_data_list = remove_jitter(final_data_list)
    print(len(final_data_list), len(data_list))
    user_dict = create_user_dict(final_data_list)
    return user_dict, start_ts, end_ts


# In[16]:


user_dict, start_ts, end_ts = preprocess_and_create_user_dict(df)


# In[17]:


a = [i for i in user_dict if len(user_dict[i]) < 500]


# In[18]:


# user_dict1 = {}
# for i in user_dict:
#     if not i in a:
#         user_dict1[i] = user_dict[i]


# In[ ]:





# In[19]:


t = map(user_dict.pop, a)


# In[20]:


for i in range(len(a)):
    try:
        next(t)
    except:
        print('fail')


# In[21]:


len(user_dict)


# In[ ]:





# In[ ]:





# In[ ]:





# In[22]:


def convert_to_timeslot1(d_time):
    slot = (d_time.hour * 60 + d_time.minute)/60
    return int(slot) if slot < int(slot) + 0.5 else int(slot) + 1

def create_data(user_dict):
    input_data = []
    c = 0
    for usr in user_dict:
        c+=1
        if c % 5000 == 0: print(c)
        for j in range(len(user_dict[usr] )-1):
            slot = convert_to_timeslot1(user_dict[usr][j][0])
            loc = user_dict[usr][j][1]
            input_data.append([usr, str(slot) + '_' + str(loc)])
            
    return input_data


# In[23]:


input_data = create_data(user_dict)


# In[24]:


df1 = pd.DataFrame(data=input_data, columns=['userID', 'arrival_slot and location'])


# In[25]:


df1.head()


# In[ ]:





# In[26]:


# recode map location
map_dict_time_loc = {}
count = 1
for i in df1['arrival_slot and location'].unique():
    map_dict_time_loc[i] = count
    count += 1
df1['arrival_slot and location'] = df1['arrival_slot and location'].map(map_dict_time_loc)


# In[27]:


num_loc_time = max(df1['arrival_slot and location'])


# In[28]:


input_dict = defaultdict(list)
for i in range(df1.shape[0]):
    if i % 500000 == 0: print(i)
    input_dict[df1.loc[i]['userID']].append(df1.loc[i]['arrival_slot and location'])
    


# In[ ]:





# In[29]:


X_data = []
y_data = []
window = 4

for usr in input_dict:
    size = len(input_dict[usr])
    
    if size == 0: continue
    elif size <= window:
        temp = []
        for _ in range(window - size):
            temp.append(0)
        for i in range(size):
            temp.append(input_dict[usr][i])
        
        X_data.append(temp)
        try:
            y_data.append(input_dict[usr][-1])
        except:
            print(usr)
    else:
        for i in range(size - (window + 1)):
            temp=[]
            for j in range(window+1):
                if j == window:
                    y_data.append(input_dict[usr][i+j])
                else: 
                    temp.append(input_dict[usr][i+j])
            X_data.append(temp)
            


# In[30]:


X_data = np.array(X_data)
y_data = np.array(y_data)


# In[25]:


# X_data


# In[ ]:


embedding_size=4


# In[121]:


# %%time
# result = model.evaluate(x_test_pad, y_test)


# In[122]:


# print("Accuracy: {0:.2%}".format(result[1]))


# In[33]:


model = Sequential()
model.add(Embedding(input_dim=num_loc_time,
                    output_dim=embedding_size,
                    input_length=window,
                    name='layer_embedding'))

# model.add(LSTM(8, return_sequences=True,
#                input_shape=(window, num_loc_time)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(4))  # return a single vector of dimension 32
model.add(Dense(4036, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-10),
              metrics=['accuracy'])

# # Generate dummy training data
# x_train = np.random.random((1000, timesteps, data_dim))
# y_train = np.random.random((1000, num_classes))

# Generate dummy validation data
# x_val = np.random.random((100, timesteps, data_dim))
# y_val = np.random.random((100, num_classes))

# model.fit(X_data, y_data, epochs=2, batch_size=64, verbose=2)

# model.fit(x_train, y_train,
#           batch_size=64, epochs=5,
#           validation_data=(x_val, y_val))


# In[36]:



input_data = 0
user_dict = 0
df = 0
df1 = 0

encoded = to_categorical(y_data)


# In[ ]:


model.fit(X_data, encoded, validation_split=0.10, epochs=2, batch_size=32, verbose=2)


# In[34]:


num_loc_time


# In[ ]:





# In[ ]:





# In[ ]:




