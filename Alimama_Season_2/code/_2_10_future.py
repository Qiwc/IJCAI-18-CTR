
# coding: utf-8

# In[1]:


import os
import pickle
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import load_pickle, dump_pickle, raw_data_path, feature_data_path
from utils import extract_ctr


# In[3]:


def get_before_2min(s):
    time_now,times = s.split('-')
    time_one_hour_before = int(time_now) - 120
    times = times.split(':')
    
    count = 0
    for t in times:
        if (int(t)<int(time_now))&(int(t)>=int(time_one_hour_before)):
            count = count + 1
    return count
def get_before_15min(s):
    time_now,times = s.split('-')
    time_one_hour_before = int(time_now) - 1000
    times = times.split(':')
    
    count = 0
    for t in times:
        if (int(t)<int(time_now))&(int(t)>=int(time_one_hour_before)):
            count = count + 1
    return count

def get_before_1hour(s):
    time_now,times = s.split('-')
    time_one_hour_before = int(time_now) - 3600
    times = times.split(':')
    
    count = 0
    for t in times:
        if (int(t)<int(time_now))&(int(t)>=int(time_one_hour_before)):
            count = count + 1
    return count

def gen_user_feature_before():

    all_data = load_pickle(raw_data_path + 'all_data_4567.pkl')

    feature_list = ['category2_label', 'category3_label',
                    'shop_id', 'item_id', 'item_brand_id', 
                   'item_sales_level_bin', 'item_price_level_bin',
                   'item_property_topic_k_15',
                   ]

    for feature in tqdm(feature_list):

        feature_path = feature_data_path + '_2_10_user_' + feature + '_before.pkl'
        if os.path.exists(feature_path):
            print('found ' + feature_path)   
        else:
            print('generating '+feature_path)

            before_2min = 'user_' + feature + '_before_2min'
            before_15min = 'user_' + feature + '_before_15min'
            before_1hour = 'user_' + feature + '_before_1hour'

            t1 = data[['user_id', feature, 'context_timestamp']]
            t1.context_timestamp = t1.context_timestamp.astype('str')
            t1 = t1.groupby(['user_id', feature])['context_timestamp'].agg(lambda x:':'.join(x)).reset_index()
            t1.rename(columns={'context_timestamp':'times'},inplace=True)

            t2 = data[['user_id', feature, 'context_timestamp']]
            t2 = pd.merge(t2, t1, on=['user_id', feature], how='left')
            t2['time_now'] = t2.context_timestamp.astype('str') + '-' + t2.times

            t2[before_2min] = t2.time_now.apply(get_before_2min)
            t2[before_15min] = t2.time_now.apply(get_before_15min)
            t2[before_1hour] = t2.time_now.apply(get_before_1hour)


            t3 = t2[[before_2min, before_15min,before_1hour]] 

            dump_pickle(t3, feature_path)


def add_user_feature_before(data):


    feature_list = ['category2_label', 'category3_label',
                    'shop_id', 'item_id', 'item_brand_id', 
                   'item_sales_level_bin', 'item_price_level_bin',
                   'item_property_topic_k_15',
                   ]


    for feature in tqdm(feature_list):
        feature_path = feature_data_path + '_2_10_user_' + feature + '_before.pkl'
        if not os.path.exists(feature_path):
            gen_user_feature_before()

        user_feature_click_rank_global = load_pickle(feature_path)
        data = data.join(user_feature_click_rank_global)

    return data


# In[2]:


def get_feature_2min(s):
    time_now,times = s.split('-')
    time_one_hour_after = int(time_now) + 120
    times = times.split(':')
    
    count = 0
    for t in times:
        if (int(t)>int(time_now))&(int(t)<=int(time_one_hour_after)):
            count = count + 1
    return count
def get_feature_15min(s):
    time_now,times = s.split('-')
    time_one_hour_after = int(time_now) + 1000
    times = times.split(':')
    
    count = 0
    for t in times:
        if (int(t)>int(time_now))&(int(t)<=int(time_one_hour_after)):
            count = count + 1
    return count


# In[4]:


def gen_user_feature_future():

    all_data = load_pickle(raw_data_path + 'all_data_4567.pkl')

    feature_list = ['category2_label', 'category3_label',
                    'shop_id', 'item_id', 'item_brand_id', 
                   'item_sales_level_bin', 'item_price_level_bin',
                   'item_property_topic_k_15',
                   ]

    for feature in tqdm(feature_list):

        feature_path = feature_data_path + '_2_10_user_' + feature + '_future.pkl'
        if os.path.exists(feature_path):
            print('found ' + feature_path)   
        else:
            print('generating '+feature_path)

            future_2min = 'user_' + feature + '_future_2min'
            future_15min = 'user_' + feature + '_future_15min'

            t1 = data[['user_id', feature, 'context_timestamp']]
            t1.context_timestamp = t1.context_timestamp.astype('str')
            t1 = t1.groupby(['user_id', feature])['context_timestamp'].agg(lambda x:':'.join(x)).reset_index()
            t1.rename(columns={'context_timestamp':'times'},inplace=True)

            t2 = data[['user_id', feature, 'context_timestamp']]
            t2 = pd.merge(t2, t1, on=['user_id', feature], how='left')
            t2['time_now'] = t2.context_timestamp.astype('str') + '-' + t2.times

            t2[future_2min] = t2.time_now.apply(get_feature_2min)
            t2[future_15min] = t2.time_now.apply(get_feature_15min)

            t3 = t2[[future_2min, future_15min,]] 

            dump_pickle(t3, feature_path)


def add_user_feature_future(data):


    feature_list = ['category2_label', 'category3_label',
                    'shop_id', 'item_id', 'item_brand_id', 
                   'item_sales_level_bin', 'item_price_level_bin',
                   'item_property_topic_k_15',
                   ]


    for feature in tqdm(feature_list):
        feature_path = feature_data_path + '_2_10_user_' + feature + '_future.pkl'
        if not os.path.exists(feature_path):
            gen_user_feature_future()

        user_feature_click_rank_global = load_pickle(feature_path)
        data = data.join(user_feature_click_rank_global)

    return data


# In[ ]:


if __name__ =='__main__':
    
    data = load_pickle(raw_data_path + 'all_data_4567.pkl')   
#     data = add_user_feature_future(data)
    data = add_user_feature_before(data)
    
    print(data.columns)


