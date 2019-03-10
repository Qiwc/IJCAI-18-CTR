
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


# In[2]:


def get_gap_before(s):
    time_now,times = s.split('-')
    times = times.split(':')
    gaps = []
    for t in times:
        this_gap = int(time_now) - int(t)
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        return -1
    else:
        return min(gaps)
        
def get_gap_after(s):
    time_now,times = s.split('-')
    times = times.split(':')
    gaps = []
    for t in times:
        this_gap = int(t) - int(time_now)
        if this_gap>0:
            gaps.append(this_gap)
    if len(gaps)==0:
        return -1
    else:
        return min(gaps)
    
def get_true_rank(s):
    time_now,times = s.split('-')
    times = times.split(':')
    gaps = []
    for t in times:
        this_gap = int(time_now) - int(t)
        if this_gap > 0:
            gaps.append(this_gap)
    return len(gaps)


# ## 用户当前点击在一天中的排序、与前后点击记录的时间差、与当天首位点击记录的时间差

# In[3]:


def user_day_rank_mapper(row):
    '''

    return:
        0: 只有一次点击
        1: 第一次点击
        2: 非首尾点击
        3: 最后一次点击

    '''
    if row['user_click_day'] <= 1:
        return 0
    elif row['user_first_click_day'] > 0:
        return 1
    elif row['user_last_click_day'] > 0:
        return 3
    else:
        return 2


def gen_user_click_rank_day():
    '''生成用户当前点击在一天中的排序

    file_name: user_click_day_rank.pkl

    features:
        'user_click_rank_day',
        'user_click_interval_first_day',
        'user_click_interval_last_day',
        'user_click_time_gap_before',
        'user_click_time_gap_after',
        'user_click_true_rank_day',

    '''

    data = load_pickle(raw_data_path + 'all_data_4567.pkl')

    feature_path = feature_data_path +'_2_3_'+ 'user_click_rank_day.pkl'

    print('generating '+feature_path)

    user_click_day = data.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_click_day'})

    data = pd.merge(data, user_click_day, how='left',
                        on=['user_id', 'day'])


    # 用户在一天内点击该物品的时间戳排序
    sorted_data = data.sort_values(by=['user_id', 'day', 'context_timestamp'], ascending=True)

    # 保留用户在一天内首次点击的记录
    first = sorted_data.drop_duplicates(['user_id', 'day']).copy()

    # 保留用户在一天内最后点击的记录
    last = sorted_data.drop_duplicates(['user_id', 'day'], keep='last').copy()
    
    # 用户一天内首次点击的时间戳
    user_first_click_time_day = sorted_data.groupby(['user_id', 'day'])['context_timestamp'].first().reset_index().rename(columns={'context_timestamp': 'user_first_click_time_day'})
        
    # 用户一天内最后点击的时间戳
    user_last_click_time_day = sorted_data.groupby(['user_id', 'day'])['context_timestamp'].last().reset_index().rename(columns={'context_timestamp': 'user_last_click_time_day'})


    first['user_first_click_day'] = 1
    first = first[['user_first_click_day']]
    data = data.join(first)

    last['user_last_click_day'] = 1
    last = last[['user_last_click_day']]
    data = data.join(last)
    
    data[['user_first_click_day', 'user_last_click_day']] = data[['user_first_click_day', 'user_last_click_day']].fillna(0)

    # 用户当前点击在一天中的排序，非真实
    data['user_click_rank_day'] = data.apply(user_day_rank_mapper, axis=1)
    
    data = pd.merge(data, user_first_click_time_day, 'left', on=['user_id', 'day'])
    data = pd.merge(data, user_last_click_time_day, 'left', on=['user_id', 'day'])
    data['user_click_interval_first_day'] = data['context_timestamp'] - data['user_first_click_time_day']
    data['user_click_interval_last_day'] = data['user_last_click_time_day'] - data['context_timestamp']
    
    #计算当前点击时间与前一次后一次的时间差
    t1 = data[['user_id', 'day', 'context_timestamp']]
    t1.context_timestamp = t1.context_timestamp.astype('str')
    t1 = t1.groupby(['user_id', 'day'])['context_timestamp'].agg(lambda x:':'.join(x)).reset_index()
    t1.rename(columns={'context_timestamp':'times'},inplace=True)

    t2 = data[['user_id', 'day', 'context_timestamp']]
    t2 = pd.merge(t2, t1, on=['user_id', 'day'], how='left')
    t2['time_now'] = t2.context_timestamp.astype('str') + '-' + t2.times
    t2['user_click_time_gap_before'] = t2.time_now.apply(get_gap_before)
    t2['user_click_time_gap_after'] = t2.time_now.apply(get_gap_after)
    t2['user_click_true_rank_day'] = t2.time_now.apply(get_true_rank)
    t3 = t2[['user_click_time_gap_before','user_click_time_gap_after', 'user_click_true_rank_day']]

    
    data = data.join(t3)
    data = data[['user_click_rank_day', 'user_click_true_rank_day',
                 'user_click_interval_first_day', 'user_click_interval_last_day', 
                 'user_click_time_gap_before','user_click_time_gap_after']]
        
    dump_pickle(data, feature_path)


def add_user_click_rank_day(data):
    '''添加用户当前点击在一天中的排序

    join_key: ['instance_id',]

    '''

    feature_path = feature_data_path +'_2_3_' + 'user_click_rank_day.pkl'
    if not os.path.exists(feature_path):
        gen_user_click_rank_day()
        
    user_click_rank_day = load_pickle(feature_path)
    data = data.join(user_click_rank_day)

    return data


# # 6，7号合并当做全局

# In[4]:


def user_global_rank_mapper(row):
    '''

    return:
        0: 只有一次点击
        1: 第一次点击
        2: 非首尾点击
        3: 最后一次点击

    '''
    if row['user_click_global'] <= 1:
        return 0
    elif row['user_first_click_global'] > 0:
        return 1
    elif row['user_last_click_global'] > 0:
        return 3
    else:
        return 2


def gen_user_click_rank_global():

    data = load_pickle(raw_data_path + 'all_data_4567.pkl')

    feature_path = feature_data_path  +'_2_3_'+ 'user_click_rank_global.pkl'

    print('generating '+feature_path)

    user_click_global = data.groupby(['user_id']).size().reset_index().rename(columns={0: 'user_click_global'})

    data = pd.merge(data, user_click_global, how='left',
                        on=['user_id', ])


    # 用户在一天内点击该物品的时间戳排序
    sorted_data = data.sort_values(by=['user_id', 'context_timestamp'], ascending=True)

    # 保留用户在一天内首次点击的记录
    first = sorted_data.drop_duplicates(['user_id']).copy()

    # 保留用户在一天内最后点击的记录
    last = sorted_data.drop_duplicates(['user_id'], keep='last').copy()
    
    # 用户一天内首次点击的时间戳
    user_first_click_time_global = sorted_data.groupby(['user_id'])['context_timestamp'].first().reset_index().rename(columns={'context_timestamp': 'user_first_click_time_global'})
        
    # 用户一天内最后点击的时间戳
    user_last_click_time_global = sorted_data.groupby(['user_id'])['context_timestamp'].last().reset_index().rename(columns={'context_timestamp': 'user_last_click_time_global'})


    first['user_first_click_global'] = 1
    first = first[['user_first_click_global']]
    data = data.join(first)

    last['user_last_click_global'] = 1
    last = last[['user_last_click_global']]
    data = data.join(last)
    
    data[['user_first_click_global', 'user_last_click_global']] = data[['user_first_click_global', 'user_last_click_global']].fillna(0)

    # 用户当前点击在一天中的排序，非真实
    data['user_click_rank_global'] = data.apply(user_global_rank_mapper, axis=1)
    
    data = pd.merge(data, user_first_click_time_global, 'left', on=['user_id'])
    data = pd.merge(data, user_last_click_time_global, 'left', on=['user_id'])
    data['user_click_interval_first_global'] = data['context_timestamp'] - data['user_first_click_time_global']
    data['user_click_interval_last_global'] = data['user_last_click_time_global'] - data['context_timestamp']
    
    #计算当前点击时间与前一次后一次的时间差
    t1 = data[['user_id', 'context_timestamp']]
    t1.context_timestamp = t1.context_timestamp.astype('str')
    t1 = t1.groupby(['user_id'])['context_timestamp'].agg(lambda x:':'.join(x)).reset_index()
    t1.rename(columns={'context_timestamp':'times'},inplace=True)

    t2 = data[['user_id', 'context_timestamp']]
    t2 = pd.merge(t2, t1, on=['user_id'], how='left')
    t2['time_now'] = t2.context_timestamp.astype('str') + '-' + t2.times
    t2['user_click_time_gap_before_global'] = t2.time_now.apply(get_gap_before)
    t2['user_click_time_gap_after_global'] = t2.time_now.apply(get_gap_after)
    t2['user_click_true_rank_global'] = t2.time_now.apply(get_true_rank)
    t3 = t2[['user_click_time_gap_before_global','user_click_time_gap_after_global', 'user_click_true_rank_global']]

    
    data = data.join(t3)
    data = data[['user_click_rank_global', 'user_click_true_rank_global',
                 'user_click_interval_first_global', 'user_click_interval_last_global', 
                 'user_click_time_gap_before_global','user_click_time_gap_after_global']]
        
    dump_pickle(data, feature_path)


def add_user_click_rank_global(data):
    '''添加用户当前点击在一天中的排序

    join_key: ['instance_id',]

    '''

    feature_path = feature_data_path  +'_2_3_'+ 'user_click_rank_global.pkl'
    if not os.path.exists(feature_path):
        gen_user_click_rank_global()
        
    user_click_rank_global = load_pickle(feature_path)
    data = data.join(user_click_rank_global)

    return data


# ## 用户当前点击feature，在一天中的排序、与前后点击记录的时间差、与当天首位点击记录的时间差

# In[5]:


def user_feature_rank_day_mapper(row):
    '''

    return:
        0: 只有一次点击
        1: 第一次点击
        2: 非首尾点击
        3: 最后一次点击

    '''
    if row['user_feature_click_day'] <= 1:
        return 0
    elif row['user_feature_first_click_day'] > 0:
        return 1
    elif row['user_feature_last_click_day'] > 0:
        return 3
    else:
        return 2


def gen_user_feature_click_rank_day(update=True):
    '''用户当天点击当前属性的排序

    file_name: user_feature_click_rank_day.pkl

    features:
        'user_category2_label_click_rank_day',
        'user_category2_label_click_true_rank_day',
        'user_category2_label_first_click_interval_day',
        'user_category2_label_last_click_interval_day',
        'user_category2_label_click_time_gap_before_day',
        'user_category2_label_click_time_gap_after_day',
        'user_category3_label_click_rank_day',
        'user_category3_label_click_true_rank_day',
        'user_category3_label_first_click_interval_day',
        'user_category3_label_last_click_interval_day',
        'user_category3_label_click_time_gap_before_day',
        'user_category3_label_click_time_gap_after_day',
        'user_shop_id_click_rank_day',
        'user_shop_id_click_true_rank_day',
        'user_shop_id_first_click_interval_day',
        'user_shop_id_last_click_interval_day',
        'user_shop_id_click_time_gap_before_day',
        'user_shop_id_click_time_gap_after_day',
        'user_item_id_click_rank_day',
        'user_item_id_click_true_rank_day',
        'user_item_id_first_click_interval_day',
        'user_item_id_last_click_interval_day',
        'user_item_id_click_time_gap_before_day',
        'user_item_id_click_time_gap_after_day',
        'user_item_brand_id_click_rank_day',
        'user_item_brand_id_click_true_rank_day',
        'user_item_brand_id_first_click_interval_day',
        'user_item_brand_id_last_click_interval_day',
        'user_item_brand_id_click_time_gap_before_day',
        'user_item_brand_id_click_time_gap_after_day',

    '''

    all_data = load_pickle(raw_data_path + 'all_data_4567.pkl')

    feature_list = ['category2_label', 'category3_label',
                    'shop_id', 'item_id', 'item_brand_id',
                    'item_property_topic_k_15'
                   ]

    for feature in tqdm(feature_list):

        feature_path = feature_data_path  +'_2_3_'+ 'user_' + feature + '_click_rank_day.pkl'
        
        if os.path.exists(feature_path):
            print('found ' + feature_path)
            
        else:
            print('generating '+feature_path)

            first_click_feature_name = 'user_' + feature + '_first_click_day'
            last_click_feature_name = 'user_' + feature + '_last_click_day'
            rank_feature_name = 'user_' + feature + '_click_rank_day'

            true_rank_feature_name = 'user_' + feature + '_click_true_rank_day'

            first_click_time_name = 'user_' + feature + '_first_click_time_day'
            last_click_time_name = 'user_' + feature + '_last_click_time_day'
            first_click_interval_name = 'user_' + feature + '_first_click_interval_day'
            last_click_interval_name = 'user_' + feature + '_last_click_interval_day'

            time_gap_before_name = 'user_' + feature + '_click_time_gap_before_day'
            time_gap_after_name = 'user_' + feature + '_click_time_gap_after_day'

            user_feature_click_day = all_data.groupby(['user_id', feature, 'day']).size(
            ).reset_index().rename(columns={0: 'user_feature_click_day'})


            data = pd.merge(all_data, user_feature_click_day,
                            how='left', on=['user_id', feature, 'day'])

            # 用户在一天内点击该特征的记录的时间戳排序
            sorted_data = data.sort_values(
                by=['user_id', feature, 'day', 'context_timestamp'], ascending=True)[['user_id', feature, 'day', 'context_timestamp']]

            # 保留用户一天内首次点击该特征的记录
            first = sorted_data.drop_duplicates(['user_id', feature, 'day']).copy()

            # 保留用户一天内最后点击该特征的记录
            last = sorted_data.drop_duplicates(
                ['user_id', feature, 'day'], keep='last').copy()

            first.rename(
                columns={'context_timestamp': first_click_time_name}, inplace=True)
            last.rename(
                columns={'context_timestamp': last_click_time_name}, inplace=True)

            data = pd.merge(data, first, 'left', on=['user_id', feature, 'day'])
            data = pd.merge(data, last, 'left', on=['user_id', feature, 'day'])

            data[first_click_interval_name] = data['context_timestamp'] -                 data[first_click_time_name]
            data[last_click_interval_name] = data[last_click_time_name] -                 data['context_timestamp']

            first['user_feature_first_click_day'] = 1
            first = first[['user_feature_first_click_day']]
            data = data.join(first)

            last['user_feature_last_click_day'] = 1
            last = last[['user_feature_last_click_day']]
            data = data.join(last)

            data[['user_feature_first_click_day', 'user_feature_last_click_day']] = data[[
                'user_feature_first_click_day', 'user_feature_last_click_day']].fillna(0)

            data['user_feature_click_rank_day'] = data.apply(
                user_feature_rank_day_mapper, axis=1)

            data.rename(
                columns={'user_feature_click_rank_day': rank_feature_name}, inplace=True)

            # 计算当前点击时间与前一次后一次的时间差
            t1 = data[['user_id', 'day', feature, 'context_timestamp']]
            t1.context_timestamp = t1.context_timestamp.astype('str')
            t1 = t1.groupby(['user_id', feature, 'day'])['context_timestamp'].agg(
                lambda x: ':'.join(x)).reset_index()
            t1.rename(columns={'context_timestamp': 'times'}, inplace=True)

            t2 = data[['user_id', 'day', feature, 'context_timestamp']]
            t2 = pd.merge(t2, t1, on=['user_id', feature, 'day'], how='left')
            t2['time_now'] = t2.context_timestamp.astype(
                'str') + '-' + t2.times
            t2[time_gap_before_name] = t2.time_now.apply(get_gap_before)
            t2[time_gap_after_name] = t2.time_now.apply(get_gap_after)
            t2[true_rank_feature_name] = t2.time_now.apply(get_true_rank)
            t3 = t2[[time_gap_before_name,
                     time_gap_after_name, true_rank_feature_name]]

            data = data.join(t3)

            data = data[[rank_feature_name, true_rank_feature_name,
                         first_click_interval_name, last_click_interval_name,
                         time_gap_before_name, time_gap_after_name
                         ]]

            dump_pickle(data, feature_path)


def add_user_feature_click_rank_day(data):
    '''用户当天点击当前属性的排序

    join_key: ['instance_id',]

    '''

    feature_list = ['category2_label', 'category3_label',
                    'shop_id', 'item_id', 'item_brand_id',
                    'item_property_topic_k_15'
                   ]
    for feature in tqdm(feature_list):
        feature_path = feature_data_path  +'_2_3_'+ 'user_' + feature + '_click_rank_day.pkl'
        if not os.path.exists(feature_path):
            gen_user_feature_click_rank_day()

        user_feature_click_rank_day = load_pickle(feature_path)
        data = data.join(user_feature_click_rank_day)

    return data


# # 6.7号合并算全局

# In[6]:


def user_feature_rank_global_mapper(row):
    '''

    return:
        0: 只有一次点击
        1: 第一次点击
        2: 非首尾点击
        3: 最后一次点击

    '''
    if row['user_feature_click_global'] <= 1:
        return 0
    elif row['user_feature_first_click_global'] > 0:
        return 1
    elif row['user_feature_last_click_global'] > 0:
        return 3
    else:
        return 2


def gen_user_feature_click_rank_global(update=True):

    all_data = load_pickle(raw_data_path + 'all_data_4567.pkl')

    feature_list = ['category2_label', 'category3_label',
                    'shop_id', 'item_id', 'item_brand_id',
                    'item_property_topic_k_15'
                   ]

    for feature in tqdm(feature_list):

        feature_path = feature_data_path  +'_2_3_'+ 'user_' + feature + '_click_rank_global.pkl'
        if os.path.exists(feature_path):
            print('found ' + feature_path)
            
        else:
            print('generating '+feature_path)

            first_click_feature_name = 'user_' + feature + '_first_click_global'
            last_click_feature_name = 'user_' + feature + '_last_click_global'
            rank_feature_name = 'user_' + feature + '_click_rank_global'

            true_rank_feature_name = 'user_' + feature + '_click_true_rank_global'

            first_click_time_name = 'user_' + feature + '_first_click_time_global'
            last_click_time_name = 'user_' + feature + '_last_click_time_global'
            first_click_interval_name = 'user_' + feature + '_first_click_interval_global'
            last_click_interval_name = 'user_' + feature + '_last_click_interval_global'

            time_gap_before_name = 'user_' + feature + '_click_time_gap_before_global'
            time_gap_after_name = 'user_' + feature + '_click_time_gap_after_global'

            user_feature_click_global = all_data.groupby(['user_id', feature]).size(
            ).reset_index().rename(columns={0: 'user_feature_click_global'})


            data = pd.merge(all_data, user_feature_click_global,
                            how='left', on=['user_id', feature])

            # 用户在一天内点击该特征的记录的时间戳排序
            sorted_data = data.sort_values(
                by=['user_id', feature, 'context_timestamp'], ascending=True)[['user_id', feature, 'context_timestamp']]

            # 保留用户一天内首次点击该特征的记录
            first = sorted_data.drop_duplicates(['user_id', feature]).copy()

            # 保留用户一天内最后点击该特征的记录
            last = sorted_data.drop_duplicates(
                ['user_id', feature], keep='last').copy()

            first.rename(
                columns={'context_timestamp': first_click_time_name}, inplace=True)
            last.rename(
                columns={'context_timestamp': last_click_time_name}, inplace=True)

            data = pd.merge(data, first, 'left', on=['user_id', feature])
            data = pd.merge(data, last, 'left', on=['user_id', feature])

            data[first_click_interval_name] = data['context_timestamp'] -                 data[first_click_time_name]
            data[last_click_interval_name] = data[last_click_time_name] -                 data['context_timestamp']

            first['user_feature_first_click_global'] = 1
            first = first[['user_feature_first_click_global']]
            data = data.join(first)

            last['user_feature_last_click_global'] = 1
            last = last[['user_feature_last_click_global']]
            data = data.join(last)

            data[['user_feature_first_click_global', 'user_feature_last_click_global']] = data[[
                'user_feature_first_click_global', 'user_feature_last_click_global']].fillna(0)

            data['user_feature_click_rank_global'] = data.apply(
                user_feature_rank_global_mapper, axis=1)

            data.rename(
                columns={'user_feature_click_rank_global': rank_feature_name}, inplace=True)

            # 计算当前点击时间与前一次后一次的时间差
            t1 = data[['user_id', feature, 'context_timestamp']]
            t1.context_timestamp = t1.context_timestamp.astype('str')
            t1 = t1.groupby(['user_id', feature])['context_timestamp'].agg(
                lambda x: ':'.join(x)).reset_index()
            t1.rename(columns={'context_timestamp': 'times'}, inplace=True)

            t2 = data[['user_id', feature, 'context_timestamp']]
            t2 = pd.merge(t2, t1, on=['user_id', feature], how='left')
            t2['time_now'] = t2.context_timestamp.astype(
                'str') + '-' + t2.times
            t2[time_gap_before_name] = t2.time_now.apply(get_gap_before)
            t2[time_gap_after_name] = t2.time_now.apply(get_gap_after)
            t2[true_rank_feature_name] = t2.time_now.apply(get_true_rank)
            t3 = t2[[time_gap_before_name,
                     time_gap_after_name, true_rank_feature_name]]

            data = data.join(t3)

            data = data[[rank_feature_name, true_rank_feature_name,
                         first_click_interval_name, last_click_interval_name,
                         time_gap_before_name, time_gap_after_name
                         ]]

            dump_pickle(data, feature_path)


def add_user_feature_click_rank_global(data):
    '''用户当天点击当前属性的排序

    join_key: ['instance_id',]

    '''

    feature_list = ['category2_label', 'category3_label',
                    'shop_id', 'item_id', 'item_brand_id',
                    'item_property_topic_k_15'
                   ]

    for feature in tqdm(feature_list):
        feature_path = feature_data_path  +'_2_3_'+ 'user_' + feature + '_click_rank_global.pkl'
        if not os.path.exists(feature_path):
            gen_user_feature_click_rank_global()

        user_feature_click_rank_global = load_pickle(feature_path)
        data = data.join(user_feature_click_rank_global)

    return data


# In[ ]:


if __name__ =='__main__':
    
    data = load_pickle(raw_data_path + 'all_data_4567.pkl')
    
    data = add_user_click_rank_day(data)
    data = add_user_click_rank_global(data)
       
    data = add_user_feature_click_rank_day(data)
    data = add_user_feature_click_rank_global(data)
    print(data.columns)


