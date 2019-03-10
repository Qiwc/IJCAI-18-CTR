
# coding: utf-8

# In[28]:


import os
import pickle
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import load_pickle, dump_pickle, raw_data_path, feature_data_path

from _2_2_gen_statistics_features import add_feature_click_stats


# In[29]:


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
        if this_gap < 0:
            gaps.append(this_gap)
    return len(gaps)


# # 用户当前点击在一天中的排序
#         0: 只有一次点击
#         1: 第一次点击
#         2: 非首尾点击
#         3: 最后一次点击

# In[30]:


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


def gen_user_click_rank_day(update=True):
    '''生成用户当前点击在一天中的排序

    file_name: user_click_day_rank.pkl

    features:
        'user_click_rank_day', 
        'user_first_click_day', 
        'user_last_click_day'

    '''

    data = load_pickle(raw_data_path + 'all_data.pkl')

    feature_path = feature_data_path + 'user_click_rank_day.pkl'

    if os.path.exists(feature_path) and update == False:
        print('found '+feature_path)
    else:
        print('generating '+feature_path)

        user_click_day = data.groupby(['user_id', 'day']).size(
        ).reset_index().rename(columns={0: 'user_click_day'})

        data = pd.merge(data, user_click_day, how='left',
                        on=['user_id', 'day'])

        instance = data.groupby(['instance_id']).size(
        ).reset_index().rename(columns={0: 'instance_num'})

        # 用户在一天内点击该物品的时间戳排序
        sorted_data = data.sort_values(
            by=['user_id', 'day', 'context_timestamp'], ascending=True)

        # 保留一天内用户首次点击该物品的记录
        first = sorted_data.drop_duplicates(['user_id', 'day']).copy()

        # 保留一天内用户最后一次点击该物品的记录
        last = sorted_data.drop_duplicates(
            ['user_id', 'day'], keep='last').copy()

        first['user_first_click_day'] = 1
        first = first[['user_first_click_day']]
        data = data.join(first)

        last['user_last_click_day'] = 1
        last = last[['user_last_click_day']]
        data = data.join(last)

        data[['user_first_click_day', 'user_last_click_day']] = data[[
            'user_first_click_day', 'user_last_click_day']].fillna(0)

        data['user_click_rank_day'] = data.apply(user_day_rank_mapper, axis=1)

        data = data[['user_click_rank_day',
                     'user_first_click_day', 'user_last_click_day']]
        dump_pickle(data, feature_path)


def add_user_click_rank_day(data):
    '''添加用户当前点击在一天中的排序

    join_key: ['instance_id',]

    '''

    feature_path = feature_data_path + 'user_click_rank_day.pkl'

    if not os.path.exists(feature_path):
        gen_user_click_rank_day()
    user_click_rank_day = load_pickle(feature_path)
    data = data.join(user_click_rank_day)

    return data


# # 添加用户当天点击的排名：user_click_true_rank_day

# In[31]:


def gen_user_click_time_interval_day(update=True):
    '''生成用户当前点击与当天首尾点击的时间间隔
    
    file_name: user_click_time_interval_day.pkl

    features:
        'user_click_interval_first_day', 
        'user_click_interval_last_day', 
        'user_click_true_rank_day', 

    '''

    data = load_pickle(raw_data_path + 'all_data.pkl')

    feature_path = feature_data_path + 'user_click_time_interval_day.pkl'

    if os.path.exists(feature_path) and update == False:
        print('found '+feature_path)
    else:
        print('generating '+feature_path)


        # 用户在一天内点击该物品的时间戳排序
        sorted_data = data.sort_values(
            by=['user_id', 'day', 'context_timestamp'], ascending=True)

        # 保留一天内用户首次点击该物品的记录
        user_first_click_time_day = sorted_data.groupby(['user_id', 'day'])['context_timestamp'].first().reset_index().rename(columns={'context_timestamp': 'user_first_click_time_day'})
        
        # 保留一天内用户最后一次点击该物品的记录
        user_last_click_time_day = sorted_data.groupby(['user_id', 'day'])['context_timestamp'].last().reset_index().rename(columns={'context_timestamp': 'user_last_click_time_day'})
        
        #保留一天内用户点击该物品的平均时间
        user_mean_click_time_day = sorted_data.groupby(['user_id', 'day'])['context_timestamp'].mean().reset_index().rename(columns={'context_timestamp': 'user_mean_click_time_day'})

        #计算一天内用户点击该物品时间的标准差
        user_std_click_time_day = sorted_data.groupby(['user_id', 'day'])['context_timestamp'].std().reset_index().rename(columns={'context_timestamp': 'user_std_click_time_day'})
        
        data = pd.merge(data, user_first_click_time_day, 'left', on=['user_id', 'day'])
        data = pd.merge(data, user_last_click_time_day, 'left', on=['user_id', 'day'])
        data = pd.merge(data, user_mean_click_time_day, 'left', on=['user_id', 'day'])
        data = pd.merge(data, user_std_click_time_day, 'left', on=['user_id', 'day'])
        
        data['user_click_interval_first_day'] = data['context_timestamp'] - data['user_first_click_time_day']
        data['user_click_interval_last_day'] = data['user_last_click_time_day'] - data['context_timestamp']
        data['user_click_interval_mean_day'] = data['context_timestamp'] - data['user_mean_click_time_day']
        data['user_click_interval_diff_day'] = data['user_last_click_time_day'] - data['user_first_click_time_day']
        data['user_click_interval_prob'] = data['user_click_interval_first_day'] / data['user_click_interval_diff_day']
        
        #计算当前点击时间与前一次后一次的时间差gap
        t1 = data[['user_id', 'day', 'context_timestamp']]
        t1.context_timestamp = t1.context_timestamp.astype('str')
        t1 = t1.groupby(['user_id', 'day'])['context_timestamp'].agg(lambda x:':'.join(x)).reset_index()
        t1.rename(columns={'context_timestamp':'times'},inplace=True)

        t2 = data[['user_id', 'day', 'context_timestamp']]
        t2 = pd.merge(t2, t1, on=['user_id', 'day'], how='left')
        t2['time_now'] = t2.context_timestamp.astype('str') + '-' + t2.times
        t2['time_gap_before'] = t2.time_now.apply(get_gap_before)
        t2['time_gap_after'] = t2.time_now.apply(get_gap_after)
        t2['user_click_true_rank_day'] = t2.time_now.apply(get_true_rank)
        t3 = t2[['time_gap_before','time_gap_after', 'user_click_true_rank_day']]
        
        
        data = data.join(t3)
        
        data = data[['user_click_interval_first_day', 'user_click_interval_last_day', 
                    'user_click_interval_diff_day','user_click_interval_prob',
                     'time_gap_before','time_gap_after', 'user_click_true_rank_day']]
        
        dump_pickle(data, feature_path)


def add_user_click_time_interval_day(data):
    '''添加用户当前点击与当天首尾点击的时间间隔
    
    join_key: ['instance_id',]

    '''

    feature_path = feature_data_path + 'user_click_time_interval_day.pkl'

    if not os.path.exists(feature_path):
        gen_user_click_time_interval_day()
        
    user_click_interval_day = load_pickle(feature_path)
    data = data.join(user_click_interval_day)

    return data


# ## 用户全局点击的时间差特征

# In[32]:


def gen_user_click_time_interval(update=True):
    '''生成用户当前点击与当天首尾点击的时间间隔
    
    file_name: user_click_time_interval.pkl

    features:

    '''

    data = load_pickle(raw_data_path + 'all_data.pkl')

    feature_path = feature_data_path + 'user_click_time_interval.pkl'
    

    if os.path.exists(feature_path) and update == False:
        print('found '+feature_path)
    else:
        print('generating '+feature_path)


        # 用户点击该物品的时间戳排序，全局
#         sorted_data = data.sort_values(
#             by=['user_id', 'context_timestamp'], ascending=True)
        
        #保留一天内用户点击该物品的平均时间
        user_mean_click_hour = data.groupby(['user_id',])['hour'].mean().reset_index().rename(columns={'hour': 'user_mean_click_hour'})
        data = pd.merge(data, user_mean_click_hour, 'left', on=['user_id',])
        data['user_click_interval_mean_hour'] = data['hour'] - data['user_mean_click_hour']
        
        #计算当前点击时间与前一次后一次的时间差gap
        t1 = data[['user_id', 'context_timestamp']]
        t1.context_timestamp = t1.context_timestamp.astype('str')
        t1 = t1.groupby(['user_id', ])['context_timestamp'].agg(lambda x:':'.join(x)).reset_index()
        t1.rename(columns={'context_timestamp':'times'},inplace=True)

        t2 = data[['user_id', 'context_timestamp']]
        t2 = pd.merge(t2, t1, on=['user_id', ], how='left')
        t2['time_now'] = t2.context_timestamp.astype('str') + '-' + t2.times
        t2['time_gap_before_total'] = t2.time_now.apply(get_gap_before)
        t2['time_gap_after_total'] = t2.time_now.apply(get_gap_after)
#         t2['user_click_true_rank_day'] = t2.time_now.apply(get_true_rank)
        t3 = t2[['time_gap_before_total','time_gap_after_total',]]
        
        
        data = data.join(t3)
        
        data = data[['user_click_interval_mean_hour', 'time_gap_before_total','time_gap_after_total',]]
        
        dump_pickle(data, feature_path)


def add_user_click_time_interval(data):
    '''添加用户当前点击与当天首尾点击的时间间隔
    
    join_key: ['instance_id',]

    '''

    feature_path = feature_data_path + 'user_click_time_interval.pkl'

    if not os.path.exists(feature_path):
        gen_user_click_time_interval()
        
    user_click_interval = load_pickle(feature_path)
    data = data.join(user_click_interval)

    return data


# # 全局：用户是第几次点击这个属性: 'user_' + feature + '_click_true_rank'

# In[33]:


def user_feature_rank_mapper(row):
    '''

    return:
        0: 只有一次点击
        1: 第一次点击
        2: 非首尾点击
        3: 最后一次点击

    '''
    if row['user_feature_click'] <= 1:
        return 0
    elif row['user_feature_first_click'] > 0:
        return 1
    elif row['user_feature_last_click'] > 0:
        return 3
    else:
        return 2


def gen_user_feature_click_rank(update=True):
    '''用户是第几次点击这个属性

    file_name: user_feature_click_rank.pkl

    features:
        'user_item_id_first_click', 'user_item_id_last_click',
        'user_item_id_click_rank', 'user_item_id_first_click_interval',
        'user_item_id_last_click_interval', 'user_item_brand_id_first_click',
        'user_item_brand_id_last_click', 'user_item_brand_id_click_rank',
        'user_item_brand_id_first_click_interval',
        'user_item_brand_id_last_click_interval',
        'user_item_city_id_first_click', 'user_item_city_id_last_click',
        'user_item_city_id_click_rank',
        'user_item_city_id_first_click_interval',
        'user_item_city_id_last_click_interval', 'user_shop_id_first_click',
        'user_shop_id_last_click', 'user_shop_id_click_rank',
        'user_shop_id_first_click_interval',
        'user_shop_id_last_click_interval'

    '''

    all_data = load_pickle(raw_data_path + 'all_data.pkl')

    feature_path = feature_data_path + 'user_feature_click_rank.pkl'

    feature_list = ['item_id', 'item_brand_id', 'shop_id', 'context_page_id', 'category2_label',]

    for feature in tqdm(feature_list):

        feature_path = feature_data_path + 'user_'+feature+'_click_rank.pkl'

        if os.path.exists(feature_path) and update == False:
            print('found '+feature_path)
        else:
            print('generating '+feature_path)

            first_click_feature_name = 'user_' + feature + '_first_click'
            last_click_feature_name = 'user_' + feature + '_last_click'
            rank_feature_name = 'user_' + feature + '_click_rank'
            
            true_rank_feature_name = 'user_' + feature + '_click_true_rank'
            
            first_click_time_name = 'user_' + feature + '_first_click_time'
            last_click_time_name = 'user_' + feature + '_last_click_time'
            first_click_interval_name = 'user_' + feature + '_first_click_interval'
            last_click_interval_name = 'user_' + feature + '_last_click_interval'
            
            mean_click_time_name = 'user_' + feature + '_mean_click_time'
            std_click_time_name = 'user_' + feature + '_std_click_time'
            mean_click_interval_name = 'user_' + feature + '_mean_click_interval'
            diff_click_interval_name = 'user_' + feature + '_diff_click_interval'
            prob_click_interval_name = 'user_' + feature + '_prob_click_interval'
            
            time_gap_before_name = feature + '_time_gap_before'
            time_gap_after_name = feature + '_time_gap_after'
            
            
            user_feature_click = all_data.groupby(['user_id', feature]).size(
            ).reset_index().rename(columns={0: 'user_feature_click'})

            data = pd.merge(all_data, user_feature_click,
                            how='left', on=['user_id', feature])

            # 用户在一天内点击该物品的时间戳排序
            sorted_data = data.sort_values(
                by=['user_id', feature, 'context_timestamp'], ascending=True)[['user_id', feature, 'context_timestamp']]

            #保留一天内用户点击该物品的平均时间
            user_mean_click_time_feature = sorted_data.groupby(['user_id', feature])['context_timestamp'].mean().reset_index().rename(columns={'context_timestamp': mean_click_time_name})
            #计算一天内用户点击该物品时间的标准差
            user_std_click_time_feature = sorted_data.groupby(['user_id', feature])['context_timestamp'].std().reset_index().rename(columns={'context_timestamp': std_click_time_name})
            
            # 保留一天内用户首次点击该物品的记录
            first = sorted_data.drop_duplicates(['user_id', feature]).copy()

            # 保留一天内用户最后一次点击该物品的记录
            last = sorted_data.drop_duplicates(
                ['user_id', feature], keep='last').copy()

            first.rename(columns = {'context_timestamp': first_click_time_name}, inplace=True)
            last.rename(columns = {'context_timestamp': last_click_time_name}, inplace=True)

            data = pd.merge(data, first, 'left', on=['user_id', feature])
            data = pd.merge(data, last, 'left', on=['user_id', feature])
            data = pd.merge(data, user_mean_click_time_feature, 'left', on=['user_id', feature])
            data = pd.merge(data, user_std_click_time_feature, 'left', on=['user_id', feature])
            

            data[first_click_interval_name] = data['context_timestamp'] -  data[first_click_time_name]
            data[last_click_interval_name] = data[last_click_time_name] -  data['context_timestamp']
            data[mean_click_interval_name] = data['context_timestamp'] -  data[mean_click_time_name]
            data[diff_click_interval_name] = data[last_click_time_name] -  data[first_click_time_name]
            data[prob_click_interval_name] = data[first_click_interval_name] / data[diff_click_interval_name]
            
            first['user_feature_first_click'] = 1
            first = first[['user_feature_first_click']]
            data = data.join(first)

            last['user_feature_last_click'] = 1
            last = last[['user_feature_last_click']]
            data = data.join(last)

            data[['user_feature_first_click', 'user_feature_last_click']] = data[[
                'user_feature_first_click', 'user_feature_last_click']].fillna(0)

            data['user_feature_click_rank'] = data.apply(user_feature_rank_mapper, axis=1)

            data.rename(columns={'user_feature_first_click': first_click_feature_name, 'user_feature_last_click': last_click_feature_name,
                                 'user_feature_click_rank': rank_feature_name}, inplace=True)
            
            
            #计算当前点击时间与前一次后一次的时间差gap
            t1 = all_data[['user_id', feature, 'context_timestamp']]
            t1.context_timestamp = t1.context_timestamp.astype('str')
            t1 = t1.groupby(['user_id', feature])['context_timestamp'].agg(lambda x:':'.join(x)).reset_index()
            t1.rename(columns={'context_timestamp':'times'},inplace=True)

            t2 = all_data[['user_id', feature, 'context_timestamp']]
            t2 = pd.merge(t2, t1, on=['user_id', feature], how='left')
            t2['time_now'] = t2.context_timestamp.astype('str') + '-' + t2.times
            t2[time_gap_before_name] = t2.time_now.apply(get_gap_before)
            t2[time_gap_after_name] = t2.time_now.apply(get_gap_after)
            t2[true_rank_feature_name] = t2.time_now.apply(get_true_rank)
            t3 = t2[[time_gap_before_name,time_gap_after_name, true_rank_feature_name]]
        
            data = data.join(t3)
            
            data = data[[first_click_feature_name, last_click_feature_name,
                         rank_feature_name, first_click_interval_name, last_click_interval_name,
                        diff_click_interval_name, prob_click_interval_name,
                        time_gap_before_name,time_gap_after_name,
                        true_rank_feature_name]]
            dump_pickle(data, feature_path)


def add_user_feature_click_rank(data):
    '''添加用户当前点击在一天中的排序

    join_key: ['instance_id',]

    '''

    feature_list = ['item_id', 'item_brand_id', 'shop_id', 'context_page_id', 'category2_label',]

    for feature in tqdm(feature_list):
        feature_path = feature_data_path + 'user_'+feature+'_click_rank.pkl'
        if not os.path.exists(feature_path):
            gen_user_feature_click_rank()
        user_feature_click_rank = load_pickle(feature_path)
        data = data.join(user_feature_click_rank)

    return data


# # 当天：用户是第几次点击这个属性: 'user_' + feature + '_click_true_rank_day'

# In[ ]:


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
    '''用户是第几次点击这个属性

    file_name: user_feature_click_rank_day.pkl

    features:
        'user_item_id_first_click', 'user_item_id_last_click',
        'user_item_id_click_rank', 'user_item_id_first_click_interval',
        'user_item_id_last_click_interval', 'user_item_brand_id_first_click',
        'user_item_brand_id_last_click', 'user_item_brand_id_click_rank',
        'user_item_brand_id_first_click_interval',
        'user_item_brand_id_last_click_interval',
        'user_item_city_id_first_click', 'user_item_city_id_last_click',
        'user_item_city_id_click_rank',
        'user_item_city_id_first_click_interval',
        'user_item_city_id_last_click_interval', 'user_shop_id_first_click',
        'user_shop_id_last_click', 'user_shop_id_click_rank',
        'user_shop_id_first_click_interval',
        'user_shop_id_last_click_interval'

    '''

    all_data = load_pickle(raw_data_path + 'all_data.pkl')

#     feature_path = feature_data_path + 'user_feature_click_rank_day.pkl'

    feature_list = ['item_id', 'item_brand_id',
                    'shop_id', 'context_page_id', 'category2_label', ]

    for feature in tqdm(feature_list):

        feature_path = feature_data_path + 'user_'+feature + '_click_rank_day.pkl'

        if os.path.exists(feature_path) and update == False:
            print('found '+feature_path)
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

            mean_click_time_name = 'user_' + feature + '_mean_click_time_day'
            std_click_time_name = 'user_' + feature + '_std_click_time_day'
            mean_click_interval_name = 'user_' + feature + '_mean_click_interval_day'
            diff_click_interval_name = 'user_' + feature + '_diff_click_interval_day'
            prob_click_interval_name = 'user_' + feature + '_prob_click_interval_day'

            time_gap_before_name = feature + '_time_gap_before_day'
            time_gap_after_name = feature + '_time_gap_after_day'

            user_feature_click_day = all_data.groupby(['user_id', feature, 'day']).size(
            ).reset_index().rename(columns={0: 'user_feature_click_day'})

            data = pd.merge(all_data, user_feature_click_day,
                            how='left', on=['user_id', feature, 'day'])

            # 用户在一天内点击该物品的时间戳排序
            sorted_data = data.sort_values(
                by=['user_id', feature, 'day', 'context_timestamp'], ascending=True)[['user_id', feature, 'day', 'context_timestamp']]

            # 保留一天内用户点击该物品的平均时间
            user_mean_click_time_feature = sorted_data.groupby(['user_id', feature, 'day'])['context_timestamp'].mean(
            ).reset_index().rename(columns={'context_timestamp': mean_click_time_name})
            # 计算一天内用户点击该物品时间的标准差
            user_std_click_time_feature = sorted_data.groupby(['user_id', feature, 'day'])['context_timestamp'].std(
            ).reset_index().rename(columns={'context_timestamp': std_click_time_name})

            # 保留一天内用户首次点击该物品的记录
            first = sorted_data.drop_duplicates(['user_id', feature, 'day']).copy()

            # 保留一天内用户最后一次点击该物品的记录
            last = sorted_data.drop_duplicates(
                ['user_id', feature, 'day'], keep='last').copy()

            first.rename(
                columns={'context_timestamp': first_click_time_name}, inplace=True)
            last.rename(
                columns={'context_timestamp': last_click_time_name}, inplace=True)

            data = pd.merge(data, first, 'left', on=['user_id', feature, 'day'])
            data = pd.merge(data, last, 'left', on=['user_id', feature, 'day'])
            data = pd.merge(data, user_mean_click_time_feature,
                            'left', on=['user_id', feature, 'day'])
            data = pd.merge(data, user_std_click_time_feature,
                            'left', on=['user_id', feature, 'day'])

            data[first_click_interval_name] = data['context_timestamp'] -                 data[first_click_time_name]
            data[last_click_interval_name] = data[last_click_time_name] -                 data['context_timestamp']
            data[mean_click_interval_name] = data['context_timestamp'] -                 data[mean_click_time_name]
            data[diff_click_interval_name] = data[last_click_time_name] -                 data[first_click_time_name]
            data[prob_click_interval_name] = data[first_click_interval_name] /                 data[diff_click_interval_name]

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

            data.rename(columns={'user_feature_first_click_day': first_click_feature_name, 'user_feature_last_click_day': last_click_feature_name,
                                 'user_feature_click_rank_day': rank_feature_name}, inplace=True)

            # 计算当前点击时间与前一次后一次的时间差gap
            t1 = all_data[['user_id', 'day', feature, 'context_timestamp']]
            t1.context_timestamp = t1.context_timestamp.astype('str')
            t1 = t1.groupby(['user_id', feature, 'day'])['context_timestamp'].agg(
                lambda x: ':'.join(x)).reset_index()
            t1.rename(columns={'context_timestamp': 'times'}, inplace=True)

            t2 = all_data[['user_id', 'day', feature, 'context_timestamp']]
            t2 = pd.merge(t2, t1, on=['user_id', feature, 'day'], how='left')
            t2['time_now'] = t2.context_timestamp.astype(
                'str') + '-' + t2.times
            t2[time_gap_before_name] = t2.time_now.apply(get_gap_before)
            t2[time_gap_after_name] = t2.time_now.apply(get_gap_after)
            t2[true_rank_feature_name] = t2.time_now.apply(get_true_rank)
            t3 = t2[[time_gap_before_name,
                     time_gap_after_name, true_rank_feature_name]]

            data = data.join(t3)

            data = data[[first_click_feature_name, last_click_feature_name,
                         rank_feature_name, first_click_interval_name, last_click_interval_name,
                         diff_click_interval_name, prob_click_interval_name,
                         time_gap_before_name, time_gap_after_name,
                         true_rank_feature_name]]
            
            dump_pickle(data, feature_path)


def add_user_feature_click_rank_day(data):
    '''添加用户当前点击在一天中的排序

    join_key: ['instance_id',]

    '''

    feature_list = ['item_id', 'item_brand_id',
                    'shop_id', 'context_page_id', 'category2_label', ]

    for feature in tqdm(feature_list):
        feature_path = feature_data_path + 'user_'+feature+'_click_rank_day.pkl'
        if not os.path.exists(feature_path):
            gen_user_feature_click_rank_day()
        user_feature_click_rank_day = load_pickle(feature_path)
        data = data.join(user_feature_click_rank_day)

    return data


# # user click interval

# In[ ]:


if __name__ =='__main__':
    all_data = load_pickle(raw_data_path + 'all_data.pkl')
    
    all_data = add_user_click_rank_day(all_data)
    all_data = add_user_click_time_interval_day(all_data)
    
    gen_user_click_time_interval()
    all_data = add_user_click_time_interval(all_data)
    
    all_data = add_user_feature_click_rank(all_data)
    all_data = add_user_feature_click_rank_day(all_data)    
    


# ## 想要添加：
#     该次点击是用户第几次访问这个物品，全局数据
