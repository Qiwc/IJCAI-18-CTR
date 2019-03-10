
# coding: utf-8

# ## 用户与相关属性的组合特征

# In[2]:


import os
import pickle
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import load_pickle, dump_pickle, raw_data_path, feature_data_path


# In[5]:


def gen_user_feature_feature_click_day(update=True):
    data = load_pickle(raw_data_path + 'all_data.pkl')

    feature_path = feature_data_path + 'user_brand_price_click_day.pkl'
    feature_path_1 = feature_data_path + 'user_label_price_click_day.pkl'
    feature_path_2 = feature_data_path + 'user_label_brand_click_day.pkl'
    if os.path.exists(feature_path) and update == False:
        print('found '+feature_path)
    else:
        print('generating '+feature_path)
        user_brand_price_click_day = data.groupby(['user_id', 'day', 'item_brand_id', 'item_price_level']).size(
        ).reset_index().rename(columns={0: 'user_brand_price_click_day'})
        dump_pickle(user_brand_price_click_day, feature_path)
        
        user_label_price_click_day = data.groupby(['user_id', 'day', 'category2_label', 'item_price_level']).size(
        ).reset_index().rename(columns={0: 'user_label_price_click_day'})
        dump_pickle(user_label_price_click_day, feature_path_1)
        
        user_label_brand_click_day = data.groupby(['user_id', 'day', 'category2_label', 'item_brand_id']).size(
        ).reset_index().rename(columns={0: 'user_label_brand_click_day'})
        dump_pickle(user_label_brand_click_day, feature_path_2)


def add_user_feature_feature_click_day(data):
   
 
    feature_path = feature_data_path + 'user_brand_price_click_day.pkl'
    feature_path_1 = feature_data_path + 'user_label_price_click_day.pkl'
    feature_path_2 = feature_data_path + 'user_label_brand_click_day.pkl'
    if not os.path.exists(feature_path):
        gen_user_feature_feature_click_day()
    feature_click_day = load_pickle(feature_path)
    data = pd.merge(data, feature_click_day, 'left',['user_id', 'day', 'item_brand_id', 'item_price_level'])

    feature_click_day = load_pickle(feature_path_1)
    data = pd.merge(data, feature_click_day, 'left',['user_id', 'day', 'category2_label', 'item_price_level'])
    
    feature_click_day = load_pickle(feature_path_2)
    data = pd.merge(data, feature_click_day, 'left',['user_id', 'day', 'category2_label', 'item_brand_id'])
    
    return data


# In[9]:


def gen_user_feature_feature_click_all(update=True):
    data = load_pickle(raw_data_path + 'all_data.pkl')

    feature_path = feature_data_path + 'user_brand_price_click_all.pkl'
    feature_path_1 = feature_data_path + 'user_label_price_click_all.pkl'
    feature_path_2 = feature_data_path + 'user_label_brand_click_all.pkl'
    if os.path.exists(feature_path) and update == False:
        print('found '+feature_path)
    else:
        print('generating '+feature_path)
        user_brand_price_click_day = data.groupby(['user_id', 'item_brand_id', 'item_price_level']).size(
        ).reset_index().rename(columns={0: 'user_brand_price_click_all'})
        dump_pickle(user_brand_price_click_day, feature_path)
        
        user_label_price_click_day = data.groupby(['user_id', 'category2_label', 'item_price_level']).size(
        ).reset_index().rename(columns={0: 'user_label_price_click_all'})
        dump_pickle(user_label_price_click_day, feature_path_1)
        
        user_label_brand_click_day = data.groupby(['user_id', 'category2_label', 'item_brand_id']).size(
        ).reset_index().rename(columns={0: 'user_label_brand_click_all'})
        dump_pickle(user_label_brand_click_day, feature_path_2)


def add_user_feature_feature_click_all(data):
   
 
    feature_path = feature_data_path + 'user_brand_price_click_all.pkl'
    feature_path_1 = feature_data_path + 'user_label_price_click_all.pkl'
    feature_path_2 = feature_data_path + 'user_label_brand_click_all.pkl'
    if not os.path.exists(feature_path):
        gen_user_feature_feature_click_all()
    feature_click_day = load_pickle(feature_path)
    data = pd.merge(data, feature_click_day, 'left',['user_id', 'item_brand_id', 'item_price_level'])

    feature_click_day = load_pickle(feature_path_1)
    data = pd.merge(data, feature_click_day, 'left',['user_id', 'category2_label', 'item_price_level'])
    
    feature_click_day = load_pickle(feature_path_2)
    data = pd.merge(data, feature_click_day, 'left',['user_id', 'category2_label', 'item_brand_id'])
    
    return data


# In[2]:


def gen_user_feature_click_day(update=True):
    """生成用户对所有分类属性的当天点击量

    file_name: user_(feature_id)_click_day.pkl
    
    features:
        'user_item_id_click_day', 
        'user_item_brand_id_click_day',
        'user_item_city_id_click_day', 
        'user_context_page_id_click_day',
        'user_shop_id_click_day',

    """

    data = load_pickle(raw_data_path + 'all_data.pkl')

    feature_list=['item_id', 'item_brand_id', 'item_city_id', 'category2_label','item_price_level','item_sales_level', 
                  'item_collected_level', 'item_pv_level',
                  'context_page_id', 
                  'shop_id', 'shop_review_num_level', 'shop_star_level',]

    for feature in tqdm(feature_list):
        feature_path = feature_data_path + 'user_'+feature+'_click_day.pkl'
        if os.path.exists(feature_path) and update == False:
            print('found '+feature_path)
        else:
            print('generating '+feature_path)
            user_feature_click_day = data.groupby(['user_id', 'day', feature]).size(
            ).reset_index().rename(columns={0: 'user_'+feature+'_click_day'})
            dump_pickle(user_feature_click_day, feature_path)


def add_user_feature_click_day(data):
    """添加用户对所有分类属性的当天点击量

    join_key: ['user_id', 'feature_id', 'day']

    """
    
    feature_list=['item_id', 'item_brand_id', 'item_city_id', 'category2_label','item_price_level','item_sales_level', 
                  'item_collected_level', 'item_pv_level',
                  'context_page_id', 
                  'shop_id', 'shop_review_num_level', 'shop_star_level',]

    for feature in tqdm(feature_list):
        feature_path = feature_data_path + 'user_'+feature+'_click_day.pkl'
        if not os.path.exists(feature_path):
            gen_user_feature_click_day()
        feature_click_day = load_pickle(feature_path)
        data = pd.merge(data, feature_click_day, 'left',
                        [feature, 'day', 'user_id'])

    return data


# In[3]:


def gen_user_feature_click_hour(update=True):
    """生成用户对所有分类属性的当前小时点击量

    file_name: user_(feature_id)_click_hour.pkl
    
    features:
        'user_item_id_click_hour',
        'user_item_brand_id_click_hour', 
        'user_context_page_id_click_hour', 
        'user_shop_id_click_hour',

    """

    data = load_pickle(raw_data_path + 'all_data.pkl')

    feature_list=['item_id', 'item_brand_id', 'item_city_id', 'category2_label','item_price_level','item_sales_level', 
                  'item_collected_level', 'item_pv_level',
                  'context_page_id', 
                  'shop_id', 'shop_review_num_level', 'shop_star_level',]

    for feature in tqdm(feature_list):
        feature_path = feature_data_path + 'user_'+feature+'_click_hour.pkl'
        if os.path.exists(feature_path) and update == False:
            print('found '+feature_path)
        else:
            print('generating '+feature_path)
            user_feature_click_day = data.groupby(['user_id', 'day', 'hour', feature]).size(
            ).reset_index().rename(columns={0: 'user_'+feature+'_click_hour'})
            dump_pickle(user_feature_click_day, feature_path)


def add_user_feature_click_hour(data):
    """添加用户对所有分类属性的当天点击统计量

    join_key: ['user_id', 'feature_id', 'day', 'hour']

    """

    feature_list=['item_id', 'item_brand_id', 'item_city_id', 'category2_label','item_price_level','item_sales_level', 
                  'item_collected_level', 'item_pv_level',
                  'context_page_id', 
                  'shop_id', 'shop_review_num_level', 'shop_star_level',]
    
    for feature in tqdm(feature_list):
        feature_path = feature_data_path + 'user_'+feature+'_click_hour.pkl'
        if not os.path.exists(feature_path):
            gen_user_feature_click_hour()
        feature_click_hour = load_pickle(feature_path)
        data = pd.merge(data, feature_click_hour, 'left', [
                        feature, 'day', 'hour', 'user_id'])

    return data


# ## 生成用户对单一特征点击数据的统计特征

# In[4]:


def gen_user_feature_click_day_stats(data, feature):
    '''生成用户对单一特征点击量的单日统计特征

    '''
    
    user_feature_click_day = pd.DataFrame(data.groupby(
        ['user_id', feature, 'day'])['context_timestamp'].count(), )
    user_feature_click_day.rename(
        columns={'context_timestamp': feature + '_m'}, inplace=True)
    user_feature_click_day.reset_index(inplace=True)
    user_feature_click_day_mean = pd.DataFrame(user_feature_click_day.groupby(['user_id'])[
        feature+'_m'].mean()).rename(columns={feature+'_m': 'user_' + feature + '_click_day_mean'}).reset_index()
    user_feature_click_day_max = pd.DataFrame(user_feature_click_day.groupby(['user_id'])[
        feature+'_m'].max()).rename(columns={feature+'_m': 'user_' + feature + '_click_day_max'}).reset_index()
    user_feature_click_day_min = pd.DataFrame(user_feature_click_day.groupby(['user_id'])[
        feature+'_m'].min()).rename(columns={feature+'_m': 'user_' + feature + '_click_day_min'}).reset_index()

    data = pd.merge(data, user_feature_click_day_mean,
                    how='left', on='user_id')
    data = pd.merge(data, user_feature_click_day_max, how='left', on='user_id')
    data = pd.merge(data, user_feature_click_day_min, how='left', on='user_id')
    return data


def gen_user_click_stats(update=True):
    """生成用户点击数据的统计特征
    
    file_name: user_feature_click_stats.pkl
    
    example:
        user_item_id_click_day_mean 用户对一个 item 平均每天点击多少次
        user_item_id_click_day_max 用户对一个 item 最多单日点击次数
    
    features:
        'user_item_id_click_day_mean', 'user_item_id_click_day_min', 'user_item_id_click_day_max', 
        'user_item_brand_id_click_day_mean', 'user_item_brand_id_click_day_min', 'user_item_brand_id_click_day_max',
        'user_shop_id_click_day_mean', 'user_shop_id_click_day_min', 'user_shop_id_click_day_max',
        'user_category2_label_click_day_mean', 'user_category2_label_click_day_min', 'user_category2_label_click_day_max',
        
    """

    data = load_pickle(raw_data_path + 'all_data.pkl')
    feature_path = feature_data_path + 'user_feature_click_stats.pkl'

    if os.path.exists(feature_path) and update == False:
        print('found ' + feature_path)
    else:
        print('generating ' + feature_path)
        
        feature_names = ['user_item_id_click_day_mean', 'user_item_id_click_day_min', 'user_item_id_click_day_max',
                         'user_item_brand_id_click_day_mean', 'user_item_brand_id_click_day_min', 'user_item_brand_id_click_day_max',
                         'user_shop_id_click_day_mean', 'user_shop_id_click_day_min', 'user_shop_id_click_day_max',
                         'user_category2_label_click_day_mean', 'user_category2_label_click_day_min', 'user_category2_label_click_day_max',
                         ]

        stats_feature = ['item_id', 'item_brand_id', 'shop_id', 'category2_label']
        for feature in tqdm(stats_feature):
            data = gen_user_feature_click_day_stats(data, feature)

        # 每个用户只保留一条记录
        data = data[feature_names + ['user_id']].drop_duplicates(['user_id'])
        dump_pickle(data, feature_path)


def add_user_click_stats(data,):
    """添加用户点击数据的统计特征

    join_key: ['user_id',]

    """

    feature_path = feature_data_path + 'user_feature_click_stats.pkl'
    if not os.path.exists(feature_path):
        gen_user_click_stats()
    user_click_stats = load_pickle(feature_path)
    data = pd.merge(data, user_click_stats, 'left', 'user_id')
    return data


# ## 测试

# In[10]:


if __name__ =='__main__':
    all_data = load_pickle(raw_data_path + 'all_data.pkl')  
    all_data = add_user_feature_click_day(all_data)
    all_data = add_user_feature_click_hour(all_data)
    all_data = add_user_click_stats(all_data)
    all_data = add_user_feature_feature_click_day(all_data)
    all_data = add_user_feature_feature_click_all(all_data)
    all_data.columns

