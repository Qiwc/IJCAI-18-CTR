
# coding: utf-8

# In[1]:


import os
import pickle
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import load_pickle, dump_pickle, raw_data_path, feature_data_path


# ## 日点击量的统计特征
# 
# 因为日点击量的统计特征是属性的固有性质，就像一个item的价格不管哪天都是确定的，那点击量也是固定的
# 
# 所以直接使用24号前面的数据计算，去掉25号的采样干扰不会有影响

# In[2]:


def gen_feature_click_day_stats(data, feature):
    '''生成分类属性日点击量的统计特征

    '''

    data = data.copy()[[feature, 'day']]
    
    # 去除测试采样的干扰
    #data = data[data['day'] < 25]

    feature_click_day = pd.DataFrame(data.groupby(['day', feature]).size(
    )).reset_index().rename(columns={0: 'feature_click_day'})

    feature_click_day_mean = pd.DataFrame(feature_click_day.groupby([feature])['feature_click_day'].mean(
    )).rename(columns={'feature_click_day': feature + '_click_day_mean'}).reset_index()

    feature_click_day_max = pd.DataFrame(feature_click_day.groupby([feature])['feature_click_day'].max(
    )).rename(columns={'feature_click_day': feature + '_click_day_max'}).reset_index()

    feature_click_day_min = pd.DataFrame(feature_click_day.groupby([feature])['feature_click_day'].min(
    )).rename(columns={'feature_click_day': feature + '_click_day_min'}).reset_index()

    # 每个类别只保留一条记录
    data = data.drop(['day',], axis=1)
    data = data.drop_duplicates([feature, ])
    data = pd.merge(data, feature_click_day_mean, how='left', on=feature)
    data = pd.merge(data, feature_click_day_max, how='left', on=feature)
    data = pd.merge(data, feature_click_day_min, how='left', on=feature)

    return data


def gen_feature_click_stats(update=True):
    """生成各个分类属性日点击量的统计特征

    file_name: (feature)_click_day_stats.pkl

    example:
        user_id_click_day_mean 该用户平均每天点击多少次
        item_id_click_day_max 该物品单日最高销量

    features:
        'user_id_click_day_mean', 'user_id_click_day_max', 'user_id_click_day_min', 
        'item_id_click_day_mean', 'item_id_click_day_max', 'item_id_click_day_min',
        'item_brand_id_click_day_mean', 'item_brand_id_click_day_max', 'item_brand_id_click_day_min', 
        'shop_id_click_day_mean', 'shop_id_click_day_max', 'shop_id_click_day_min',
        'context_page_id_click_day_mean', 'context_page_id_click_day_max', 'context_page_id_click_day_min',
        'category2_label_click_day_mean', 'category2_label_click_day_max', 'category2_label_click_day_min'
        

    """

    data = load_pickle(raw_data_path + 'all_data.pkl')

    stats_feature = ['user_id', 'item_id', 'item_brand_id', 'shop_id']
    
    for feature in tqdm(stats_feature):
        feature_path = feature_data_path + feature + '_click_day_stats.pkl'
        if os.path.exists(feature_path) and update == False:
            print('found ' + feature_path)
        else:
            print('generating ' + feature_path)
            feature_stats = gen_feature_click_day_stats(data, feature)
            print(feature_stats.columns)
            dump_pickle(feature_stats, feature_path)
            
def add_feature_click_stats(data,):
    """添加分类属性日点击量的统计特征

    join_key: ['feature_id',]

    """
    
    stats_feature = ['user_id', 'item_id', 'item_brand_id', 'shop_id']

    for feature in tqdm(stats_feature):
        feature_path = feature_data_path + feature + '_click_day_stats.pkl'
        if not os.path.exists(feature_path):
            gen_feature_click_stats()
        feature_stats = load_pickle(feature_path)
        data = pd.merge(data, feature_stats, 'left', [feature,])
    
    return data


# ## 日转化量的统计特征

# ## 测试

# In[3]:


if __name__ =='__main__':
    all_data = load_pickle(raw_data_path + 'all_data.pkl')
    all_data = add_feature_click_stats(all_data)
    print(all_data.columns)

