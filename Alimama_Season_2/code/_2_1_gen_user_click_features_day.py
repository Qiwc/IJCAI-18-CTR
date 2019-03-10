
# coding: utf-8

# # 用户当天点击特征的次数

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


def gen_user_feature_click_day():
    """生成用户对所有分类属性的当天点击量
    """
    data = load_pickle(raw_data_path + 'all_data_4567.pkl')

    feature_list = ['category2_label', 'category3_label',
                    'shop_id', 'item_id', 'item_brand_id',
                    'context_page_id', 
                    'item_price_level_bin', 'item_sales_level_bin', 
                    'item_property_topic_k_15',
                    ]

    for feature in tqdm(feature_list):
        feature_path = feature_data_path +'_2_1_'+'user_' + feature + '_click_day.pkl'
        
        if os.path.exists(feature_path):
            print('found ' + feature_path)   
        else:
            print('generating '+feature_path)

            user_feature_click_day = data.groupby(['user_id', 'day', feature]).size(
            ).reset_index().rename(columns={0: 'user_'+feature+'_click_day'})
            dump_pickle(user_feature_click_day, feature_path)


def add_user_feature_click_day(data):
    """添加用户对所有分类属性的当天点击量

    join_key: ['user_id', 'feature_id', 'day']

    """

    feature_list = ['category2_label', 'category3_label',
                    'shop_id', 'item_id', 'item_brand_id',
                    'context_page_id', 
                    'item_price_level_bin', 'item_sales_level_bin', 
                    'item_property_topic_k_15',
                    ]

    for feature in tqdm(feature_list):
        feature_path = feature_data_path +'_2_1_'+ 'user_'+feature+'_click_day.pkl'
        if not os.path.exists(feature_path):
            gen_user_feature_click_day()
            
        user_feature_click_day = load_pickle(feature_path)
        data = pd.merge(data, user_feature_click_day,
                        'left', [feature, 'day', 'user_id'])

    return data


# In[4]:


def gen_user_feature_click_hour():
    """生成用户对所有分类属性的当前小时点击量
    """

    data = load_pickle(raw_data_path + 'all_data_4567.pkl')

    feature_list = ['category2_label', 'category3_label',
                    'shop_id', 'item_id', 'item_brand_id',
                    'context_page_id', 
                    'item_price_level_bin', 'item_sales_level_bin', 
                    'item_property_topic_k_15',
                    ]

    for feature in tqdm(feature_list):
        feature_path = feature_data_path +'_2_1_'+ 'user_'+feature+'_click_hour.pkl'
        if os.path.exists(feature_path):
            print('found ' + feature_path)   
        else:        
            print('generating '+feature_path)

            user_feature_click_hour = data.groupby(['user_id', 'day', 'hour', feature]).size(
            ).reset_index().rename(columns={0: 'user_' + feature + '_click_hour'})
            dump_pickle(user_feature_click_hour, feature_path)


def add_user_feature_click_hour(data):
    """添加用户对所有分类属性的当天点击统计量

    join_key: ['user_id', 'feature_id', 'day', 'hour']

    """

    feature_list = ['category2_label', 'category3_label',
                    'shop_id', 'item_id', 'item_brand_id',
                    'context_page_id', 
                    'item_price_level_bin', 'item_sales_level_bin', 
                    'item_property_topic_k_15',
                    ]

    for feature in tqdm(feature_list):
        feature_path = feature_data_path +'_2_1_'+ 'user_' +feature+'_click_hour.pkl'
        if not os.path.exists(feature_path):
            gen_user_feature_click_hour()
            
        user_feature_click_hour = load_pickle(feature_path)
        data = pd.merge(data, user_feature_click_hour, 'left', [feature, 'day', 'hour', 'user_id'])

    return data


# In[ ]:


if __name__ =='__main__':
    
    data = load_pickle(raw_data_path + 'all_data_4567.pkl')
    
    data = add_user_feature_click_day(data)
    data = add_user_feature_click_hour(data)

    print(data.columns)


