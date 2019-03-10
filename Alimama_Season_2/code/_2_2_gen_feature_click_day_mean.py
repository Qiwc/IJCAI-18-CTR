
# coding: utf-8

# # 特征的平均日点击量

# In[1]:


import os
import pickle
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import load_pickle, dump_pickle, raw_data_path, feature_data_path


# In[2]:


def gen_feature_click_day_stats(data, feature):
    '''生成分类属性日点击量的统计特征

    '''

    data = data.copy()[[feature, 'day']]

    feature_click_day = pd.DataFrame(data.groupby(['day', feature]).size(
    )).reset_index().rename(columns={0: 'feature_click_day'})

    feature_click_day_mean = pd.DataFrame(feature_click_day.groupby([feature])['feature_click_day'].mean(
    )).rename(columns={'feature_click_day': feature + '_click_day_mean'}).reset_index()

    # 每个类别只保留一条记录
    data = data.drop(['day', ], axis=1)
    data = data.drop_duplicates([feature, ])
    data = pd.merge(data, feature_click_day_mean, how='left', on=feature)

    return data


def gen_feature_click_stats():
    """生成各个分类属性日点击量的统计特征

    file_name: (feature_id)_click_day_mean.pkl

    example:
        user_id_click_day_mean 该用户平均每天点击多少次

    features:
        'user_id_click_day_mean',  
        'item_id_click_day_mean', 
        'item_brand_id_click_day_mean', 
        'shop_id_click_day_mean', 
        'context_page_id_click_day_mean', 
        'category2_label_click_day_mean',
        'category2_label_click_day_mean',

    """

    all_data = load_pickle(raw_data_path + 'all_data.pkl')

    feature_list = ['user_id',
                    'category2_label', 'category3_label',
                    'shop_id', 'item_id', 'item_brand_id',
                    'context_page_id',
                    ]

    for feature in tqdm(feature_list):
        feature_path = feature_data_path +'_2_2_' + feature + '_click_day_mean.pkl'
        print('generating ' + feature_path)

        feature_stats = gen_feature_click_day_stats(all_data, feature)

        print(feature_stats.columns)
        dump_pickle(feature_stats, feature_path)


def add_feature_click_stats(data,):
    """添加分类属性日点击量的统计特征

    join_key: ['feature_id',]

    """

    feature_list = ['user_id',
                    'category2_label', 'category3_label',
                    'shop_id', 'item_id', 'item_brand_id',
                    'context_page_id',
                    ]

    for feature in tqdm(feature_list):
        feature_path = feature_data_path +'_2_2_'+ feature + '_click_day_mean.pkl'
        if not os.path.exists(feature_path):
            gen_feature_click_stats()
            
        feature_click_day_stats = load_pickle(feature_path)
        data = pd.merge(data, feature_click_day_stats, 'left', [feature, ])

    return data


# In[ ]:


if __name__ =='__main__':
    data = load_pickle(raw_data_path + 'all_data_4567.pkl')
    data = add_feature_click_stats(data)
    print(data.columns)


