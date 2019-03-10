
# coding: utf-8

# In[1]:


import os
import pickle
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from utils import raw_data_path, feature_data_path,result_path,cache_pkl_path,dump_pickle,load_pickle
from utils import extract_ctr


# In[2]:


def add_cross_feature(data, feature_1, feature_2):
    comb_index = data[[feature_1, feature_2]].drop_duplicates()
    comb_index[feature_1 + '_' + feature_2] = np.arange(comb_index.shape[0])
    data = pd.merge(data, comb_index, 'left', on=[feature_1, feature_2])
    
    return data

def cut_features(data):
    
    features_to_cut = ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level',
                       'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level',
                       'shop_score_service', 'shop_score_delivery', 'shop_score_description',]
    
    for feature in features_to_cut:
        data[feature] = pd.qcut(data[feature], q=10, duplicates='drop')
        
    data['context_page_id'] = pd.qcut(data.context_page_id, q=5, duplicates='drop')
    data['hour'] = pd.cut(data.hour, bins=8)
    
    data.user_age_level.replace(to_replace=[-1,], value = data.user_age_level.mean(), inplace=True)
    data['user_age_level'] = pd.cut(data.user_age_level, bins=5)

    data.user_star_level.replace(to_replace=[-1,], value = data.user_star_level.mean(), inplace=True)
    data['user_star_level'] = pd.cut(data.user_star_level, bins=5)
    
    return data


def gen_feature_interaction_2_order():
    '''生成交叉特征，2 order

    文件名：feature_interaction_2_order.pkl

    '''
    data = load_pickle(raw_data_path + 'all_data_4567.pkl')
    data = cut_features(data)

    cross_features = list()

    feature_path = feature_data_path + 'feature_interaction_2_order.pkl'
    print('generating '+feature_path)

#     user与各种特征交叉
    for feature_1 in (['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']):
        for feature_2 in tqdm(['item_price_level', 'item_sales_level', 
                               'shop_star_level', 'shop_review_num_level', 'shop_review_positive_rate',
                               'category2_label', 'category3_label',
                               'context_page_id', 'hour','item_property_topic_k_10'
                              ]):

            data = add_cross_feature(data, feature_1, feature_2)
            cross_features.append(feature_1 + '_' + feature_2)
            
#     user自身特征交叉
    user_features = ['user_gender_id', 'user_age_level', 'user_occupation_id']
    for i, feature_1 in enumerate(user_features):
        for j, feature_2 in enumerate(user_features):
            if i < j:
                data = add_cross_feature(data, feature_1, feature_2)
                cross_features.append(feature_1 + '_' + feature_2)
    
    data = data[cross_features]

    dump_pickle(data, feature_path)


def add_feature_interaction_2_order(data):

    feature_path = feature_data_path + 'feature_interaction_2_order.pkl'
    if not os.path.exists(feature_path):
        gen_feature_interaction_2_order()

    cross_features = load_pickle(feature_path)
    data = pd.concat([data, cross_features], axis=1)

    return data


# ## 交叉特征与day交叉 

# In[8]:


def add_cross_feature_day(data, feature_1, feature_2):
    comb_index = data[[feature_1, feature_2, 'day']].drop_duplicates()
    comb_index[feature_1 + '_' + feature_2 + '_day'] = np.arange(comb_index.shape[0])
    data = pd.merge(data, comb_index, 'left', on=[feature_1, feature_2, 'day'])
    
    return data

def gen_feature_interaction_day_2_order():
    '''生成交叉特征，2 order

    文件名：feature_interaction_day_2_order.pkl

    '''
    data = load_pickle(raw_data_path + 'all_data_4567.pkl')
    
    data.user_age_level.replace(to_replace=[-1,], value = data.user_age_level.mean(), inplace=True)
    data['user_age_level'] = pd.cut(data.user_age_level, bins=5)

    cross_features = list()

    feature_path = feature_data_path + 'feature_interaction_day_2_order.pkl'
    print('generating '+feature_path)

#     user与各种特征交叉
    for feature_1 in (['user_gender_id', 'user_age_level', 'user_occupation_id',]):
        for feature_2 in tqdm(['item_price_level_bin', 
                               'category2_label', 'category3_label',
                               'item_property_topic_k_15'
                              ]):

            data = add_cross_feature_day(data, feature_1, feature_2)
            cross_features.append(feature_1 + '_' + feature_2 + '_day')
            
#     user自身特征交叉
    user_features = ['user_gender_id', 'user_age_level', 'user_occupation_id']
    for i, feature_1 in enumerate(user_features):
        for j, feature_2 in enumerate(user_features):
            if i < j:
                data = add_cross_feature_day(data, feature_1, feature_2)
                cross_features.append(feature_1 + '_' + feature_2 + '_day')
    
    data = data[cross_features]

    dump_pickle(data, feature_path)


def add_feature_interaction_day_2_order(data):

    feature_path = feature_data_path + 'feature_interaction_day_2_order.pkl'
    if not os.path.exists(feature_path):
        gen_feature_interaction_day_2_order()

    cross_features = load_pickle(feature_path)
    data = pd.concat([data, cross_features], axis=1)

    return data


# ## 单特征与day交叉 

# In[9]:


def add_feature_day(data, feature):
    comb_index = data[[feature, 'day']].drop_duplicates()
    comb_index[feature + '_day'] = np.arange(comb_index.shape[0])
    data = pd.merge(data, comb_index, 'left', on=[feature, 'day'])
    
    return data

def gen_feature_interaction_day_1_order():
    data = load_pickle(raw_data_path + 'all_data_4567.pkl')
    
    data.user_age_level.replace(to_replace=[-1,], value = data.user_age_level.mean(), inplace=True)
    data['user_age_level'] = pd.cut(data.user_age_level, bins=5)

    cross_features = list()

    feature_path = feature_data_path + 'feature_interaction_day_1_order.pkl'
    print('generating '+feature_path)

#     user与各种特征交叉
    for feature in tqdm(['user_gender_id', 'user_age_level', 'user_occupation_id', 
                     'item_price_level_bin', 'item_sales_level_bin', 'item_collected_level_bin', 'item_pv_level_bin',
                     'shop_review_num_level_bin', 'shop_review_positive_rate_bin', 'shop_star_level_bin',
                     'shop_score_service_bin', 'shop_score_delivery_bin', 'shop_score_description_bin',
                     'category2_label', 'category3_label', 'item_property_topic_k_15',
                     'context_page_id',
                    ]):


            data = add_feature_day(data, feature)
            cross_features.append(feature + '_day')

    data = data[cross_features]
    dump_pickle(data, feature_path)

def add_feature_interaction_day_1_order(data):

    feature_path = feature_data_path + 'feature_interaction_day_1_order.pkl'

    if not os.path.exists(feature_path):
        gen_feature_interaction_day_1_order()

    cross_features = load_pickle(feature_path)
    data = pd.concat([data, cross_features], axis=1)

    return data


# In[ ]:


if __name__ =='__main__':
    data = load_pickle(raw_data_path + 'all_data_4567.pkl')
#     data = add_feature_interaction_2_order(data)

    data = add_feature_interaction_day_1_order(data)
    data = add_feature_interaction_day_2_order(data)
    
    print(data.columns)

