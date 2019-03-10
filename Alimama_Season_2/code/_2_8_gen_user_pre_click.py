
# coding: utf-8

# In[1]:


import os
import pickle
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import load_pickle, dump_pickle, raw_data_path, feature_data_path


# In[2]:


pre_user_id = None
pre_feature = None
continue_cnt = 0 

def get_user_feature_pre_click(row, feature):
    
    global pre_user_id
    global pre_feature
    global continue_cnt
    
    if row['user_id'] == pre_user_id: 
        if row[feature] == pre_feature:
            # 该用户当前点击与上次一样的feature
            continue_cnt += 1
            return 1
        else:
            # 记录用户当前点击的物品
            pre_feature = row[feature]
            return 0
    
    else:
        # 上一个用户已经计算完成
        pre_user_id = row['user_id']
        pre_feature = row[feature]
        return 0
    
def get_user_feature_continue_click(row, feature):
    
    global pre_user_id
    global pre_feature
    global continue_cnt
    
    if row['user_id'] == pre_user_id: 
        if row[feature] == pre_feature:
            # 该用户当前点击与上次一样的feature
            continue_cnt += 1
            return continue_cnt
        else:
            # 记录用户当前点击的物品
            pre_feature = row[feature]
            continue_cnt = 1
            return continue_cnt
    
    else:
        # 上一个用户已经计算完成
        pre_user_id = row['user_id']
        pre_feature = row[feature]
        continue_cnt = 1
        return continue_cnt


def gen_user_feature_pre_click(update=True):
    '''用户当前点击与上次一样的feature

    file_name: user_feature_pre_click.pkl

    features:
        'user_item_id_pre_click', 'user_item_brand_id_pre_click',
        'user_shop_id_pre_click', 'user_category2_label_pre_click',

    '''

    all_data = load_pickle(raw_data_path + 'all_data_4567.pkl')
    feature_list = ['item_id', 'item_brand_id', 'shop_id', 'category2_label', 'category3_label',
                    'item_property_topic_k_15',
                    'item_sales_level_bin', 'item_price_level_bin']

    for feature in tqdm(feature_list):

        feature_path = feature_data_path +'_2_8'+ 'user_'+feature+'_pre_click.pkl'

        if os.path.exists(feature_path):
            print('found '+feature_path)
        else:
            print('generating '+feature_path)

            pre_click_feature_name = 'user_' + feature + '_pre_click'
            continue_click_feature_name = 'user_' + feature + '_continue_click'
            

            # 用户点击时间戳排序
            sorted_data = all_data.sort_values(
                by=['user_id', 'context_timestamp'], ascending=True)[['user_id', feature, 'context_timestamp']]
            
            sorted_data[pre_click_feature_name] = sorted_data.apply(lambda row: get_user_feature_pre_click(row, feature), axis=1)
            sorted_data[continue_click_feature_name] = sorted_data.apply(lambda row: get_user_feature_continue_click(row, feature), axis=1)
            
            sorted_data = sorted_data[[pre_click_feature_name, continue_click_feature_name]]
            
            dump_pickle(sorted_data, feature_path)


def add_user_feature_pre_click(data):


    feature_list = ['item_id', 'item_brand_id', 'shop_id', 'category2_label', 'category3_label',
                    'item_property_topic_k_15',
                    'item_sales_level_bin', 'item_price_level_bin']

    for feature in tqdm(feature_list):
        feature_path = feature_data_path +'_2_8'+ 'user_'+feature+'_pre_click.pkl'
        if not os.path.exists(feature_path):
            gen_user_feature_pre_click()
        user_feature_pre_click = load_pickle(feature_path)
        data = data.join(user_feature_pre_click)

    return data


# # user click interval

# In[3]:


if __name__ =='__main__':
    all_data = load_pickle(raw_data_path + 'all_data_4567.pkl')

    all_data = add_user_feature_pre_click(all_data)


