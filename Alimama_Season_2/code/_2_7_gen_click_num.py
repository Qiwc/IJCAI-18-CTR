
# coding: utf-8

# In[1]:


import os
import pickle
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from utils import raw_data_path, feature_data_path,result_path,cache_pkl_path,dump_pickle,load_pickle


# # 计算每天的

# In[2]:


def gen_cross_feature_click_day(update=True):
    
    all_data = load_pickle(raw_data_path + 'all_data_4567.pkl')

    for feature_1 in (['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']):
        for feature_2 in tqdm(['item_price_level', 'category2_label', 'category3_label',
                              ]):

            feature_path = feature_data_path+'_2_7_'+feature_1 + '_' + feature_2+'_clicks_day.pkl' #要存放的目录
            if os.path.exists(feature_path) and update==False:
                print('found ' + feature_path)
            else:
                feature = feature_1 + '_' + feature_2
                print('generating ' + feature_path)
                feature_click_day = pd.DataFrame(all_data.groupby(['day', feature]).size(
                )).reset_index().rename(columns={0: feature + '_click_day'})
                dump_pickle(feature_click_day, feature_path)
               
    #     user自身特征交叉
    user_features = ['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
    for i, feature_1 in enumerate(user_features):
        for j, feature_2 in enumerate(user_features):
            if i < j:
                feature_path = feature_data_path+'_2_7_'+feature_1 + '_' + feature_2+'_clicks_day.pkl' #要存放的目录
                if os.path.exists(feature_path) and update==False:
                    print('found ' + feature_path)
                else:
                    feature = feature_1 + '_' + feature_2
                    print('generating ' + feature_path)
                    feature_click_day = pd.DataFrame(all_data.groupby(['day', feature]).size(
                    )).reset_index().rename(columns={0: feature + '_click_day'})
                    dump_pickle(feature_click_day, feature_path)
                    
    for feature_1 in tqdm(['user_gender_id', 'user_age_level', 'user_occupation_id']):
        for feature_2 in tqdm(['shop_id', 'item_id', 'item_brand_id',]):  
            feature_path = feature_data_path+'_2_7_'+feature_1 + '_' + feature_2+'_clicks_day.pkl' #要存放的目录
            if os.path.exists(feature_path) and update==False:
                print('found ' + feature_path)
            else:
                feature = feature_1 + '_' + feature_2
                print('generating ' + feature_path)
                feature_click_day = pd.DataFrame(all_data.groupby(['day', feature_1, feature_2]).size(
                )).reset_index().rename(columns={0: feature + '_click_day'})
                dump_pickle(feature_click_day, feature_path)


def add_cross_feature_click_day(all_data):


    
    for feature_1 in (['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']):
        for feature_2 in tqdm(['item_price_level', 'category2_label', 'category3_label',
                              ]):

            feature_path = feature_data_path+'_2_7_'+feature_1 + '_' + feature_2+'_clicks_day.pkl' #要存放的目录
            if not os.path.exists(feature_path):
                gen_cross_feature_click_day()
            feature = feature_1 + '_' + feature_2
            feature_click_day = load_pickle(feature_path)
            all_data = pd.merge(all_data, feature_click_day, how='left', on=[feature, 'day'])
               
    #     user自身特征交叉
    user_features = ['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
    for i, feature_1 in enumerate(user_features):
        for j, feature_2 in enumerate(user_features):
            if i < j:
                feature_path = feature_data_path+'_2_7_'+feature_1 + '_' + feature_2+'_clicks_day.pkl' #要存放的目录
                if not os.path.exists(feature_path):
                    gen_cross_feature_click_day()
                feature = feature_1 + '_' + feature_2
                feature_click_day = load_pickle(feature_path)
                all_data = pd.merge(all_data, feature_click_day, how='left', on=[feature, 'day'])
                
    for feature_1 in tqdm(['user_gender_id', 'user_age_level', 'user_occupation_id']):
        for feature_2 in tqdm(['shop_id', 'item_id', 'item_brand_id',]):  
            feature_path = feature_data_path+'_2_7_'+feature_1 + '_' + feature_2+'_clicks_day.pkl' #要存放的目录
            if not os.path.exists(feature_path):
                gen_cross_feature_click_day()
            feature_click_day = load_pickle(feature_path)
            all_data = pd.merge(all_data, feature_click_day, how='left', on=[feature_1, feature_2, 'day'])
    
    
    return all_data


def gen_feature_click_day(update=True):
    '''
    计算feature=['user_id', 'item_id', 'item_brand_id', 'shop_id', 'user_gender_id', 'context_page_id',
                        'user_occupation_id', 'user_age_level','item_property_topic_k_10']的点击量
    计算的是每天的

    文件名：[feature]_clicks_day.pkl
    '''

    all_data = load_pickle(raw_data_path + 'all_data_4567.pkl')

    for feature in tqdm(['user_id', 
                         'item_id', 'item_brand_id',
                         'category2_label', 'category3_label',
                         'context_page_id', 
                         'shop_id', 
                         'item_property_topic_k_15'
                        ]):

        feature_path = feature_data_path+'_2_7_'+feature+'_clicks_day.pkl'  # 要存放的目录
        if os.path.exists(feature_path) and update == False:
            print('found ' + feature_path)
        else:
            print('generating ' + feature_path)
            feature_click_day = pd.DataFrame(all_data.groupby(['day', feature]).size(
            )).reset_index().rename(columns={0: feature + '_click_day'})
            dump_pickle(feature_click_day, feature_path)


def add_feature_click_day(all_data):
    '''
    向总体数据添加特征
    feature=['user_id', 'item_id', 'item_brand_id', 'shop_id', 'user_gender_id', 'context_page_id',
                        'user_occupation_id', 'user_age_level']
    拼接键[feature, 'day']
    '''

    for feature in tqdm(['user_id', 
                         'item_id', 'item_brand_id',
                         'category2_label', 'category3_label',
                         'context_page_id', 
                         'shop_id', 
                         'item_property_topic_k_15'
                        ]):
        feature_path = feature_data_path+'_2_7_' + feature + '_clicks_day.pkl'
        if not os.path.exists(feature_path):
            gen_feature_click_day()
        feature_click_day = load_pickle(feature_path)
        all_data = pd.merge(all_data, feature_click_day,
                            how='left', on=[feature, 'day'])
    return all_data


# # 计算每天每小时的

# In[3]:


def gen_feature_click_day_hour(update=True):
    '''
    计算feature=['user_id', 'item_id', 'item_brand_id', 'shop_id', 'user_gender_id', 'context_page_id',
                        'user_occupation_id', 'user_age_level']的点击量
    计算的是每天每小时

    文件名：[feature]_click_hour.pkl
    '''

    all_data = load_pickle(raw_data_path+'all_data_4567.pkl')
    
    for feature in tqdm(['user_id', 
                         'item_id', 'item_brand_id',
                         'category2_label', 'category3_label',
                         'context_page_id', 
                         'shop_id', 
                         'item_property_topic_k_15'
                        ]):
        feature_path = feature_data_path+'_2_7_'+feature+'_click_day_hour.pkl'  # 要存放的目录
        if os.path.exists(feature_path) and update == False:
            print('found ' + feature_path)
        else:
            print('generating ' + feature_path)
            feature_click_day_hour = all_data.groupby([feature, 'day', 'hour']).size(
            ).reset_index().rename(columns={0: feature+'_click_hour'})
            dump_pickle(feature_click_day_hour, feature_path)  # 存储


def add_feature_click_day_hour(all_data):
    '''
    向总体数据添加特征
    feature=['user_id', 'item_id', 'item_brand_id', 'shop_id', 'user_gender_id', 'context_page_id',
                        'user_occupation_id', 'user_age_level', 'category2_label']
    拼接键[feature, 'day']
    '''
    for feature in tqdm(['user_id', 
                         'item_id', 'item_brand_id',
                         'category2_label', 'category3_label',
                         'context_page_id', 
                         'shop_id', 
                         'item_property_topic_k_15'
                        ]):
        feature_path = feature_data_path+'_2_7_'+feature+'_click_day_hour.pkl'
        if not os.path.exists(feature_path):
            gen_feature_click_day_hour()
        feature_click_day_hour = load_pickle(feature_path)
        all_data = pd.merge(all_data, feature_click_day_hour,
                            how='left', on=[feature, 'day', 'hour'])
    return all_data


# # 计算每小时的

# In[4]:


def gen_feature_click_hour(update=True):
    '''
    计算feature=['user_id', 'item_id', 'item_brand_id', 'shop_id', 'user_gender_id', 'context_page_id',
                        'user_occupation_id', 'user_age_level']的点击量
    计算的是每天每小时

    文件名：[feature]_click_hour.pkl
    '''

    all_data = load_pickle(raw_data_path+'all_data.pkl')
    for feature in tqdm(['user_id', 'user_occupation_id', 'user_age_level', 'user_gender_id', 
                         'item_id', 'item_brand_id','category2_label','category3_label',
                         'context_page_id', 
                         'shop_id'] ):
        feature_path = feature_data_path+'_2_7_'+feature+'_click_hour.pkl'  # 要存放的目录
        if os.path.exists(feature_path) and update == False:
            print('found ' + feature_path)
        else:
            print('generating ' + feature_path)
            feature_click_hour = all_data.groupby([feature, 'hour']).size(
            ).reset_index().rename(columns={0: feature+'_click_hour'})
            dump_pickle(feature_click_hour, feature_path)  # 存储


def add_feature_click_hour(all_data):
    '''
    向总体数据添加特征
    feature=['user_id', 'item_id', 'item_brand_id', 'shop_id', 'user_gender_id', 'context_page_id',
                        'user_occupation_id', 'user_age_level', 'category2_label']
    拼接键[feature, 'day']
    '''
    for feature in tqdm(['user_id', 'user_occupation_id', 'user_age_level', 'user_gender_id', 
                         'item_id', 'item_brand_id','category2_label','category3_label',
                         'context_page_id', 
                         'shop_id']):
        feature_path = feature_data_path+'_2_7_'+feature+'_click_hour.pkl'
        if not os.path.exists(feature_path):
            gen_feature_click_hour()
        feature_click_hour = load_pickle(feature_path)
        all_data = pd.merge(all_data, feature_click_hour,
                            how='left', on=[feature, 'hour'])
    return all_data


# In[ ]:


if __name__ =='__main__':  
    all_data = load_pickle(raw_data_path + 'all_data_4567.pkl')
    
    all_data = add_feature_click_day(all_data)
#    all_data = add_feature_click_hour(all_data)
    all_data = add_feature_click_day_hour(all_data)
    
    print(all_data.columns)


