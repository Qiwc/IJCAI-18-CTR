
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


def gen_feature_click_day(update=True):
    '''
    计算feature=['user_id', 'item_id', 'item_brand_id', 'shop_id', 'user_gender_id', 'context_page_id',
                        'user_occupation_id', 'user_age_level']的点击量
    计算的是每天的

    文件名：[feature]_clicks_day.pkl
    '''

    all_data = load_pickle(raw_data_path + 'all_data.pkl')

    for feature in tqdm(['user_id', 'user_occupation_id', 'user_age_level', 'user_gender_id', 'user_star_level',
                         'item_id', 'item_brand_id', 'item_city_id', 'category2_label','item_price_level','item_sales_level', 
                         'item_collected_level', 'item_pv_level',
                         'context_page_id', 
                         'shop_id', 'shop_review_num_level', 'shop_star_level',]):

        feature_path = feature_data_path+feature+'_clicks_day.pkl'  # 要存放的目录
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

    for feature in tqdm(['user_id', 'user_occupation_id', 'user_age_level', 'user_gender_id', 'user_star_level',
                         'item_id', 'item_brand_id', 'item_city_id', 'category2_label','item_price_level','item_sales_level', 
                         'item_collected_level', 'item_pv_level',
                         'context_page_id', 
                         'shop_id', 'shop_review_num_level', 'shop_star_level',]):
        feature_path = feature_data_path + feature + '_clicks_day.pkl'
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

    all_data = load_pickle(raw_data_path+'all_data.pkl')
    for feature in tqdm(['user_id', 'user_occupation_id', 'user_age_level', 'user_gender_id', 'user_star_level',
                         'item_id', 'item_brand_id', 'item_city_id', 'category2_label','item_price_level','item_sales_level', 
                         'item_collected_level', 'item_pv_level',
                         'context_page_id', 
                         'shop_id', 'shop_review_num_level', 'shop_star_level',]):
        feature_path = feature_data_path+feature+'_click_day_hour.pkl'  # 要存放的目录
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
    for feature in tqdm(['user_id', 'user_occupation_id', 'user_age_level', 'user_gender_id', 'user_star_level',
                         'item_id', 'item_brand_id', 'item_city_id', 'category2_label','item_price_level','item_sales_level', 
                         'item_collected_level', 'item_pv_level',
                         'context_page_id', 
                         'shop_id', 'shop_review_num_level', 'shop_star_level',]):
        feature_path = feature_data_path+feature+'_click_day_hour.pkl'
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
    for feature in tqdm(['user_id', 'user_occupation_id', 'user_age_level', 'user_gender_id', 'user_star_level',
                         'item_id', 'item_brand_id', 'item_city_id', 'category2_label','item_price_level','item_sales_level', 
                         'item_collected_level', 'item_pv_level',
                         'context_page_id', 
                         'shop_id', 'shop_review_num_level', 'shop_star_level',]):
        feature_path = feature_data_path+feature+'_click_hour.pkl'  # 要存放的目录
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
    for feature in tqdm(['user_id', 'user_occupation_id', 'user_age_level', 'user_gender_id', 'user_star_level',
                         'item_id', 'item_brand_id', 'item_city_id', 'category2_label','item_price_level','item_sales_level', 
                         'item_collected_level', 'item_pv_level',
                         'context_page_id', 
                         'shop_id', 'shop_review_num_level', 'shop_star_level',]):
        feature_path = feature_data_path+feature+'_click_hour.pkl'
        if not os.path.exists(feature_path):
            gen_feature_click_hour()
        feature_click_hour = load_pickle(feature_path)
        all_data = pd.merge(all_data, feature_click_hour,
                            how='left', on=[feature, 'hour'])
    return all_data


# # 计算历史的，考虑只计算前一天的 放弃

# In[5]:


def gen_feature_click_history(update=True):
    '''
    计算feature=['user_id', 'item_id', 'item_brand_id', 'shop_id', 'user_gender_id', 'context_page_id',
                        'user_occupation_id', 'user_age_level']的点击量
    计算的是每天的

    文件名：[feature]_click_history.pkl
    
    features:
        'user_id_click_history', 'item_id_click_history',
       'item_brand_id_click_history', 'shop_id_click_history',
       'user_gender_id_click_history', 'context_page_id_click_history',
       'user_occupation_id_click_history', 'user_age_level_click_history'
    
    '''

    all_data = load_pickle(raw_data_path+'all_data.pkl')    
    for feature in tqdm(['user_id', 'item_id', 'item_brand_id', 'shop_id', 'user_gender_id', 'context_page_id',
                        'user_occupation_id', 'user_age_level', 'category2_label', 'item_sales_level', 'item_price_level', 'user_star_level']):        
        feature_path = feature_data_path+feature+'_click_history.pkl' #要存放的目录
        if os.path.exists(feature_path) and update == False:
            print('found ' + feature_path)
        else:
            print('generating ' + feature_path)        
            data = pd.DataFrame()
            for day in range(18,26):               
                now_data = all_data[all_data['day'] <= day]            
                feature_click_history = now_data.groupby([feature]).size().reset_index().rename(columns={0: feature+'_click_history'})       
                feature_click_history['day'] = day
                data = data.append(feature_click_history)
            dump_pickle(data,feature_path)  #存储

            
def add_feature_click_history(all_data):
    '''
    向总体数据添加特征
    feature=['user_id', 'item_id', 'item_brand_id', 'shop_id', 'user_gender_id', 'context_page_id',
                        'user_occupation_id', 'user_age_level', 'category2_label']
    拼接键[feature, 'day']
    '''
    
    for feature in tqdm(['user_id', 'item_id', 'item_brand_id', 'shop_id', 'user_gender_id', 'context_page_id',
                        'user_occupation_id', 'user_age_level', 'category2_label', 'item_sales_level', 'item_price_level', 'user_star_level']):  
        feature_path = feature_data_path+feature+'_click_history.pkl'
        if not os.path.exists(feature_path):
            gen_feature_click_history()
        Clicks_data = load_pickle(feature_path)
        all_data = pd.merge(all_data, Clicks_data, how='left', on=[feature, 'day'])
    
    return all_data   


# In[6]:


if __name__ =='__main__':
    

    
    all_data = load_pickle(raw_data_path + 'all_data.pkl')
    all_data = add_feature_click_day(all_data)
    all_data = add_feature_click_hour(all_data)
    all_data = add_feature_click_day_hour(all_data)
    
    print(all_data.columns)

