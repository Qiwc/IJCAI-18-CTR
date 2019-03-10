
# coding: utf-8

# In[1]:


import os
import pickle
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from utils import raw_data_path, feature_data_path,result_path,cache_pkl_path,dump_pickle,load_pickle
from smooth import BayesianSmoothing


# # 弃用

# In[2]:


from sklearn.feature_extraction.text import TfidfTransformer
def gen_TfidfTransformer():
    '''
    分别groupby['shop_id'], ['item_id'], ['item_brand_id']
    计算用户在['user_gender_id', 'user_age_level', 'user_occupation_id']几个属性下的点击量（one_hot）
    
    计算的是每天的

    文件名：['shop_id', 'item_id', 'item_brand_id']_CountVector.pkl
    '''

    TF_IDF = TfidfTransformer('l2')

    
    all_data = load_pickle(raw_data_path+'all_data.pkl')    
    for feature in tqdm(['shop_id', 'item_id', 'item_brand_id']):   
        for one_hot_feature in (['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']):
            feature_path = feature_data_path+'TfidfTransformer'+'_'+feature+'_'+one_hot_feature+'.pkl' #要存放的目录
            if os.path.exists(feature_path):
                print('found ' + feature_path)
            else:
                print('generating ' + feature_path)  
                data = all_data[[feature,one_hot_feature]]
                data_hot = pd.get_dummies(data,prefix_sep='_tfid_'+feature+'_', dummy_na=True, columns=[one_hot_feature])
                data_sum = data_hot.groupby([feature]).sum()
                
                vec_columns = data_sum.columns
                local_tfidf_vec = TF_IDF.fit_transform(data_sum).toarray()
                local_tfidf_vec = pd.DataFrame(local_tfidf_vec,columns=vec_columns,index=data_sum.index).reset_index()
                dump_pickle(local_tfidf_vec,feature_path)  #存储
                
                
def add_TfidfTransformer(all_data):
    '''
    向总体数据添加特征
    feature=['item_id', 'item_brand_id', 'shop_id']
    拼接键[feature, 'day']
    '''
    for feature in tqdm(['item_id', 'item_brand_id', 'shop_id']):  
        for one_hot_feature in (['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']):
            feature_path = feature_data_path+'TfidfTransformer'+'_'+feature+'_'+one_hot_feature+'.pkl'
            if not os.path.exists(feature_path):
                gen_TfidfTransformer()
            CountVector_data = load_pickle(feature_path)
            all_data = pd.merge(all_data, CountVector_data, how='left', on=[feature])
    return all_data    


# In[3]:


def gen_feature_user_property():
    '''
    分别groupby['shop_id'], ['item_id'], ['item_brand_id']
    计算item在['user_gender_id', 'user_age_level', 'user_occupation_id']几个属性下的点击量
    
    文件名：feature_user_property_click.pkl
    
    '''
    all_data = load_pickle(raw_data_path+'all_data.pkl')
    for feature in tqdm(['shop_id', 'item_id', 'item_brand_id','category2_label', 'category3_label','hour'
                             ,'item_sales_level_bin', 'item_price_level_bin']):
        for user_property in (['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']):

            feature_path = feature_data_path +'_2_6_'+feature +                 '_' + user_property + '_click.pkl'  # 要存放的目录
            if os.path.exists(feature_path):
                print('found ' + feature_path)
            else:
                print('generating ' + feature_path)

                feature_user_property_click_feat = feature + '_' + user_property + '_click'
                feature_click_feat = feature + '_click'
                feature_user_property_click_rate_feat = feature +                     '_' + user_property + '_click_rate'

                data = all_data[[feature, user_property]]
                feature_user_property_click = data.groupby([feature, user_property]).size(
                ).reset_index().rename(columns={0: feature_user_property_click_feat})
                feature_click = data.groupby([feature]).size(
                ).reset_index().rename(columns={0: feature_click_feat})

                feature_user_property_click_rate = pd.merge(
                    feature_click, feature_user_property_click, how='inner', on=[feature])
                
                
#                 考虑添加平滑
                feature_user_property_click_rate[feature_user_property_click_rate_feat] = feature_user_property_click_rate[
                    feature_user_property_click_feat] / feature_user_property_click_rate[feature_click_feat]
                
                feature_user_property_click_rate = feature_user_property_click_rate[[feature, user_property, feature_user_property_click_rate_feat]]
 
                dump_pickle(feature_user_property_click_rate, feature_path)


def add_feature_user_property(all_data):

    for feature in tqdm(['shop_id', 'item_id', 'item_brand_id','category2_label', 'category3_label','hour'
                             , 'item_sales_level_bin', 'item_price_level_bin']):
        for user_property in (['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']):
            feature_path = feature_data_path +'_2_6_'+feature +                 '_' + user_property + '_click.pkl'  # 要存放的目录
            if not os.path.exists(feature_path):
                gen_feature_user_property()
            else:
                feature_user_property_click_rate = load_pickle(feature_path)
                all_data = pd.merge(all_data, feature_user_property_click_rate, 'left', [feature, user_property])
        
    return all_data


# In[2]:


def gen_feature_user_property_hour():
    '''
    分别groupby['shop_id'], ['item_id'], ['item_brand_id']
    计算item在['user_gender_id', 'user_age_level', 'user_occupation_id']几个属性下的点击量
    
    文件名：feature_user_property_click.pkl
    
    '''
    all_data = load_pickle(raw_data_path+'all_data.pkl')
    for feature in tqdm(['shop_id', 'item_id', 'item_brand_id','category2_label', 'category3_label'
                             ,'item_sales_level_bin', 'item_price_level_bin']):
        for user_property in (['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']):

            feature_path = feature_data_path +'_2_6_'+feature +                 '_' + user_property + '_click_hour.pkl'  # 要存放的目录
            if os.path.exists(feature_path):
                print('found ' + feature_path)
            else:
                print('generating ' + feature_path)

                feature_user_property_click_feat = feature + '_' + user_property + '_click_hour'
                feature_click_feat = feature + '_click_hour'
                feature_user_property_click_rate_feat = feature +                     '_' + user_property + '_click_rate_hour'

                data = all_data[[feature, user_property, 'hour_bin']]
                feature_user_property_click = data.groupby([feature, user_property, 'hour_bin']).size(
                ).reset_index().rename(columns={0: feature_user_property_click_feat})
                feature_click = data.groupby([feature, 'hour_bin']).size(
                ).reset_index().rename(columns={0: feature_click_feat})

                feature_user_property_click_rate = pd.merge(
                    feature_click, feature_user_property_click, how='inner', on=[feature, 'hour_bin'])
                
                
#                 考虑添加平滑
                feature_user_property_click_rate[feature_user_property_click_rate_feat] = feature_user_property_click_rate[
                    feature_user_property_click_feat] / feature_user_property_click_rate[feature_click_feat]
                
                feature_user_property_click_rate = feature_user_property_click_rate[[feature, user_property, feature_user_property_click_rate_feat, 'hour_bin']]
 
                dump_pickle(feature_user_property_click_rate, feature_path)


def add_feature_user_property_hour(all_data):

    for feature in tqdm(['shop_id', 'item_id', 'item_brand_id','category2_label', 'category3_label'
                             , 'item_sales_level_bin', 'item_price_level_bin']):
        for user_property in (['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']):
            feature_path = feature_data_path +'_2_6_'+feature +                 '_' + user_property + '_click_hour.pkl'  # 要存放的目录
            if not os.path.exists(feature_path):
                gen_feature_user_property_hour()
            else:
                feature_user_property_click_rate = load_pickle(feature_path)
                all_data = pd.merge(all_data, feature_user_property_click_rate, 'left', [feature, user_property, 'hour_bin'])
        
    return all_data


# In[ ]:


if __name__ =='__main__':
    all_data = load_pickle(raw_data_path+'all_data_4567.pkl')
    
#     all_data = add_feature_user_property(all_data)
    all_data = add_feature_user_property_hour(all_data)
    print(all_data.columns)  

