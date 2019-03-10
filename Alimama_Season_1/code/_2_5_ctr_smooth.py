
# coding: utf-8

# In[18]:


import os
import pickle
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from utils import raw_data_path, feature_data_path,result_path,cache_pkl_path,dump_pickle,load_pickle
from smooth import BayesianSmoothing
from _2_4_gen_predict_category_property import add_category_predict_rank


# # 单特征smooth平滑ctr。历史

# In[25]:


def gen_features_smooth_ctr():
    '''
    贝叶斯平滑版
    提取每天前些天的，分别以feature=['user_id', 'item_id', 'item_brand_id', 'shop_id']分类的，总点击次数_I,总购买次数_C,点击率_CTR
    以['day', feature, I_alias, C_alias, CTR_alias]存储
    文件名，【】_CTR.pkl
    '''
    all_data = load_pickle(raw_data_path+'all_data.pkl')    
    all_data = add_category_predict_rank(all_data)
    for feature in tqdm(['user_id', 'category_predict_rank', 'user_occupation_id', 'user_age_level', 'user_gender_id', 'user_star_level',
                         'item_id', 'item_brand_id', 'item_city_id', 'category2_label','item_price_level','item_sales_level', 
                         'item_collected_level', 'item_pv_level',
                         'context_page_id', 
                         'shop_id', 'shop_review_num_level', 'shop_star_level', 'hour']):  
       
        feature_path = feature_data_path+feature+'_smooth_CTR.pkl' #要存放的目录
        if os.path.exists(feature_path):
            print('found ' + feature_path)
        else:
            
            alpha_beta_dispaly = []
            
            print('generating ' + feature_path)
            I_alias = feature+'_smooth_I' #总点击次数
            C_alias = feature+'_smooth_C' #购买次数
            CTR_alias = feature+'_smooth_CTR'
            history_ctr = pd.DataFrame()
            for day in range(19,26):
                
                history_data = all_data[all_data['day'] < day]
                I = history_data.groupby([feature]).size().reset_index().rename(columns={0: I_alias})
                C = history_data[history_data['is_trade'] == 1].groupby([feature]).size().reset_index().rename(columns={0: C_alias})
                CTR = pd.merge(I, C, how='left', on=[feature])
                CTR[C_alias] = CTR[C_alias].fillna(0)
                
                hyper = BayesianSmoothing(1, 1)
                hyper.update(CTR[I_alias].values, CTR[C_alias].values, 10000, 0.0000001)
                alpha = hyper.alpha
                beta = hyper.beta
                
                alpha_beta_dispaly.append(alpha)
                alpha_beta_dispaly.append(beta)
                
                print(feature)
                print(alpha_beta_dispaly)
                dump_pickle(alpha_beta_dispaly,feature_data_path+'1'+feature+'.pkl')  #存储
                
                CTR[CTR_alias] = (CTR[C_alias] + alpha) / (CTR[I_alias] + alpha + beta)
                CTR['day'] = day
                history_ctr = history_ctr.append(CTR)
            
            print('-----------------------------------------------------------------------')
            print(feature)
            print(alpha_beta_dispaly)
            dump_pickle(alpha_beta_dispaly,feature_data_path+'1'+feature+'.pkl')  #存储
            print('-----------------------------------------------------------------------')
            dump_pickle(history_ctr[['day', feature, I_alias, C_alias, CTR_alias]],feature_path)  #存储

def add_features_smooth_ctr(all_data):
    '''
    向总体数据添加特征
    feature=['user_id', 'item_id', 'item_brand_id', 'shop_id', 'context_page_id', 'category2_label', 'item_price_level', 'category_predict_rank']
    拼接键[feature, 'day']
    '''
    for feature in tqdm(['user_id', 'category_predict_rank', 'user_occupation_id', 'user_age_level', 'user_gender_id', 'user_star_level',
                         'item_id', 'item_brand_id', 'item_city_id', 'category2_label','item_price_level','item_sales_level', 
                         'item_collected_level', 'item_pv_level',
                         'context_page_id', 
                         'shop_id', 'shop_review_num_level', 'shop_star_level', 'hour']):  
        feature_path = feature_data_path+feature+'_smooth_CTR.pkl'
        if not os.path.exists(feature_path):
            gen_features_smooth_ctr()
        ctr_data = load_pickle(feature_path)
        all_data = pd.merge(all_data, ctr_data, how='left', on=[feature, 'day'])
    return all_data       


# # 单特征无平滑ctr前一天

# In[26]:


def gen_features_day_ctr():
    '''
    贝叶斯平滑版
    提取每天前些天的，分别以feature=['user_id', 'item_id', 'item_brand_id', 'shop_id']分类的，总点击次数_I,总购买次数_C,点击率_CTR
    以['day', feature, I_alias, C_alias, CTR_alias]存储
    文件名，【】_CTR.pkl
    '''
    all_data = load_pickle(raw_data_path+'all_data.pkl')    
    all_data = add_category_predict_rank(all_data)
    for feature in tqdm(['user_id', 'category_predict_rank', 'user_occupation_id', 'user_age_level', 'user_gender_id', 'user_star_level',
                         'item_id', 'item_brand_id', 'item_city_id', 'category2_label','item_price_level','item_sales_level', 
                         'item_collected_level', 'item_pv_level',
                         'context_page_id', 
                         'shop_id', 'shop_review_num_level', 'shop_star_level',]):  
        feature_path = feature_data_path+feature+'_day_CTR.pkl' #要存放的目录
        if os.path.exists(feature_path):
            print('found ' + feature_path)
        else:
            
            alpha_beta_dispaly = []
            
            print('generating ' + feature_path)
            I_alias = feature+'_day_I' #总点击次数
            C_alias = feature+'_day_C' #购买次数
            CTR_alias = feature+'_CTR'
            history_ctr = pd.DataFrame()
            for day in range(19,26):
                
                history_data = all_data[all_data['day'] == day-1]
                I = history_data.groupby([feature]).size().reset_index().rename(columns={0: I_alias})
                C = history_data[history_data['is_trade'] == 1].groupby([feature]).size().reset_index().rename(columns={0: C_alias})
                CTR = pd.merge(I, C, how='left', on=[feature])
                CTR[C_alias] = CTR[C_alias].fillna(0)
                
                CTR[CTR_alias] = CTR[C_alias] / CTR[I_alias]
                CTR['day'] = day
                history_ctr = history_ctr.append(CTR)
            
#             print('-----------------------------------------------------------------------')
#             print(feature)
#             print(alpha_beta_dispaly)
#             dump_pickle(alpha_beta_dispaly,feature_data_path+'1_day_'+feature+'.pkl')  #存储
#             print('-----------------------------------------------------------------------')
            dump_pickle(history_ctr[['day', feature, I_alias, C_alias, CTR_alias]],feature_path)  #存储

def add_features_day_ctr(all_data):
    '''
    向总体数据添加特征
    feature=['user_id', 'item_id', 'item_brand_id', 'shop_id', 'context_page_id', 'category2_label', 'item_price_level', 'category_predict_rank']
    拼接键[feature, 'day']
    '''
    for feature in tqdm(['user_id', 'category_predict_rank', 'user_occupation_id', 'user_age_level', 'user_gender_id', 'user_star_level',
                         'item_id', 'item_brand_id', 'item_city_id', 'category2_label','item_price_level','item_sales_level', 
                         'item_collected_level', 'item_pv_level',
                         'context_page_id', 
                         'shop_id', 'shop_review_num_level', 'shop_star_level',]):  
        feature_path = feature_data_path+feature+'_day_CTR.pkl'
        if not os.path.exists(feature_path):
            gen_features_day_ctr()
        ctr_data = load_pickle(feature_path)
        all_data = pd.merge(all_data, ctr_data, how='left', on=[feature, 'day'])
        
        all_data[feature+'_day_I'] = all_data[feature+'_day_I'].fillna(0)
        all_data[feature+'_day_C'] = all_data[feature+'_day_C'].fillna(0)
    return all_data       


# # user_id前一天点击某某某的数量

# In[27]:


def gen_features_cross_day_ctr():
    '''
    贝叶斯平滑版
    提取每天前些天的，分别以feature=['user_id', 'item_id', 'item_brand_id', 'shop_id']分类的，总点击次数_I,总购买次数_C,点击率_CTR
    以['day', feature, I_alias, C_alias, CTR_alias]存储
    文件名，【】_CTR.pkl
    '''
    all_data = load_pickle(raw_data_path+'all_data.pkl')    
 
    for feature in tqdm(['user_id']):
   
        for feature2 in tqdm(['item_id', 'item_brand_id', 'shop_id', 'category2_label', 'item_price_level', ]):  
            
            feature_path = feature_data_path+feature+'_'+feature2+'_before_day_CTR.pkl' #要存放的目录
            if os.path.exists(feature_path):
                print('found ' + feature_path)
            else:

                print('generating ' + feature_path)
                I_alias = feature+'_'+feature2+'_day_I' #总点击次数
                C_alias = feature+'_'+feature2+'_day_C' #购买次数
             
                history_ctr = pd.DataFrame()
                for day in range(19,26):

                    history_data = all_data[all_data['day'] == day-1]
                    I = history_data.groupby([feature, feature2]).size().reset_index().rename(columns={0: I_alias})
                    C = history_data[history_data['is_trade'] == 1].groupby([feature, feature2]).size().reset_index().rename(columns={0: C_alias})
                    CTR = pd.merge(I, C, how='left', on=[feature, feature2])
                    CTR[C_alias] = CTR[C_alias].fillna(0)
                    CTR['day'] = day
                    history_ctr = history_ctr.append(CTR)

                dump_pickle(history_ctr[['day', feature, feature2, I_alias, C_alias]],feature_path)  #存储

def add_features_cross_day_ctr(all_data):
    '''
    向总体数据添加特征
    feature=['user_id', 'item_id', 'item_brand_id', 'shop_id', 'context_page_id', 'category2_label', 'item_price_level', 'category_predict_rank']
    拼接键[feature, 'day']
    '''
    for feature in tqdm(['user_id',]):
   
        for feature2 in tqdm(['item_id', 'item_brand_id', 'shop_id', 'category2_label', 'item_price_level',]):  
            
            I_alias = feature+'_'+feature2+'_day_I' #总点击次数
            C_alias = feature+'_'+feature2+'_day_C' #购买次数
            feature_path = feature_data_path+feature+'_'+feature2+'_before_day_CTR.pkl' #要存放的目录
       
            if not os.path.exists(feature_path):
                gen_features_cross_day_ctr()
            ctr_data = load_pickle(feature_path)
            all_data = pd.merge(all_data, ctr_data, how='left', on=[feature, feature2, 'day'])
            all_data[I_alias] = all_data[I_alias].fillna(0)
            all_data[C_alias] = all_data[C_alias].fillna(0)
    return all_data       


# # user_id历史点击某某某的数量

# In[28]:


def gen_features_cross_history_ctr():
    '''
    贝叶斯平滑版
    提取每天前些天的，分别以feature=['user_id', 'item_id', 'item_brand_id', 'shop_id']分类的，总点击次数_I,总购买次数_C,点击率_CTR
    以['day', feature, I_alias, C_alias, CTR_alias]存储
    文件名，【】_CTR.pkl
    '''
    all_data = load_pickle(raw_data_path+'all_data.pkl')    
 
    for feature in tqdm(['user_id']):
   
        for feature2 in tqdm(['item_id', 'item_brand_id', 'shop_id', 'category2_label', 'item_price_level',]):  
            
            feature_path = feature_data_path+feature+'_'+feature2+'_before_history_CTR.pkl' #要存放的目录
            if os.path.exists(feature_path):
                print('found ' + feature_path)
            else:

                print('generating ' + feature_path)
                I_alias = feature+'_'+feature2+'_history_I' #总点击次数
                C_alias = feature+'_'+feature2+'_history_C' #购买次数
             
                history_ctr = pd.DataFrame()
                for day in range(19,26):

                    history_data = all_data[all_data['day'] <= day-1]
                    I = history_data.groupby([feature, feature2]).size().reset_index().rename(columns={0: I_alias})
                    C = history_data[history_data['is_trade'] == 1].groupby([feature, feature2]).size().reset_index().rename(columns={0: C_alias})
                    CTR = pd.merge(I, C, how='left', on=[feature, feature2])
                    CTR[C_alias] = CTR[C_alias].fillna(0)
                    CTR['day'] = day
                    history_ctr = history_ctr.append(CTR)

                dump_pickle(history_ctr[['day', feature, feature2, I_alias, C_alias]],feature_path)  #存储

def add_features_cross_history_ctr(all_data):
    '''
    向总体数据添加特征
    feature=['user_id', 'item_id', 'item_brand_id', 'shop_id', 'context_page_id', 'category2_label', 'item_price_level', 'category_predict_rank']
    拼接键[feature, 'day']
    '''
    for feature in tqdm(['user_id',]):
   
        for feature2 in tqdm(['item_id', 'item_brand_id', 'shop_id', 'category2_label', 'item_price_level',]):  
            
            I_alias = feature+'_'+feature2+'_history_I' #总点击次数
            C_alias = feature+'_'+feature2+'_history_C' #购买次数
            feature_path = feature_data_path+feature+'_'+feature2+'_before_history_CTR.pkl' #要存放的目录
       
            if not os.path.exists(feature_path):
                gen_features_cross_history_ctr()
            ctr_data = load_pickle(feature_path)
            all_data = pd.merge(all_data, ctr_data, how='left', on=[feature, feature2, 'day'])
            all_data[I_alias] = all_data[I_alias].fillna(0)
            all_data[C_alias] = all_data[C_alias].fillna(0)
    return all_data       


# # 特征交叉，有平滑，历史

# In[29]:


def gen_features_cross_smooth_ctr():
    '''
    贝叶斯平滑版
    提取每天前些天的，分别以feature=['user_id', 'item_id', 'item_brand_id', 'shop_id']分类的，总点击次数_I,总购买次数_C,点击率_CTR
    以['day', feature, I_alias, C_alias, CTR_alias]存储
    文件名，【】_CTR.pkl
    '''
    all_data = load_pickle(raw_data_path+'all_data.pkl')    
 
    for feature in tqdm(['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']):
   
        for feature2 in tqdm(['item_id', 'item_brand_id', 'shop_id', 'item_price_level', 'hour']):  
            
            feature_path = feature_data_path+feature+'_'+feature2+'_smooth_CTR.pkl' #要存放的目录
            if os.path.exists(feature_path):
                print('found ' + feature_path)
            else:

                alpha_beta_dispaly = []

                print('generating ' + feature_path)
                I_alias = feature+'_'+feature2+'_smooth_I' #总点击次数
                C_alias = feature+'_'+feature2+'_smooth_C' #购买次数
                CTR_alias = feature+'_'+feature2+'_smooth_CTR'
                history_ctr = pd.DataFrame()
                for day in range(19,26):

                    history_data = all_data[all_data['day'] < day]
                    I = history_data.groupby([feature, feature2]).size().reset_index().rename(columns={0: I_alias})
                    C = history_data[history_data['is_trade'] == 1].groupby([feature, feature2]).size().reset_index().rename(columns={0: C_alias})
                    CTR = pd.merge(I, C, how='left', on=[feature, feature2])
                    CTR[C_alias] = CTR[C_alias].fillna(0)

                    hyper = BayesianSmoothing(1, 1)
                    hyper.update(CTR[I_alias].values, CTR[C_alias].values, 1000, 0.000001)
                    alpha = hyper.alpha
                    beta = hyper.beta

                    alpha_beta_dispaly.append(alpha)
                    alpha_beta_dispaly.append(beta)

                    print(feature)
                    print(alpha_beta_dispaly)
                    dump_pickle(alpha_beta_dispaly,feature_data_path+'1'+feature+'_'+feature2+'.pkl')  #存储

                    CTR[CTR_alias] = (CTR[C_alias] + alpha) / (CTR[I_alias] + alpha + beta)
                    CTR['day'] = day
                    history_ctr = history_ctr.append(CTR)

                print('-----------------------------------------------------------------------')
                print(feature)
                print(alpha_beta_dispaly)
                dump_pickle(alpha_beta_dispaly,feature_data_path+'1'+feature+'_'+feature2+'.pkl')  #存储
                print('-----------------------------------------------------------------------')
                dump_pickle(history_ctr[['day', feature, feature2, I_alias, C_alias, CTR_alias]],feature_path)  #存储

def add_features_cross_smooth_ctr(all_data):
    '''
    向总体数据添加特征
    feature=['user_id', 'item_id', 'item_brand_id', 'shop_id', 'context_page_id', 'category2_label', 'item_price_level', 'category_predict_rank']
    拼接键[feature, 'day']
    '''
    for feature in tqdm(['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']):
   
        for feature2 in tqdm(['item_id', 'item_brand_id', 'shop_id', 'item_price_level', 'hour']):  
            
            feature_path = feature_data_path+feature+'_'+feature2+'_smooth_CTR.pkl' #要存放的目录
       
            if not os.path.exists(feature_path):
                gen_features_cross_smooth_ctr()
            ctr_data = load_pickle(feature_path)
            all_data = pd.merge(all_data, ctr_data, how='left', on=[feature, feature2, 'day'])
    return all_data     


# # 测试

# In[30]:


if __name__ =='__main__':
    all_data = load_pickle(raw_data_path+'all_data.pkl')
    
    all_data = add_category_predict_rank(all_data)
    all_data = add_features_smooth_ctr(all_data)
    all_data = add_features_day_ctr(all_data)
    all_data = add_features_cross_day_ctr(all_data)
    all_data = add_features_cross_smooth_ctr(all_data)
    all_data = add_features_cross_history_ctr(all_data)
    
    print(all_data.columns)  


# In[ ]:




