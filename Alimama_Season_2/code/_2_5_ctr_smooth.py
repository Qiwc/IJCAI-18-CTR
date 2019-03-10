
# coding: utf-8

# In[2]:


import os
import pickle
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from utils import raw_data_path, feature_data_path,result_path,cache_pkl_path,dump_pickle,load_pickle
from smooth import BayesianSmoothing


# # 2-9历史CTR

# In[3]:


def gen_29_smooth_ctr():
    all_data = load_pickle(raw_data_path+'all_data.pkl')    
    for feature_1 in (['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']):
        for feature_2 in tqdm(['item_price_level', 'item_sales_level', 
                                'shop_star_level', 'shop_review_num_level', 'shop_review_positive_rate',
                               'category2_label', 'category3_label',
                              ]):

            feature_path = feature_data_path+'_2_5_'+feature_1 + '_' + feature_2+'_CTR.pkl' #要存放的目录
            if os.path.exists(feature_path):
                print('found ' + feature_path)
            else:
                feature = feature_1 + '_' + feature_2
                print('generating ' + feature_path)
                I_alias = feature+'_smooth_I' #总点击次数
                C_alias = feature+'_smooth_C' #购买次数
                CTR_alias = feature+'_smooth_CTR'
                history_ctr = pd.DataFrame()
                for day in range(4,8):            
                    history_data = all_data[all_data['day'] < day]
                    I = history_data.groupby([feature]).size().reset_index().rename(columns={0: I_alias})
                    C = history_data[history_data['is_trade'] == 1].groupby([feature]).size().reset_index().rename(columns={0: C_alias})
                    CTR = pd.merge(I, C, how='left', on=[feature])
                    CTR[C_alias] = CTR[C_alias].fillna(0)
                    CTR[CTR_alias] = (CTR[C_alias]) / (CTR[I_alias])
                    CTR['day'] = day
                    history_ctr = history_ctr.append(CTR)

#                 dump_pickle(history_ctr[['day', feature, I_alias, C_alias, CTR_alias]],feature_path)  #存储
                dump_pickle(history_ctr[['day', feature, CTR_alias]],feature_path)  #存储
                   
    #     user自身特征交叉
    user_features = ['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
    for i, feature_1 in enumerate(user_features):
        for j, feature_2 in enumerate(user_features):
            if i < j:
                feature_path = feature_data_path+'_2_5_'+feature_1 + '_' + feature_2+'_CTR.pkl' #要存放的目录
                if os.path.exists(feature_path):
                    print('found ' + feature_path)
                else:
                    feature = feature_1 + '_' + feature_2
                    print('generating ' + feature_path)
                    I_alias = feature+'_smooth_I' #总点击次数
                    C_alias = feature+'_smooth_C' #购买次数
                    CTR_alias = feature+'_smooth_CTR'
                    history_ctr = pd.DataFrame()
                    for day in range(4,8):            
                        history_data = all_data[all_data['day'] < day]
                        I = history_data.groupby([feature]).size().reset_index().rename(columns={0: I_alias})
                        C = history_data[history_data['is_trade'] == 1].groupby([feature]).size().reset_index().rename(columns={0: C_alias})
                        CTR = pd.merge(I, C, how='left', on=[feature])
                        CTR[C_alias] = CTR[C_alias].fillna(0)
                        CTR[CTR_alias] = (CTR[C_alias]) / (CTR[I_alias])
                        CTR['day'] = day
                        history_ctr = history_ctr.append(CTR)

                    dump_pickle(history_ctr[['day', feature, CTR_alias]],feature_path)  #存储
                
                
                

def add_29_smooth_ctr(all_data):
    for feature_1 in (['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']):
        for feature_2 in tqdm(['item_price_level', 'item_sales_level', 
                               'shop_star_level', 'shop_review_num_level', 'shop_review_positive_rate',
                               'category2_label', 'category3_label',
                              ]):
            feature_path = feature_data_path+'_2_5_'+feature_1 + '_' + feature_2+'_CTR.pkl' #要存放的目录
            if not os.path.exists(feature_path):
                gen_29_smooth_ctr()
            ctr_data = load_pickle(feature_path)
            feature = feature_1 + '_' + feature_2
            all_data = pd.merge(all_data, ctr_data, how='left', on=[feature, 'day'])
#             all_data[feature+'_smooth_I'] = all_data[feature+'_smooth_I'].fillna(0)
#             all_data[feature+'_smooth_C'] = all_data[feature+'_smooth_C'].fillna(0)
            
     #     user自身特征交叉
    user_features = ['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
    for i, feature_1 in enumerate(user_features):
        for j, feature_2 in enumerate(user_features):
            if i < j:
                feature_path = feature_data_path+'_2_5_'+feature_1 + '_' + feature_2+'_CTR.pkl' #要存放的目录
                if not os.path.exists(feature_path):
                    gen_29_smooth_ctr()
                ctr_data = load_pickle(feature_path)
                feature = feature_1 + '_' + feature_2
                all_data = pd.merge(all_data, ctr_data, how='left', on=[feature, 'day'])
#                 all_data[feature+'_smooth_I'] = all_data[feature+'_smooth_I'].fillna(0)
#                 all_data[feature+'_smooth_C'] = all_data[feature+'_smooth_C'].fillna(0)
            
    return all_data       


# # 2-9hour CTR

# In[4]:


def gen_29_hour_ctr():
    all_data = load_pickle(raw_data_path+'all_data.pkl')    
    for feature_1 in (['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']):
        for feature_2 in tqdm(['item_price_level', 'item_sales_level', 
                                'shop_star_level', 'shop_review_num_level', 'shop_review_positive_rate',
                               'category2_label', 'category3_label',
                              ]):

            feature_path = feature_data_path+'_2_5_'+feature_1 + '_' + feature_2+'_hour_CTR.pkl' #要存放的目录
            if os.path.exists(feature_path):
                print('found ' + feature_path)
            else:
                feature = feature_1 + '_' + feature_2
                print('generating ' + feature_path)
                I_alias = feature+'_hour_I' #总点击次数
                C_alias = feature+'_hour_C' #购买次数
                CTR_alias = feature+'_hour_CTR'
                history_ctr = pd.DataFrame()
                for day in range(4,8):            
                    history_data = all_data[all_data['day'] < day]
                    I = history_data.groupby([feature, 'hour_bin']).size().reset_index().rename(columns={0: I_alias})
                    C = history_data[history_data['is_trade'] == 1].groupby([feature, 'hour_bin']).size().reset_index().rename(columns={0: C_alias})
                    CTR = pd.merge(I, C, how='left', on=[feature, 'hour_bin'])
                    CTR[C_alias] = CTR[C_alias].fillna(0)
                    CTR[CTR_alias] = (CTR[C_alias]) / (CTR[I_alias])
                    CTR['day'] = day
                    history_ctr = history_ctr.append(CTR)

                dump_pickle(history_ctr[['day', 'hour_bin', feature, CTR_alias]],feature_path)  #存储
                   
    #     user自身特征交叉
    user_features = ['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
    for i, feature_1 in enumerate(user_features):
        for j, feature_2 in enumerate(user_features):
            if i < j:
                feature_path = feature_data_path+'_2_5_'+feature_1 + '_' + feature_2+'_hour_CTR.pkl' #要存放的目录
                if os.path.exists(feature_path):
                    print('found ' + feature_path)
                else:
                    feature = feature_1 + '_' + feature_2
                    print('generating ' + feature_path)
                    I_alias = feature+'_hour_I' #总点击次数
                    C_alias = feature+'_hour_C' #购买次数
                    CTR_alias = feature+'_hour_CTR'
                    history_ctr = pd.DataFrame()
                    for day in range(4,8):            
                        history_data = all_data[all_data['day'] < day]
                        I = history_data.groupby([feature, 'hour_bin']).size().reset_index().rename(columns={0: I_alias})
                        C = history_data[history_data['is_trade'] == 1].groupby([feature, 'hour_bin']).size().reset_index().rename(columns={0: C_alias})
                        CTR = pd.merge(I, C, how='left', on=[feature, 'hour_bin'])
                        CTR[C_alias] = CTR[C_alias].fillna(0)
                        CTR[CTR_alias] = (CTR[C_alias]) / (CTR[I_alias])
                        CTR['day'] = day
                        history_ctr = history_ctr.append(CTR)

                    dump_pickle(history_ctr[['day', 'hour_bin', feature, CTR_alias]],feature_path)  #存储
                
                
                

def add_29_hour_ctr(all_data):
    for feature_1 in (['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']):
        for feature_2 in tqdm(['item_price_level', 'item_sales_level', 
                                'shop_star_level', 'shop_review_num_level', 'shop_review_positive_rate',
                               'category2_label', 'category3_label',
                              ]):
            feature_path = feature_data_path+'_2_5_'+feature_1 + '_' + feature_2+'_hour_CTR.pkl' #要存放的目录
            if not os.path.exists(feature_path):
                gen_29_hour_ctr()
            ctr_data = load_pickle(feature_path)
            feature = feature_1 + '_' + feature_2
            all_data = pd.merge(all_data, ctr_data, how='left', on=[feature, 'day', 'hour_bin'])
#             all_data[feature+'_hour_I'] = all_data[feature+'_hour_I'].fillna(0)
#             all_data[feature+'_hour_C'] = all_data[feature+'_hour_C'].fillna(0)
            
     #     user自身特征交叉
    user_features = ['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']
    for i, feature_1 in enumerate(user_features):
        for j, feature_2 in enumerate(user_features):
            if i < j:
                feature_path = feature_data_path+'_2_5_'+feature_1 + '_' + feature_2+'_hour_CTR.pkl' #要存放的目录
                if not os.path.exists(feature_path):
                    gen_29_hour_ctr()
                ctr_data = load_pickle(feature_path)
                feature = feature_1 + '_' + feature_2
                all_data = pd.merge(all_data, ctr_data, how='left', on=[feature, 'day', 'hour_bin'])
#                 all_data[feature+'_hour_I'] = all_data[feature+'_hour_I'].fillna(0)
#                 all_data[feature+'_hour_C'] = all_data[feature+'_hour_C'].fillna(0)
            
    return all_data     


# # 单特征历史CTR

# In[5]:


def gen_features_smooth_ctr():
    all_data = load_pickle(raw_data_path+'all_data.pkl')    
    for feature in tqdm(['user_id', 
                         'item_id', 'item_brand_id',
                         'category2_label', 'category3_label',
                         'context_page_id', 
                         'shop_id',
                         'item_sales_level_bin', 'item_price_level_bin','item_collected_level_bin','item_pv_level_bin',
                         'shop_review_num_level_bin', 'shop_review_positive_rate_bin', 'shop_star_level_bin',
                         'shop_score_service_bin', 'shop_score_delivery_bin', 'shop_score_description_bin',
                         'hour'
                        ]):    
        feature_path = feature_data_path+'_2_5_'+feature+'_smooth_CTR.pkl' #要存放的目录
        if os.path.exists(feature_path):
            print('found ' + feature_path)
        else:
            print('generating ' + feature_path)
            I_alias = feature+'_smooth_I' #总点击次数
            C_alias = feature+'_smooth_C' #购买次数
            CTR_alias = feature+'_smooth_CTR'
            history_ctr = pd.DataFrame()
            for day in range(4,8):            
                history_data = all_data[all_data['day'] < day]
                I = history_data.groupby([feature]).size().reset_index().rename(columns={0: I_alias})
                C = history_data[history_data['is_trade'] == 1].groupby([feature]).size().reset_index().rename(columns={0: C_alias})
                CTR = pd.merge(I, C, how='left', on=[feature])
                CTR[C_alias] = CTR[C_alias].fillna(0)
                CTR[CTR_alias] = (CTR[C_alias]) / (CTR[I_alias])
                CTR['day'] = day
                history_ctr = history_ctr.append(CTR)
 
            dump_pickle(history_ctr[['day', feature, I_alias, C_alias, CTR_alias]],feature_path)  #存储

def add_features_smooth_ctr(all_data):
    for feature in tqdm(['user_id', 
                         'item_id', 'item_brand_id',
                         'category2_label', 'category3_label',
                         'context_page_id', 
                         'shop_id',
                         'item_sales_level_bin', 'item_price_level_bin','item_collected_level_bin','item_pv_level_bin',
                         'shop_review_num_level_bin', 'shop_review_positive_rate_bin', 'shop_star_level_bin',
                         'shop_score_service_bin', 'shop_score_delivery_bin', 'shop_score_description_bin',
                         'hour'
                        ]):  
        feature_path = feature_data_path+'_2_5_'+feature+'_smooth_CTR.pkl'
        if not os.path.exists(feature_path):
            gen_features_smooth_ctr()
        ctr_data = load_pickle(feature_path)
        all_data = pd.merge(all_data, ctr_data, how='left', on=[feature, 'day'])
        all_data[feature+'_smooth_I'] = all_data[feature+'_smooth_I'].fillna(0)
        all_data[feature+'_smooth_C'] = all_data[feature+'_smooth_C'].fillna(0)
    return all_data       


# # 单特征与hour组合

# In[6]:


def gen_features_hour_ctr():
    all_data = load_pickle(raw_data_path+'all_data.pkl')    
    for feature in tqdm(['user_id', 
                         'item_id', 'item_brand_id',
                         'category2_label', 'category3_label',
                         'context_page_id', 
                         'shop_id',
                         'item_sales_level_bin', 'item_price_level_bin','item_collected_level_bin','item_pv_level_bin',
                         'shop_review_num_level_bin', 'shop_review_positive_rate_bin', 'shop_star_level_bin',
                         'shop_score_service_bin', 'shop_score_delivery_bin', 'shop_score_description_bin',
                        ]):    
        feature_path = feature_data_path+'_2_5_'+feature+'_hour_CTR.pkl' #要存放的目录
        if os.path.exists(feature_path):
            print('found ' + feature_path)
        else:
            print('generating ' + feature_path)
            I_alias = feature+'_hour_I' #总点击次数
            C_alias = feature+'_hour_C' #购买次数
            CTR_alias = feature+'_hour_CTR'
            history_ctr = pd.DataFrame()
            for day in range(4,8):            
                history_data = all_data[all_data['day'] < day]
                I = history_data.groupby([feature, 'hour_bin']).size().reset_index().rename(columns={0: I_alias})
                C = history_data[history_data['is_trade'] == 1].groupby([feature, 'hour_bin']).size().reset_index().rename(columns={0: C_alias})
                CTR = pd.merge(I, C, how='left', on=[feature, 'hour_bin'])
                CTR[C_alias] = CTR[C_alias].fillna(0)
                CTR[CTR_alias] = (CTR[C_alias]) / (CTR[I_alias])
                CTR['day'] = day
                history_ctr = history_ctr.append(CTR)
 
            dump_pickle(history_ctr[['day', 'hour_bin', feature,CTR_alias]],feature_path)  #存储

def add_features_hour_ctr(all_data):
    for feature in tqdm(['user_id', 
                         'item_id', 'item_brand_id',
                         'category2_label', 'category3_label',
                         'context_page_id', 
                         'shop_id',
                         'item_sales_level_bin', 'item_price_level_bin','item_collected_level_bin','item_pv_level_bin',
                         'shop_review_num_level_bin', 'shop_review_positive_rate_bin', 'shop_star_level_bin',
                         'shop_score_service_bin', 'shop_score_delivery_bin', 'shop_score_description_bin',
                        ]):  
        feature_path = feature_data_path+'_2_5_'+feature+'_hour_CTR.pkl'
        if not os.path.exists(feature_path):
            gen_features_hour_ctr()
        ctr_data = load_pickle(feature_path)
        all_data = pd.merge(all_data, ctr_data, how='left', on=[feature, 'day', 'hour_bin'])

    return all_data       


# # user_id历史点击某某某的数量

# In[7]:


def gen_features_cross_history_ctr():
    all_data = load_pickle(raw_data_path+'all_data.pkl')    
    for feature in tqdm(['user_id']):  
        for feature2 in tqdm(['item_id', 'item_brand_id',
                             'category2_label', 'category3_label', 
                             'shop_id', 'item_sales_level_bin', 'item_price_level_bin']):            
            feature_path = feature_data_path+'_2_5_'+feature+'_'+feature2+'_before_history_CTR.pkl' #要存放的目录
            if os.path.exists(feature_path):
                print('found ' + feature_path)
            else:

                print('generating ' + feature_path)
                I_alias = feature+'_'+feature2+'_history_I' #总点击次数
                C_alias = feature+'_'+feature2+'_history_C' #购买次数
                CTR_alias = feature+'_'+feature2+'_history_CTR'
                history_ctr = pd.DataFrame()
                for day in range(4,8):

                    history_data = all_data[all_data['day'] <= day-1]
                    I = history_data.groupby([feature, feature2]).size().reset_index().rename(columns={0: I_alias})
                    C = history_data[history_data['is_trade'] == 1].groupby([feature, feature2]).size().reset_index().rename(columns={0: C_alias})
                    CTR = pd.merge(I, C, how='left', on=[feature, feature2])
                    CTR[C_alias] = CTR[C_alias].fillna(0)
                    CTR[CTR_alias] = CTR[C_alias] / CTR[I_alias]
                    CTR['day'] = day
                    history_ctr = history_ctr.append(CTR)

                dump_pickle(history_ctr[['day', feature, feature2, I_alias, C_alias, CTR_alias]],feature_path)  #存储

def add_features_cross_history_ctr(all_data):
    for feature in tqdm(['user_id',]):
   
        for feature2 in tqdm(['item_id', 'item_brand_id',
                         'category2_label', 'category3_label', 
                         'shop_id', 'item_sales_level_bin', 'item_price_level_bin']):  
            
            I_alias = feature+'_'+feature2+'_history_I' #总点击次数
            C_alias = feature+'_'+feature2+'_history_C' #购买次数
            feature_path = feature_data_path+'_2_5_'+feature+'_'+feature2+'_before_history_CTR.pkl' #要存放的目录
       
            if not os.path.exists(feature_path):
                gen_features_cross_history_ctr()
            ctr_data = load_pickle(feature_path)
            all_data = pd.merge(all_data, ctr_data, how='left', on=[feature, feature2, 'day'])
            all_data[I_alias] = all_data[I_alias].fillna(0)
            all_data[C_alias] = all_data[C_alias].fillna(0)
    return all_data       


# # 特征交叉

# In[ ]:


def gen_features_cross_smooth_ctr():
    '''
    贝叶斯平滑版
    提取每天前些天的，分别以feature=['user_id', 'item_id', 'item_brand_id', 'shop_id']分类的，总点击次数_I,总购买次数_C,点击率_CTR
    以['day', feature, I_alias, C_alias, CTR_alias]存储
    文件名，【】_CTR.pkl
    '''
    all_data = load_pickle(raw_data_path+'all_data.pkl')    
 
    for feature in tqdm(['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']):
   
        for feature2 in tqdm(['shop_id', 'item_id', 'item_brand_id',]):  
            
            feature_path = feature_data_path+feature+'_'+feature2+'_smooth_CTR.pkl' #要存放的目录
            if os.path.exists(feature_path):
                print('found ' + feature_path)
            else:

                print('generating ' + feature_path)
                I_alias = feature+'_'+feature2+'_smooth_I' #总点击次数
                C_alias = feature+'_'+feature2+'_smooth_C' #购买次数
                CTR_alias = feature+'_'+feature2+'_smooth_CTR'
                history_ctr = pd.DataFrame()
                for day in range(4,8):

                    history_data = all_data[all_data['day'] < day]
                    I = history_data.groupby([feature, feature2]).size().reset_index().rename(columns={0: I_alias})
                    C = history_data[history_data['is_trade'] == 1].groupby([feature, feature2]).size().reset_index().rename(columns={0: C_alias})
                    CTR = pd.merge(I, C, how='left', on=[feature, feature2])
                    CTR[C_alias] = CTR[C_alias].fillna(0)

                    CTR[CTR_alias] = (CTR[C_alias]) / (CTR[I_alias])
                    CTR['day'] = day
                    history_ctr = history_ctr.append(CTR)
                dump_pickle(history_ctr[['day', feature, feature2, CTR_alias]],feature_path)  #存储

def add_features_cross_smooth_ctr(all_data):
    '''
    向总体数据添加特征
    feature=['user_id', 'item_id', 'item_brand_id', 'shop_id', 'context_page_id', 'category2_label', 'item_price_level', 'category_predict_rank']
    拼接键[feature, 'day']
    '''
    for feature in tqdm(['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']):
   
        for feature2 in tqdm(['shop_id', 'item_id', 'item_brand_id',]):  
            
            feature_path = feature_data_path+feature+'_'+feature2+'_smooth_CTR.pkl' #要存放的目录
       
            if not os.path.exists(feature_path):
                gen_features_cross_smooth_ctr()
            ctr_data = load_pickle(feature_path)
            all_data = pd.merge(all_data, ctr_data, how='left', on=[feature, feature2, 'day'])
    return all_data     


# # 测试

# In[ ]:


if __name__ =='__main__':

    all_data = load_pickle(raw_data_path + 'all_data_4567.pkl')
    

    
    all_data = add_29_smooth_ctr(all_data)
    all_data = add_29_hour_ctr(all_data)
    all_data = add_features_smooth_ctr(all_data)
    all_data = add_features_hour_ctr(all_data) 
    all_data = add_features_cross_history_ctr(all_data)
    all_data = add_features_cross_smooth_ctr(all_data)
    
    
    
    print(all_data.columns)  



