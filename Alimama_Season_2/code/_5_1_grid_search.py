
# coding: utf-8

# In[6]:


import os
import zipfile
import time
import pickle
import gc

import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import load_pickle, dump_pickle, get_feature_value, feature_spearmanr, feature_target_spearmanr, addCrossFeature, calibration
from utils import raw_data_path, feature_data_path, cache_pkl_path, analyse

from sklearn.metrics import log_loss
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit


# In[7]:


def CustomCV(data,):    
    fold_index_train = data[(data.hour < 11) & (data.hour > 0)].index
    fold_index_test = data[data.hour >= 11].index
    yield fold_index_train, fold_index_test
    


# In[8]:


all_data_path = feature_data_path + 'all_data_all_features_new.pkl'
all_data = load_pickle(all_data_path)

target = 'is_trade'

features = load_pickle('all_features_day_7_0511.pkl')
categorical_feature = load_pickle('categorical_feature.pkl')


len(features), len(categorical_feature)


hour_features=[
    'user_gender_id_item_price_level_hour_CTR',
        'user_gender_id_item_sales_level_hour_CTR',
        'user_gender_id_shop_star_level_hour_CTR',
        'user_gender_id_shop_review_num_level_hour_CTR',
        'user_gender_id_shop_review_positive_rate_hour_CTR',
        'user_gender_id_category2_label_hour_CTR',
        'user_gender_id_category3_label_hour_CTR',
        'user_age_level_item_price_level_hour_CTR',
        'user_age_level_item_sales_level_hour_CTR',
        'user_age_level_shop_star_level_hour_CTR',
        'user_age_level_shop_review_num_level_hour_CTR',
        'user_age_level_shop_review_positive_rate_hour_CTR',
        'user_age_level_category2_label_hour_CTR',
        'user_age_level_category3_label_hour_CTR',
        'user_occupation_id_item_price_level_hour_CTR',
        'user_occupation_id_item_sales_level_hour_CTR',
        'user_occupation_id_shop_star_level_hour_CTR',
        'user_occupation_id_shop_review_num_level_hour_CTR',
        'user_occupation_id_shop_review_positive_rate_hour_CTR',
        'user_occupation_id_category2_label_hour_CTR',
        'user_occupation_id_category3_label_hour_CTR',
        'user_star_level_item_price_level_hour_CTR',
        'user_star_level_item_sales_level_hour_CTR',
        'user_star_level_shop_star_level_hour_CTR',
        'user_star_level_shop_review_num_level_hour_CTR',
        'user_star_level_shop_review_positive_rate_hour_CTR',
        'user_star_level_category2_label_hour_CTR',
        'user_star_level_category3_label_hour_CTR',
        'user_gender_id_user_age_level_hour_CTR',
        'user_gender_id_user_occupation_id_hour_CTR',
        'user_gender_id_user_star_level_hour_CTR',
        'user_age_level_user_occupation_id_hour_CTR',
        'user_age_level_user_star_level_hour_CTR',
        'user_occupation_id_user_star_level_hour_CTR',
    
        'user_id_hour_CTR',
         'item_id_hour_CTR',
         'item_brand_id_hour_CTR',
         'category2_label_hour_CTR',
         'category3_label_hour_CTR',
         'context_page_id_hour_CTR',
         'shop_id_hour_CTR',
         'item_sales_level_bin_hour_CTR',
         'item_price_level_bin_hour_CTR',
         'item_collected_level_bin_hour_CTR',
         'item_pv_level_bin_hour_CTR',
         'shop_review_num_level_bin_hour_CTR',
         'shop_review_positive_rate_bin_hour_CTR',
         'shop_star_level_bin_hour_CTR',
         'shop_score_service_bin_hour_CTR',
         'shop_score_delivery_bin_hour_CTR',
         'shop_score_description_bin_hour_CTR',
    
        'hour_user_gender_id_click_rate',
         'hour_user_age_level_click_rate',
         'hour_user_occupation_id_click_rate',
         'hour_user_star_level_click_rate',
    
    
]

# In[10]:


if __name__ == '__main__':

#     data = all_data[((all_data.day == 7) | (all_data.day == 6)) & (all_data.is_trade != -1)]
#     data = data.reset_index()

    data = all_data[(all_data.day == 7) & (all_data.is_trade >= 0)]
    data.loc[data.hour <= 1, hour_features] = np.NAN
    data = data.reset_index()
    
    del all_data
    gc.collect()
    
    eval_data = data[(data.day == 7) & (data.hour >= 11)]
    eval_set = [(eval_data[features], eval_data[target])]

    lgb_clf = lgb.LGBMClassifier(objective='binary', device='gpu',  n_jobs=1, silent=False)

#  参数的组合
    lgb_param_grad = {'n_estimators': (4000, ),
                      'learning_rate': (0.03, ),

                      'max_depth': (5, ),
                      'num_leaves': (20, ),
                      'min_child_samples': (100, ),
                      'min_child_weight': (0.001, ),
                      'min_split_gain': (0.1, ),
                      
                      'colsample_bytree': (0.8,),
                      'subsample': (0.7, ),
                      'subsample_freq': (1,),
                      
                      'reg_lambda': (10, ),
                      
                      'max_bin': (63, ),
                      
                      'gpu_use_dp': (True, ),
                      }


    clf = GridSearchCV(lgb_clf, param_grid=lgb_param_grad, scoring='neg_log_loss',
                       cv=CustomCV(data), n_jobs=2, verbose=1, refit=False, return_train_score=True)

    clf.fit(data[features], data[target],
            feature_name=features,
            categorical_feature=categorical_feature,
            early_stopping_rounds=300, eval_set=eval_set, verbose=50
           )

    print('=====')
    print("Best parameters set found on development set:")
    print(clf.best_params_)

    print('=====')
    print("Best parameters set found on development set:")
    print(clf.best_score_)
    

    
    
    
 



