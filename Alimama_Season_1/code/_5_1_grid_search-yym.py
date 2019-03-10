
# coding: utf-8

# In[41]:


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


# In[42]:


all_data_path = feature_data_path + 'all_data_all_features.pkl'
all_data = load_pickle(all_data_path)

target = 'is_trade'


# In[43]:


features = load_pickle(feature_data_path + 'features_0413.pkl')

nominal_feats = ['hour',
                 'item_sales_level', 'item_price_level', 
                 'user_star_level', 'user_age_level', 'user_gender_id', 'user_occupation_id', 
                 'context_page_id', 
                 'category2_label', 'category_predict_rank',
                 'user_item_city_id_click_rank',
                 'user_item_id_click_rank',
                 'user_item_brand_id_click_rank',
                 'user_item_city_id_click_rank',
                 'user_shop_id_click_rank',
                 'user_context_page_id_click_rank',
                 'user_category2_label_click_rank',
                 'user_item_sales_level_click_rank',
                 'user_item_price_level_click_rank',
                ]

# features = list(set(features + nominal_feats))

len(features)


# In[44]:


if __name__ =='__main__':

    train_data = all_data[(all_data.day >= 19) & (all_data.day <= 23)]
    test_data = all_data[all_data.day == 24]

    lgb_train = lgb.Dataset(train_data[features], train_data['is_trade'])
    lgb_eval = lgb.Dataset(test_data[features], test_data['is_trade'], reference=lgb_train)

    train_data = all_data[(all_data.day >= 19) & (all_data.day <= 24)]
    train_data = train_data.reset_index()
    train_data_index = train_data[(train_data.day >= 19) & (train_data.day <= 23)].index
    val_data_index = train_data[train_data.day == 24].index

    lgb_clf = lgb.LGBMClassifier(objective='binary', n_jobs=-1, silent=False)

    # 参数的组合
    lgb_param_grad = {'n_estimators': (1000,), 
                      'learning_rate': (0.02,), 

                      'max_depth': (6, ), 
                      'num_leaves': (35, ), 
                      'min_child_samples': (200,),
    #                   'min_child_weight': (1e-3, 0.1),

                      'colsample_bytree': (0.9, 1.0),
                      'subsample': (0.7, 0.9),
                      'subsample_freq': (1, 5),

                      'reg_lambda': (1, 5)
                     }

    clf = GridSearchCV(lgb_clf, param_grid=lgb_param_grad, scoring='neg_log_loss',
                       cv=((train_data_index, val_data_index),), n_jobs=-1, verbose=1, refit=False)


    clf.fit(train_data[features], train_data[target], feature_name=features, early_stopping_rounds=20, 
            valid_sets=[lgb_eval, lgb_train], valid_names = ['eval', 'train'], verbose=50)


    print('=====')
    print("Best parameters set found on development set:")
    print(clf.best_params_)

    print('=====')
    print("Best parameters set found on development set:")
    print(clf.best_score_)

    pd.DataFrame(data=clf.cv_results_)


# In[ ]:




