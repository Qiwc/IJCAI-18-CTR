
# coding: utf-8

# In[2]:


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


# In[3]:


features=[
         'item_id',
         'item_brand_id',
         'item_city_id',
         'item_price_level',
         'item_sales_level',
         'item_collected_level',
         'item_pv_level',
#          'user_id',
         'user_gender_id',
         'user_age_level',
         'user_occupation_id',
         'user_star_level',
#          'context_id',
         'context_timestamp',
         'context_page_id',
         'shop_id',
         'shop_review_num_level',
         'shop_review_positive_rate',
         'shop_star_level',
         'shop_score_service',
         'shop_score_delivery',
         'shop_score_description',
#          'day',
#          'hour',
#          'minute',
         'category2_label',
         'category3_label',
#2-9
#          'user_gender_id_item_price_level',
#          'user_gender_id_item_sales_level',
#          'user_gender_id_shop_star_level',
#          'user_gender_id_shop_review_num_level',
#          'user_gender_id_shop_review_positive_rate',
         'user_gender_id_category2_label',
         'user_gender_id_category3_label',
#          'user_gender_id_context_page_id',
#          'user_gender_id_hour',
         'user_age_level_item_price_level',
#          'user_age_level_item_sales_level',
#          'user_age_level_shop_star_level',
#          'user_age_level_shop_review_num_level',
#          'user_age_level_shop_review_positive_rate',
         'user_age_level_category2_label',
         'user_age_level_category3_label',
#          'user_age_level_context_page_id',
#          'user_age_level_hour',
         'user_occupation_id_item_price_level',
#          'user_occupation_id_item_sales_level',
#          'user_occupation_id_shop_star_level',
#          'user_occupation_id_shop_review_num_level',
#          'user_occupation_id_shop_review_positive_rate',
         'user_occupation_id_category2_label',
         'user_occupation_id_category3_label',
#          'user_occupation_id_context_page_id',
#          'user_occupation_id_hour',
#          'user_star_level_item_price_level',
#          'user_star_level_item_sales_level',
#          'user_star_level_shop_star_level',
#          'user_star_level_shop_review_num_level',
#          'user_star_level_shop_review_positive_rate',
#          'user_star_level_category2_label',
#          'user_star_level_category3_label',
#          'user_star_level_context_page_id',
#          'user_star_level_hour',
#          'user_gender_id_user_age_level',
#          'user_gender_id_user_occupation_id',
#          'user_gender_id_user_star_level',
#          'user_age_level_user_occupation_id',
#          'user_age_level_user_star_level',
#          'user_occupation_id_user_star_level',
    
    
# category
#          'item_price_level_bin',
#          'item_sales_level_bin',
#          'item_collected_level_bin',
#          'item_pv_level_bin',
#          'shop_review_num_level_bin',
#          'shop_review_positive_rate_bin',
#          'shop_star_level_bin',
#          'shop_score_service_bin',
#          'shop_score_delivery_bin',
#          'shop_score_description_bin',
#          'hour_bin',

         'item_property_topic_k_15',
    
#2-1
        'user_category2_label_click_day',
        'user_category3_label_click_day',
        'user_shop_id_click_day',
        'user_item_id_click_day',
        'user_item_brand_id_click_day',
        'user_context_page_id_click_day',
        'user_item_price_level_bin_click_day',
        'user_item_sales_level_bin_click_day',
        'user_item_property_topic_k_15_click_day',
    
#         'user_category2_label_click_hour',
#         'user_category3_label_click_hour',
#         'user_shop_id_click_hour',
#         'user_item_id_click_hour',
#         'user_item_brand_id_click_hour',
#         'user_context_page_id_click_hour',
#         'user_item_price_level_bin_click_hour',
#         'user_item_sales_level_bin_click_hour',
#         'user_item_property_topic_k_15_click_hour',
    
    
#2-2    
#          'user_id_click_day_mean',
#          'category2_label_click_day_mean',
#          'category3_label_click_day_mean',
#          'shop_id_click_day_mean',
#          'item_id_click_day_mean',
#          'item_brand_id_click_day_mean',
#          'context_page_id_click_day_mean',
    
    
#2-3
        
        'user_click_rank_day',
        'user_click_true_rank_day',
        'user_click_interval_first_day',
        'user_click_interval_last_day',
        'user_click_time_gap_before',
        'user_click_time_gap_after',
    
#         'user_click_rank_global',
#         'user_click_true_rank_global',
#         'user_click_interval_first_global',
#         'user_click_interval_last_global',
#         'user_click_time_gap_before_global',
#         'user_click_time_gap_after_global',
    
        'user_category2_label_click_rank_day',
        'user_category2_label_click_true_rank_day',
        'user_category2_label_first_click_interval_day',
        'user_category2_label_last_click_interval_day',
        'user_category2_label_click_time_gap_before_day',
        'user_category2_label_click_time_gap_after_day',
        'user_category3_label_click_rank_day',
        'user_category3_label_click_true_rank_day',
        'user_category3_label_first_click_interval_day',
        'user_category3_label_last_click_interval_day',
        'user_category3_label_click_time_gap_before_day',
        'user_category3_label_click_time_gap_after_day',
        'user_shop_id_click_rank_day',
        'user_shop_id_click_true_rank_day',
        'user_shop_id_first_click_interval_day',
        'user_shop_id_last_click_interval_day',
        'user_shop_id_click_time_gap_before_day',
        'user_shop_id_click_time_gap_after_day',
        'user_item_id_click_rank_day',
        'user_item_id_click_true_rank_day',
        'user_item_id_first_click_interval_day',
        'user_item_id_last_click_interval_day',
        'user_item_id_click_time_gap_before_day',
        'user_item_id_click_time_gap_after_day',
        'user_item_brand_id_click_rank_day',
        'user_item_brand_id_click_true_rank_day',
        'user_item_brand_id_first_click_interval_day',
        'user_item_brand_id_last_click_interval_day',
        'user_item_brand_id_click_time_gap_before_day',
        'user_item_brand_id_click_time_gap_after_day',
        'user_item_property_topic_k_15_click_rank_day',
        'user_item_property_topic_k_15_click_true_rank_day',
        'user_item_property_topic_k_15_first_click_interval_day',
        'user_item_property_topic_k_15_last_click_interval_day',
        'user_item_property_topic_k_15_click_time_gap_before_day',
        'user_item_property_topic_k_15_click_time_gap_after_day',
    
#         'user_category2_label_click_rank_global',
#         'user_category2_label_click_true_rank_global',
#         'user_category2_label_first_click_interval_global',
#         'user_category2_label_last_click_interval_global',
#         'user_category2_label_click_time_gap_before_global',
#         'user_category2_label_click_time_gap_after_global',
#         'user_category3_label_click_rank_global',
#         'user_category3_label_click_true_rank_global',
#         'user_category3_label_first_click_interval_global',
#         'user_category3_label_last_click_interval_global',
#         'user_category3_label_click_time_gap_before_global',
#         'user_category3_label_click_time_gap_after_global',
#         'user_shop_id_click_rank_global',
#         'user_shop_id_click_true_rank_global',
#         'user_shop_id_first_click_interval_global',
#         'user_shop_id_last_click_interval_global',
#         'user_shop_id_click_time_gap_before_global',
#         'user_shop_id_click_time_gap_after_global',
#         'user_item_id_click_rank_global',
#         'user_item_id_click_true_rank_global',
#         'user_item_id_first_click_interval_global',
#         'user_item_id_last_click_interval_global',
#         'user_item_id_click_time_gap_before_global',
#         'user_item_id_click_time_gap_after_global',
#         'user_item_brand_id_click_rank_global',
#         'user_item_brand_id_click_true_rank_global',
#         'user_item_brand_id_first_click_interval_global',
#         'user_item_brand_id_last_click_interval_global',
#         'user_item_brand_id_click_time_gap_before_global',
#         'user_item_brand_id_click_time_gap_after_global',
#         'user_item_property_topic_k_15_click_rank_global',
#         'user_item_property_topic_k_15_click_true_rank_global',
#         'user_item_property_topic_k_15_first_click_interval_global',
#         'user_item_property_topic_k_15_last_click_interval_global',
#         'user_item_property_topic_k_15_click_time_gap_before_global',
#         'user_item_property_topic_k_15_click_time_gap_after_global',
    
#2-4
        'item_property_topic_0',
        'item_property_topic_1',
        'item_property_topic_2',
        'item_property_topic_3',
        'item_property_topic_4',
        'item_property_topic_5',
        'item_property_topic_6',
        'item_property_topic_7',
        'item_property_topic_8',
        'item_property_topic_9',
        'item_property_topic_10',
        'item_property_topic_11',
        'item_property_topic_12',
        'item_property_topic_13',
        'item_property_topic_14',
        'property_sim',
        'category_predict_rank',

    
    
#2-5
#     交叉特征历史ctr

#         'user_gender_id_item_price_level_smooth_CTR',
#         'user_gender_id_item_sales_level_smooth_CTR',
#         'user_gender_id_shop_star_level_smooth_CTR',
#         'user_gender_id_shop_review_num_level_smooth_CTR',
#         'user_gender_id_shop_review_positive_rate_smooth_CTR',
        'user_gender_id_category2_label_smooth_CTR',
        'user_gender_id_category3_label_smooth_CTR',
        'user_age_level_item_price_level_smooth_CTR',
#         'user_age_level_item_sales_level_smooth_CTR',
#         'user_age_level_shop_star_level_smooth_CTR',
#         'user_age_level_shop_review_num_level_smooth_CTR',
#         'user_age_level_shop_review_positive_rate_smooth_CTR',
        'user_age_level_category2_label_smooth_CTR',
        'user_age_level_category3_label_smooth_CTR',
        'user_occupation_id_item_price_level_smooth_CTR',
#         'user_occupation_id_item_sales_level_smooth_CTR',
#         'user_occupation_id_shop_star_level_smooth_CTR',
#         'user_occupation_id_shop_review_num_level_smooth_CTR',
#         'user_occupation_id_shop_review_positive_rate_smooth_CTR',
        'user_occupation_id_category2_label_smooth_CTR',
        'user_occupation_id_category3_label_smooth_CTR',
#         'user_star_level_item_price_level_smooth_CTR',
#         'user_star_level_item_sales_level_smooth_CTR',
#         'user_star_level_shop_star_level_smooth_CTR',
#         'user_star_level_shop_review_num_level_smooth_CTR',
#         'user_star_level_shop_review_positive_rate_smooth_CTR',
#         'user_star_level_category2_label_smooth_CTR',
#         'user_star_level_category3_label_smooth_CTR',
#         'user_gender_id_user_age_level_smooth_CTR',
#         'user_gender_id_user_occupation_id_smooth_CTR',
#         'user_gender_id_user_star_level_smooth_CTR',
#         'user_age_level_user_occupation_id_smooth_CTR',
#         'user_age_level_user_star_level_smooth_CTR',
#         'user_occupation_id_user_star_level_smooth_CTR',
    
    
#     交叉特征当前小时历史ctr
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
    
    
#     单特征历史点击与ctr    
         'user_id_smooth_I',
         'user_id_smooth_C',
#          'user_id_smooth_CTR',
         'item_id_smooth_I',
         'item_id_smooth_C',
#          'item_id_smooth_CTR',
#          'item_brand_id_smooth_I',
#          'item_brand_id_smooth_C',
         'item_brand_id_smooth_CTR',
#          'category2_label_smooth_I',
#          'category2_label_smooth_C',
         'category2_label_smooth_CTR',
#          'category3_label_smooth_I',
#          'category3_label_smooth_C',
         'category3_label_smooth_CTR',
#          'context_page_id_smooth_I',
#          'context_page_id_smooth_C',
#          'context_page_id_smooth_CTR',
#          'shop_id_smooth_I',
#          'shop_id_smooth_C',
         'shop_id_smooth_CTR',
#          'item_sales_level_bin_smooth_I',
#          'item_sales_level_bin_smooth_C',
         'item_sales_level_bin_smooth_CTR',
#          'item_price_level_bin_smooth_I',
#          'item_price_level_bin_smooth_C',
         'item_price_level_bin_smooth_CTR',
#          'item_collected_level_bin_smooth_I',
#          'item_collected_level_bin_smooth_C',
         'item_collected_level_bin_smooth_CTR',
#          'item_pv_level_bin_smooth_I',
#          'item_pv_level_bin_smooth_C',
         'item_pv_level_bin_smooth_CTR',
#          'shop_review_num_level_bin_smooth_I',
#          'shop_review_num_level_bin_smooth_C',
         'shop_review_num_level_bin_smooth_CTR',
#          'shop_review_positive_rate_bin_smooth_I',
#          'shop_review_positive_rate_bin_smooth_C',
         'shop_review_positive_rate_bin_smooth_CTR',
#          'shop_star_level_bin_smooth_I',
#          'shop_star_level_bin_smooth_C',
         'shop_star_level_bin_smooth_CTR',
#          'shop_score_service_bin_smooth_I',
#          'shop_score_service_bin_smooth_C',
         'shop_score_service_bin_smooth_CTR',
#          'shop_score_delivery_bin_smooth_I',
#          'shop_score_delivery_bin_smooth_C',
         'shop_score_delivery_bin_smooth_CTR',
#          'shop_score_description_bin_smooth_I',
#          'shop_score_description_bin_smooth_C',
         'shop_score_description_bin_smooth_CTR',
#          'hour_smooth_I',
#          'hour_smooth_C',
         'hour_smooth_CTR',
    
    
         'user_id_hour_CTR',
         'item_id_hour_CTR',
         'item_brand_id_hour_CTR',
         'category2_label_hour_CTR',
         'category3_label_hour_CTR',
#          'context_page_id_hour_CTR',
         'shop_id_hour_CTR',
#          'item_sales_level_bin_hour_CTR',
#          'item_price_level_bin_hour_CTR',
#          'item_collected_level_bin_hour_CTR',
#          'item_pv_level_bin_hour_CTR',
#          'shop_review_num_level_bin_hour_CTR',
#          'shop_review_positive_rate_bin_hour_CTR',
#          'shop_star_level_bin_hour_CTR',
#          'shop_score_service_bin_hour_CTR',
#          'shop_score_delivery_bin_hour_CTR',
#          'shop_score_description_bin_hour_CTR',
    
    
#     用户交叉特征历史ctr
        'user_id_item_id_history_I',
        'user_id_item_id_history_C',
#         'user_id_item_id_history_CTR',
        'user_id_item_brand_id_history_I',
        'user_id_item_brand_id_history_C',
#         'user_id_item_brand_id_history_CTR',
        'user_id_category2_label_history_I',
        'user_id_category2_label_history_C',
#         'user_id_category2_label_history_CTR',
        'user_id_category3_label_history_I',
        'user_id_category3_label_history_C',
#         'user_id_category3_label_history_CTR',
        'user_id_shop_id_history_I',
        'user_id_shop_id_history_C',
#         'user_id_shop_id_history_CTR',
#         'user_id_item_sales_level_bin_history_I',
#         'user_id_item_sales_level_bin_history_C',
#         'user_id_item_sales_level_bin_history_CTR',
#         'user_id_item_price_level_bin_history_I',
#         'user_id_item_price_level_bin_history_C',
#         'user_id_item_price_level_bin_history_CTR',

    
#     各种id属性与用户属性交叉历史ctr
        'user_gender_id_shop_id_smooth_CTR',
        'user_gender_id_item_id_smooth_CTR',
        'user_gender_id_item_brand_id_smooth_CTR',
        'user_age_level_shop_id_smooth_CTR',
        'user_age_level_item_id_smooth_CTR',
        'user_age_level_item_brand_id_smooth_CTR',
        'user_occupation_id_shop_id_smooth_CTR',
        'user_occupation_id_item_id_smooth_CTR',
        'user_occupation_id_item_brand_id_smooth_CTR',
#         'user_star_level_shop_id_smooth_CTR',
#         'user_star_level_item_id_smooth_CTR',
#         'user_star_level_item_brand_id_smooth_CTR',
    
    
    
#2-6
         'shop_id_user_gender_id_click_rate',
         'shop_id_user_age_level_click_rate',
         'shop_id_user_occupation_id_click_rate',
#          'shop_id_user_star_level_click_rate',
         'item_id_user_gender_id_click_rate',
         'item_id_user_age_level_click_rate',
         'item_id_user_occupation_id_click_rate',
#          'item_id_user_star_level_click_rate',
         'item_brand_id_user_gender_id_click_rate',
         'item_brand_id_user_age_level_click_rate',
         'item_brand_id_user_occupation_id_click_rate',
#          'item_brand_id_user_star_level_click_rate',
         'category2_label_user_gender_id_click_rate',
         'category2_label_user_age_level_click_rate',
         'category2_label_user_occupation_id_click_rate',
#          'category2_label_user_star_level_click_rate',
         'category3_label_user_gender_id_click_rate',
         'category3_label_user_age_level_click_rate',
         'category3_label_user_occupation_id_click_rate',
#          'category3_label_user_star_level_click_rate',
         'hour_user_gender_id_click_rate',
         'hour_user_age_level_click_rate',
         'hour_user_occupation_id_click_rate',
#          'hour_user_star_level_click_rate',
#          'item_sales_level_bin_user_gender_id_click_rate',
#          'item_sales_level_bin_user_age_level_click_rate',
#          'item_sales_level_bin_user_occupation_id_click_rate',
#          'item_sales_level_bin_user_star_level_click_rate',
         'item_price_level_bin_user_gender_id_click_rate',
         'item_price_level_bin_user_age_level_click_rate',
         'item_price_level_bin_user_occupation_id_click_rate',
#          'item_price_level_bin_user_star_level_click_rate',
    
    
#2-7
#          'user_id_click_day',
#          'item_id_click_day',
#          'item_brand_id_click_day',
#          'category2_label_click_day',
#          'category3_label_click_day',
#          'context_page_id_click_day',
#          'shop_id_click_day',
#          'item_property_topic_k_15_click_day',
    
#          'user_id_click_hour',
#          'item_id_click_hour',
#          'item_brand_id_click_hour',
#          'category2_label_click_hour',
#          'category3_label_click_hour',
#          'context_page_id_click_hour',
#          'shop_id_click_hour',
#          'item_property_topic_k_15_click_hour',
    
    
#2-8
#          'user_item_id_pre_click',
         'user_item_id_continue_click',
#          'user_item_brand_id_pre_click',
         'user_item_brand_id_continue_click',
#          'user_shop_id_pre_click',
         'user_shop_id_continue_click',
#          'user_category2_label_pre_click',
         'user_category2_label_continue_click',
#          'user_category3_label_pre_click',
         'user_category3_label_continue_click',
#          'user_item_property_topic_k_15_pre_click',
         'user_item_property_topic_k_15_continue_click',
#          'user_item_sales_level_bin_pre_click',
#          'user_item_sales_level_bin_continue_click',
#          'user_item_price_level_bin_pre_click',
#          'user_item_price_level_bin_continue_click',
    
    
#2-10  
         'user_category2_label_future_2min',
#          'user_category2_label_future_15min',
         'user_category3_label_future_2min',
#          'user_category3_label_future_15min',
         'user_shop_id_future_2min',
#          'user_shop_id_future_15min',
         'user_item_id_future_2min',
#          'user_item_id_future_15min',
         'user_item_brand_id_future_2min',
#          'user_item_brand_id_future_15min',
         'user_item_property_topic_k_15_future_2min',
#          'user_item_property_topic_k_15_future_15min',
#          'user_item_sales_level_bin_future_2min',
#          'user_item_sales_level_bin_future_15min',
#          'user_item_price_level_bin_future_2min',
#          'user_item_price_level_bin_future_15min'
         ]
len(features)


# In[4]:


all_data_path = feature_data_path + 'all_data_all_features.pkl'
all_data = load_pickle(all_data_path)

target = 'is_trade'


# In[5]:


categorical_feature = [
             'item_price_level',
             'item_sales_level',
             'item_collected_level',
             'item_pv_level',
           
            'user_age_level',
            'user_gender_id',
#             'user_occupation_id',
            'user_star_level',
            'item_property_topic_k_15',
    
            'shop_review_num_level',
            'shop_star_level',
    
#             'category2_label',
#             'category3_label',

    
            'user_click_rank_day',
            'user_category2_label_click_rank_day',
            'user_category3_label_click_rank_day',
            'user_shop_id_click_rank_day',
            'user_item_id_click_rank_day',
            'user_item_brand_id_click_rank_day',
    
#          'user_item_id_pre_click',
         'user_item_id_continue_click',
#          'user_item_brand_id_pre_click',
         'user_item_brand_id_continue_click',
#          'user_shop_id_pre_click',
         'user_shop_id_continue_click',
#          'user_category2_label_pre_click',
         'user_category2_label_continue_click',
#          'user_category3_label_pre_click',
         'user_category3_label_continue_click',
#          'user_item_property_topic_k_15_pre_click',
         'user_item_property_topic_k_15_continue_click',
    
            'category_predict_rank',
    

]
len(categorical_feature)


# In[6]:


def CustomCV(data,):    
    fold_index_train = data[(data.hour < 11)].index
    fold_index_test = data[data.hour >= 11].index
    
    yield fold_index_train, fold_index_test


# In[ ]:


if __name__ == '__main__':


    data = all_data[(all_data.day == 7) & (all_data.is_trade != -1)& (all_data.is_trade != -2)]
    data = data.reset_index()
    
    eval_data = data[(data.day == 7) & (data.hour >= 11)]
    eval_set = [(eval_data[features], eval_data[target])]

    print(data.shape,eval_data.shape)
    
    lgb_clf = lgb.LGBMClassifier(objective='binary', device='gpu',  n_jobs=2, silent=False)


#  参数的组合
    lgb_param_grad = {'n_estimators': (4000, ),
                      'learning_rate': (0.03, ),

                      'max_depth': (4,),
                      'num_leaves': (12,),
                      #'min_child_samples': (500, 1000),
                      #'min_child_weight': (0.01, 0.1),
                      #'min_split_gain': (0.1, 0.02),
                      

                      'colsample_bytree': (0.9,),
                      'subsample': (0.8,),
                      'subsample_freq': (1,),
                      
                      'reg_lambda': (10,),
                      
                      'max_bin': (63, ),
                      
                      'gpu_use_dp': (False, ),
                      }

    clf = GridSearchCV(lgb_clf, param_grid=lgb_param_grad, scoring='neg_log_loss',
                       cv=CustomCV(data), n_jobs=4, verbose=1, refit=False, return_train_score=True)

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
    
    dump_pickle(pd.DataFrame(data=clf.cv_results_), '05010-gridsearch-qwc.pkl')


# In[ ]:


# pd.DataFrame(data=clf.cv_results_)[['rank_test_score', 'mean_test_score', 'mean_train_score', 
#                                      'param_num_leaves', 'param_subsample', 'param_colsample_bytree', 
#                                     'param_subsample_freq',
#                                     'param_max_bin', 'param_gpu_use_dp','param_reg_lambda',
#                                 #    'param_min_child_weight','param_min_child_samples','param_min_split_gain',
#                                    ]]



