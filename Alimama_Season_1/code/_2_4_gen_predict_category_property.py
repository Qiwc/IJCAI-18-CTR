
# coding: utf-8

# In[1]:


import os
import pickle
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import load_pickle, dump_pickle, raw_data_path, feature_data_path


# ## category在predict_category_property里面的排序

# In[2]:


def get_category_2_predict_rank(item_category_list, predict_category_property):
    category_2 = item_category_list.split(";")[1]
    predict_category_list = [category_property.split(":")[0] for category_property in predict_category_property.split(";")]
    category_predict_rank = predict_category_list.index(category_2) if category_2 in predict_category_list else -1
    return category_predict_rank

def get_category_3_predict_rank(item_category_list, predict_category_property):
    item_category_list = item_category_list.split(";")
    predict_category_list = [category_property.split(":")[0] for category_property in predict_category_property.split(";")]
    
    if len(item_category_list) < 3:
        return 0
    
    category_3 = item_category_list[2]
    if category_3 in predict_category_list:
        return 1.0
    else: 
        return 0.5

def get_category_predict_rank(item_category_list, predict_category_property):
    item_category_list = item_category_list.split(";")
    predict_category_list = [category_property.split(":")[0] for category_property in predict_category_property.split(";")]
    top_rank = 100
    for item_category in item_category_list:
        if item_category in predict_category_list:
            rank = predict_category_list.index(item_category)
            if rank < top_rank:
                top_rank = rank
    return top_rank


def gen_category_predict_rank(update=True):
    '''生成实际类别在预测类别里的排序

    file_name: category_predict_rank.pkl

    features: category_predict_rank

    '''

    all_data = load_pickle(raw_data_path + 'all_data.pkl')

    

    feature_path = feature_data_path + 'category_predict_rank.pkl'
    if os.path.exists(feature_path) and update == False:
        print('found ' + feature_path)
    else:
        print('generating ' + feature_path)
        all_data['category_predict_rank'] = all_data.apply(lambda row: get_category_predict_rank(
        row['item_category_list'], row['predict_category_property']), axis=1)
        
        all_data['category_3'] = all_data.apply(lambda row: get_category_3_predict_rank(row['item_category_list'], row['predict_category_property']), axis=1)
        
        all_data = all_data[['category_predict_rank', 'category_3']]
        dump_pickle(all_data, feature_path)


def add_category_predict_rank(data,):
    """添加分类属性日点击量的统计特征

    join_key: ['index',]

    """

    feature_path = feature_data_path + 'category_predict_rank.pkl'

    if not os.path.exists(feature_path):
        gen_category_predict_rank()
    category_predict_rank = load_pickle(feature_path)
    data = data.join(category_predict_rank)

    return data


# In[5]:


def get_property_sim(item_category_list, item_property_list, predict_category_property):
    
    item_category_list = item_category_list.split(";")
    predict_category_property_dict = {category_property.split(":")[0]:category_property.split(":")[1]
                             for category_property in predict_category_property.split(";") if category_property != '-1'}
    predict_property_set = set()
    flag = 0
    for category in item_category_list[1:]:
        if category in predict_category_property_dict:
            flag = 1
            p_list = predict_category_property_dict[category].split(";")
            predict_property_set.update(p_list)
    if flag == 1:
        item_property_set = set(item_property_list.split(";"))
        intersect = len(item_property_set.intersection(predict_property_set))
        if intersect > 0:
            sim = intersect
        else:
            sim = 0.5
    else:
        sim = 0.0
    return sim


def gen_property_sim(update=True):
    '''生成实际属性与预测属性的相似度

    file_name: property_sim.pkl

    features: property_sim

    '''

    all_data = load_pickle(raw_data_path + 'all_data.pkl')


    feature_path = feature_data_path + 'property_sim.pkl'
    if os.path.exists(feature_path) and update == False:
        print('found ' + feature_path)
    else:
        print('generating ' + feature_path)
        all_data['property_sim'] = all_data.apply(lambda row: get_property_sim(row['item_category_list'], row['item_property_list'], row['predict_category_property']), axis=1)
        
        all_data = all_data[['property_sim', ]]
        dump_pickle(all_data, feature_path)


def add_property_sim(data,):
    """添加实际属性与预测属性的相似度

    join_key: ['index',]

    """

    feature_path = feature_data_path + 'property_sim.pkl'

    if not os.path.exists(feature_path):
        gen_property_sim()
    property_sim = load_pickle(feature_path)
    data = data.join(property_sim)

    return data


# ## 测试

# In[9]:


if __name__ =='__main__':
    all_data = load_pickle(raw_data_path + 'all_data.pkl')
    gen_category_predict_rank()
    all_data = add_property_sim(all_data)
    all_data = add_category_predict_rank(all_data)
    print(all_data.columns)


# In[33]:


# In[ ]:




