
# coding: utf-8

# In[1]:


import os
import pickle
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import load_pickle, dump_pickle, raw_data_path, feature_data_path, model_path
from utils import extract_ctr


# In[2]:


def get_category_2_predict_rank(item_category_list, predict_category_property):
    category_2 = item_category_list.split(";")[1]
    predict_category_list = [category_property.split(":")[0] for category_property in predict_category_property.split(";") if category_property != '-1']
    category_predict_rank = predict_category_list.index(category_2) if category_2 in predict_category_list else -1
    return category_predict_rank

def get_category_predict_rank(item_category_list, predict_category_property):
    item_category_list = [category for category in item_category_list.split(";") if category != '-1']
    predict_category_list = [category_property.split(":")[0] for category_property in predict_category_property.split(";")  if category_property != '-1']
    top_rank = 1000
    for item_category in item_category_list[1:]:
        if item_category in predict_category_list:
            rank = predict_category_list.index(item_category)
            if rank < top_rank:
                top_rank = rank
    return top_rank


def gen_category_predict_rank():
    '''生成实际类别在预测类别里的排序

    file_name: category_predict_rank.pkl

    features: 'category_predict_rank'

    '''

    all_data = load_pickle(raw_data_path + 'all_data_4567.pkl')

    feature_path = feature_data_path + 'category_predict_rank.pkl'
    
    print('generating ' + feature_path)
    all_data['category_predict_rank'] = all_data.apply(lambda row: get_category_predict_rank(row['item_category_list'], row['predict_category_property']), axis=1)
        
    all_data = all_data[['category_predict_rank', ]]
    dump_pickle(all_data, feature_path)


def add_category_predict_rank(data,):
    """实际类别在预测类别里的排序

    join_key: ['index',]

    """

    feature_path = feature_data_path + 'category_predict_rank.pkl'
    if not os.path.exists(feature_path):
        gen_category_predict_rank()
        
    category_predict_rank = load_pickle(feature_path)
    data = data.join(category_predict_rank)
    return data


# In[3]:


def get_property_sim(item_category_list, item_property_list, predict_category_property):

    item_category_list = item_category_list.split(";")
    predict_category_property_dict = {category_property.split(":")[0]: category_property.split(":")[1]
                                      for category_property in predict_category_property.split(";") if category_property != '-1'}
    predict_property_set = set()
    flag = 0
    for category in item_category_list[1:]:
        if (category != '-1') and (category in predict_category_property_dict):
            flag = 1
            if predict_category_property_dict[category] != '-1':
                p_list = predict_category_property_dict[category].split(",")
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

    all_data = load_pickle(raw_data_path + 'all_data_4567.pkl')

    feature_path = feature_data_path + 'property_sim.pkl'
    print('generating ' + feature_path)

    all_data['property_sim'] = all_data.apply(lambda row: get_property_sim(
        row['item_category_list'], row['item_property_list'], row['predict_category_property']), axis=1)
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


# In[4]:


from sklearn.feature_extraction.text import CountVectorizer
import lda

def add_item_property_lda(data,):
    """拼接item_property_list的主题分布向量

    join_key: ['index',]

    """

    lda_model_path = model_path + '4567_item_property_lda_model_k_15.pkl'

    lda_model = load_pickle(lda_model_path)
    
    k = 15
    topic_vector = lda_model.doc_topic_
    
    topic_vector = pd.DataFrame(topic_vector)
    for i in range(k):
        topic_vector.rename(columns={i: 'item_property_topic_' + str(i)}, inplace=True)
    print('Shape of Topic Distributions: {}'.format(topic_vector.shape))

    data = pd.concat([data, topic_vector], axis=1)
    
    return data


# In[5]:


if __name__ =='__main__':
    data = load_pickle(raw_data_path + 'all_data_4567.pkl')

    data = add_category_predict_rank(data)
    data = add_property_sim(data)
    data = add_item_property_lda(data)



