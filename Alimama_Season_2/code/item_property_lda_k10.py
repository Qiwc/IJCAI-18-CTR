
# coding: utf-8

# In[2]:


import os
import pickle
import pandas as pd
import numpy as np
import time
from utils import raw_data_path, feature_data_path,result_path,model_path,cache_pkl_path,dump_pickle,load_pickle


# In[3]:


all_data = load_pickle(raw_data_path + 'all_data_4567.pkl')


# In[4]:


from sklearn.feature_extraction.text import CountVectorizer
import lda

def get_item_property_doc(item_category_list, item_property_list, k=10):
    item_category_list = [category for category in item_category_list.split(";") if category != '-1']
    item_property_list = [item_property for item_property in item_property_list.split(";") if item_property != '-1']
    item_category_list = item_category_list
    item_property_list = item_property_list[:k]
    doc = list()
    for item_category in item_category_list:
        doc.append('c_' + item_category)
    for item_property in item_property_list:
        doc.append(item_property)
    return ';'.join(doc)

    
def get_predict_property_doc(item_category_list, predict_category_property_list):

    doc = list()
#     只取前五个预测类别
    for predict_category_property in predict_category_property_list.split(";")[:5]:
        if predict_category_property != '-1':
            category = predict_category_property.split(":")[0]
            if category != -1:
                doc.append('c_' + category)
            property_list = predict_category_property.split(":")[1]
            if property_list != -1:
                for item_property in property_list.split(','):
                    doc.append(item_property)
    return ';'.join(doc)

def category_property_iter():
    for item_property_doc in all_data['item_property_doc']:
        yield item_property_doc
        
    for predict_category_property_doc in all_data['predict_property_doc']:
        yield predict_category_property_doc


# ## 重新生成一个k=10，不考虑property

# In[6]:


print('-----read all data-----')

def split(s):
    return s.split(';')

all_data['item_property_doc'] = all_data.apply(lambda row: get_item_property_doc(row['item_category_list'], row['item_property_list'], k=10), axis=1)

cv = CountVectorizer(analyzer=split)
count_vector_item_property = cv.fit_transform(all_data['item_property_doc'])

print('Shape of item_property Count Vector: {}'.format(count_vector_item_property.shape))


# In[7]:


k = 15

lda_model = lda.LDA(n_topics=k, n_iter=1000, random_state=1, refresh=10)
lda_model.fit(count_vector_item_property)

dump_pickle(lda_model, model_path + '4567_item_property_lda_model_k_15.pkl')



