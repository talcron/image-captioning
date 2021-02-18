#!/usr/bin/env python
# coding: utf-8

# In[8]:


import csv
import random
from shutil import copyfile
from pycocotools.coco import COCO
from tqdm import tqdm


# In[9]:


#make directory and get annotations for training and testing
get_ipython().system(u'mkdir data')
get_ipython().system(u'wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip -P ./data/')
get_ipython().system(u'unzip ./data/captions_train-val2014.zip -d ./data/')
get_ipython().system(u'rm ./data/captions_train-val2014.zip')


# In[10]:


get_ipython().system(u'mkdir data/images')
get_ipython().system(u'mkdir data/images/train')
get_ipython().system(u'mkdir data/images/val')
get_ipython().system(u'mkdir data/images/test')


# In[11]:


coco = COCO('./data/annotations/captions_train2014.json')


# In[12]:


#get ids of training images
with open('train_ids.csv', 'r') as f:
    reader = csv.reader(f)
    trainIds = list(reader)
    
trainIds = [int(i) for i in trainIds[0]]

with open('val_ids.csv', 'r') as f:
    reader = csv.reader(f)
    valIds = list(reader)
    
valIds = [int(i) for i in valIds[0]]


# In[13]:


for img_id in trainIds:
    path = coco.loadImgs(img_id)[0]['file_name']
    copyfile('/datasets/COCO-2015/train2014/'+path, './data/images/train/'+path)
for img_id in valIds:
    path = coco.loadImgs(img_id)[0]['file_name']
    copyfile('/datasets/COCO-2015/train2014/'+path, './data/images/val/'+path)


# In[14]:


cocoTest = COCO('./data/annotations/captions_val2014.json')


# In[15]:


with open('TestImageIds.csv', 'r') as f:
    reader = csv.reader(f)
    testIds = list(reader)
    
testIds = [int(i) for i in testIds[0]]


# In[16]:


for img_id in testIds:
    path = cocoTest.loadImgs(img_id)[0]['file_name']
    copyfile('/datasets/COCO-2015/val2014/'+path, './data/images/test/'+path)


# In[17]:


print("done")


# In[ ]:




