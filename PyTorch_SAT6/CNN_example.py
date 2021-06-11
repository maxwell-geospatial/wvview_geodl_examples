#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import pandas as pd
from skimage import io, transform, exposure, img_as_uint, img_as_float
import imageio as io
import torch


# In[3]:


def annoPrep(anno_in, out_anno):
    anno_in = anno_in
    out_anno = out_anno
    labels = pd.read_csv(anno_in, header=None)
    labels.columns = ["building","barren_land","trees","grassland","road","water"]
    labels['class'] = labels.idxmax(axis=1)
    labels['class'].to_csv(out_anno)


# In[5]:


annoPrep("C:/Maxwell_Data/archive/y_test_sat6.csv", "C:/Maxwell_Data/archive/chips/y_test.csv")
annoPrep("C:/Maxwell_Data/archive/y_train_sat6.csv", "C:/Maxwell_Data/archive/chips/y_train.csv")


# In[6]:


def csv2img(csvData, outDir):
    data = pd.read_csv(csvData,header=None)
    data_rows, data_cols = data.shape
    out_dir = outDir

    for i in range(data_rows):

        row_i = data.iloc[i, 0:].to_numpy()
        im = np.reshape(row_i, (28,28,4))

        image_name = str(i)+'.png'
        output_path = os.path.join(out_dir,image_name)
        io.imwrite(output_path, im)


# In[7]:


parentPath = "C:/Maxwell_Data/archive/chips/"
pathTrainX = os.path.join(parentPath, "train_x") 
pathTestX = os.path.join(parentPath, "test_x") 
os.mkdir(pathTrainX)
os.mkdir(pathTestX)


# In[ ]:


csv2img("C:/Maxwell_Data/archive/X_train_sat6.csv", pathTrainX)
csv2img("C:/Maxwell_Data/archive/X_test_sat6.csv", pathTestX)


# In[4]:


labelsTrain = pd.read_csv("D:/archive/sat6b/y_train.csv")
labelsTest = pd.read_csv("D:/archive/sat6b/y_test.csv")


# In[5]:


labelsTest.head(), labelsTrain.head()


# In[6]:


labelsTrain.columns = ["name", "label"]
labelsTrain["name"] = "train_x/" + labelsTrain["name"].astype(str) + ".png"
labelsTrain["is_valid"] = False
labelsTrain


# In[7]:


labelsTest.columns = ["name", "label"]
labelsTest["name"] = "test_x/" + labelsTest["name"].astype(str) + ".png"
labelsTest["is_valid"] = True
labelsTest


# In[8]:


df = pd.concat([labelsTrain, labelsTest])
len(df), df.shape, df.head(), df.tail()


# In[23]:


parentPath = "D:/archive/sat6b/"
dls = ImageDataLoaders.from_df(df, parentPath, label_col=1, valid_col=2, bs=35, num_workers=0)


# In[25]:


matplotlib.rcParams['pdf.fonttype'] = 42
dls.show_batch(max_n=35)
plt.savefig('C:/users/amaxwel6/Desktop/set7.pdf')  


# In[12]:


get_ipython().run_line_magic('pinfo2', 'ImageList')


# In[23]:


learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(2)


# In[24]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:




