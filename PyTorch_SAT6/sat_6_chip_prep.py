# Load libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, transform, exposure, img_as_uint, img_as_float
from skimage.io import imsave, imread
import imageio as io
import torch

# Define function to convert one-hot encoded labels to single column
def annoPrep(anno_in, out_anno):
    anno_in = anno_in
    out_anno = out_anno
    labels = pd.read_csv(anno_in, header=None)
    labels.columns = [0,1,2,3,4,5]
    labels['class'] = labels.idxmax(axis=1)
    labels['class'].to_csv(out_anno)

# Apply function to the training and testing labels
annoPrep("C:/Maxwell_Data/archive/y_test_sat6.csv", "C:/Maxwell_Data/archive/chips2/y_test.csv")
annoPrep("C:/Maxwell_Data/archive/y_train_sat6.csv", "C:/Maxwell_Data/archive/chips2/y_train.csv")
    
# Define function to convert CSV data to image chips
def csv2img(csvData, outDir):
    data = pd.read_csv(csvData,header=None)
    data_rows, data_cols = data.shape
    out_dir = outDir
    for i in range(data_rows):
        row_i = data.iloc[i, 0:].to_numpy()
        im = np.reshape(row_i, (28,28,4))
        image_name = str(i)+'.png'
        output_path = os.path.join(out_dir,image_name)
        imsave(output_path, im)
        
# Create directories
parentPath = "C:/Maxwell_Data/archive/chips2/"
pathTrainX = os.path.join(parentPath, "train_x") 
pathTestX = os.path.join(parentPath, "test_x") 
os.mkdir(pathTrainX)
os.mkdir(pathTestX)

# Convert CSV data to image chips
csv2img("C:/Maxwell_Data/archive/X_train_sat6.csv", pathTrainX)
csv2img("C:/Maxwell_Data/archive/X_test_sat6.csv", pathTestX)
