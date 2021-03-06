#Based on this example: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb
#Import Libraries =======================================
from typing import Optional, List
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
import albumentations as A
import segmentation_models_pytorch as smp
import rasterio

# Change directory
os.chdir("C:/Maxwell_Data/wvlcDL_2/")

#Check if GPU is available ===================================
avail = torch.cuda.is_available()
devCnt = torch.cuda.device_count()
devName = torch.cuda.get_device_name(0)
print("Available: " + str(avail) + ", Count: " + str(devCnt) + ", Name: " + str(devName))


# Define Variables ========================================
MULTICLASS_MODE: str = "multiclass"
ENCODER = "resnet34"
ENCODER_WEIGHTS = None
CLASSES = ['background', 'barren', 'crop', 'forest', 'grass', 'imperv', 'mixdev', 'water']
ACTIVATION = None
DEVICE = 'cuda'

# Initiate UNet++ Model and send to GPU ======================================
model = smp.UnetPlusPlus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    decoder_use_batchnorm=True,
    in_channels=4,
    classes=8,
    activation=ACTIVATION,
).to(torch.device("cuda", 0))

# Read in trained model =====================================
model.load_state_dict(torch.load('model_out_11.pth'))
model.eval()

# Define inference function ================================
def geoInfer(image_in, pred_out, chip_size, stride_x, stride_y, crop, n_channels, n_classes):
    
    #Read in topo map and convert to tensor===========================
    image_name = image_in
    source1 = rasterio.open(image_name)
    image1 = source1.read()
    source1.close()
    image1 = image1.astype('uint8')
    image1 = torch.from_numpy(image1)
    iamge1 = image1.float()/255
    t_arr = iamge1
    
    #Make blank grid to write predictions two with same height and width as topo===========================
    source2 = rasterio.open(image_name)
    image2 = source2.read()
    source2.close()
    image2 = image2.astype('uint8')
    image2 = torch.from_numpy(image2)
    iamge2 = image2.float()/255
    t_arr = iamge2
    p_arr = image2[0, :, :]
    p_arr[:,:] = 0
    
    #Predict to entire topo using overlapping chips, merge back to original extent=============
    size= chip_size
    stride_x = stride_x
    stride_y = stride_y
    crop = crop
    n_channels=n_channels
    
    across_cnt = t_arr.shape[2]
    down_cnt = t_arr.shape[1]
    tile_size_across = size
    tile_size_down = size
    overlap_across = stride_x
    overlap_down = stride_y
    across = math.ceil(across_cnt/overlap_across)
    down = math.ceil(down_cnt/overlap_down)
    across_seq = list(range(0, across, 1))
    down_seq = list(range(0, down, 1))
    across_seq2 = [(x*overlap_across) for x in across_seq]
    down_seq2 = [(x*overlap_down) for x in down_seq]
    #Loop through row/column combinations to make predictions for entire image 
    for c in across_seq2:
        for r in down_seq2:
            c1 = c
            r1 = r
            c2 = c + size
            r2 = r + size
            #Default
            if c2 <= across_cnt and r2 <= down_cnt: 
                r1b = r1
                r2b = r2
                c1b = c1
                c2b = c2
            #Last column 
            elif c2 > across_cnt and r2 <= down_cnt: 
                r1b = r1
                r2b = r2
                c1b = across_cnt - size
                c2b = across_cnt + 1
            #Last row
            elif c2 <= across_cnt and r2 > down_cnt: 
                r1b = down_cnt - size
                r2b = down_cnt + 1
                c1b = c1
                c2b = c2
            #Last row, last column 
            else: 
                c1b = across_cnt - size
                c2b = across_cnt + 1
                r1b = down_cnt - size
                r2b = down_cnt + 1
            ten1 = t_arr[0:n_channels, r1b:r2b, c1b:c2b]
            ten1 = ten1.to(DEVICE).unsqueeze(0)
            ten2 = model.predict(ten1)
            m = nn.Softmax(dim=1)
            pr_probs = m(ten2)
            pr_probs2 = pr_probs[:, 1:n_classes+1, :, :]
            ten_p = torch.argmax(pr_probs2, dim=1).squeeze(1)
            ten_p = ten_p.squeeze()
            #print("executed for " + str(r1) + ", " + str(c1))
            if(r1b == 0 and c1b == 0): #Write first row, first column
                p_arr[r1b:r2b-crop, c1b:c2b-crop] = ten_p[0:size-crop, 0:size-crop]
            elif(r1b == 0 and c2b == across_cnt+1): #Write first row, last column
                p_arr[r1b:r2b-crop, c1b+crop:c2b] = ten_p[0:size-crop, 0+crop:size]
            elif(r2b == down_cnt+1 and c1b == 0): #Write last row, first column
                p_arr[r1b+crop:r2b, c1b:c2b-crop] = ten_p[crop:size+1, 0:size-crop]
            elif(r2b == down_cnt+1 and c2b == across_cnt+1): #Write last row, last column
                p_arr[r1b+crop:r2b, c1b+crop:c2b] = ten_p[crop:size, 0+crop:size+1]
            elif((r1b == 0 and c1b != 0) or (r1b == 0 and c2b != across_cnt+1)): #Write first row
                p_arr[r1b:r2b-crop, c1b+crop:c2b-crop] = ten_p[0:size-crop, 0+crop:size-crop]
            elif((r2b == down_cnt+1 and c1b != 0) or (r2b == down_cnt+1 and c2b != across_cnt+1)): # Write last row
                p_arr[r1b+crop:r2b, c1b+crop:c2b-crop] = ten_p[crop:size, 0+crop:size-crop]
            elif((c1b == 0 and r1b !=0) or (c1b ==0 and r2b != down_cnt+1)): #Write first column
                p_arr[r1b+crop:r2b-crop, c1b:c2b-crop] = ten_p[crop:size-crop, 0:size-crop]
            elif (c2b == across_cnt+1 and r1b != 0) or (c2b == across_cnt+1 and r2b != down_cnt+1): # write last column
                p_arr[r1b+crop:r2b-crop, c1b+crop:c2b] = ten_p[crop:size-crop, 0+crop:size]
            else: #Write middle chips
                p_arr[r1b+crop:r2b-crop, c1b+crop:c2b-crop] = ten_p[crop:size-crop, crop:size-crop]
            
    #Read in a GeoTIFF to get CRS info=======================================
    image3 = rasterio.open(image_in)
    profile1 = image3.profile.copy()
    image3.close()
    #interesting_keys = ('driver', 'dtype', 'nodata', 'width', 'height', 'count', 'crs')
    #profile2 = {x: profile1[x] for x in interesting_keys if x in profile1}
    profile1["driver"] = "GTiff"
    profile1["dtype"] = "uint8"
    profile1["count"] = 1
    
    pr_out = p_arr.unsqueeze(0).cpu().numpy().round().astype('uint8')
    
    #Write out result========================================================
    with rasterio.open(pred_out, "w", **profile1) as f:
        f.write(pr_out)
        
        
# Use inference function ============================================
geoInfer(image_in="C:/Maxwell_Data/wvlcDL_2/pred_test/test_qqquad5m.tif", pred_out="C:/Maxwell_Data/wvlcDL_2/pred_test/test_predOut.tif", chip_size=256, stride_x=64, stride_y=64, crop=70, n_channels=4, n_classes=8)
