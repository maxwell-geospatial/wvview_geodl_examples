#Based on this example: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb
#Import Libraries =======================================
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import os
import math
import cv2
import torch
from torch.utils.data.dataset import Dataset
import segmentation_models_pytorch as smp
import rasterio

# Check GPU availability============================================
avail = torch.cuda.is_available()
devCnt = torch.cuda.device_count()
devName = torch.cuda.get_device_name(0)
print("Available: " + str(avail) + ", Count: " + str(devCnt) + ", Name: " + str(devName))

# Initiate UNet Model ===========================================
ENCODER = "resnet18"
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ["mine"]
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'
model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)
loss = smp.utils.losses.DiceLoss() + smp.utils.losses.BCELoss()
metrics = [
    smp.utils.metrics.Fscore(beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None),
    smp.utils.metrics.Accuracy(threshold=0.5, activation=None, ignore_channels=None),
    smp.utils.metrics.Recall(eps=1e-7, threshold=0.5, activation=None, ignore_channels=None),
    smp.utils.metrics.Precision(eps=1e-7, threshold=0.5, activation=None, ignore_channels=None),
]
optimizer = torch.optim.Adam([ 
    dict(params=model.parameters()),
])

#Load saved model=====================================================
best_model = torch.load('C:/Maxwell_Data/topo_work/scripts/best_model.pth')

#Read in topo map and convert to tensor===========================
image = cv2.imread("C:/Maxwell_Data/topo_work/data/img/KY_Offutt_709431_1954_24000_geo.tif")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.astype('uint8')
image = torch.from_numpy(image)
image = image.permute(2, 0, 1)
image = image.float()/255
t_arr = image

#Make blank grid to write predictions two with same height and width as topo===========================
image2 = cv2.imread("C:/Maxwell_Data/topo_work/data/img/KY_Offutt_709431_1954_24000_geo.tif")
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
image2 = image2.astype('uint8')
image2 = torch.from_numpy(image2)
image2 = image2.permute(2, 0, 1)
image2 = image2.float()/255
p_arr = image2[0, :, :]
p_arr[:,:] = 0

#Predict to entire topo using overlapping chips, merge back to original shap=============
size= 256
stride_x = 128
stride_y = 128
crop = 50
n_channels=3

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
        ten1 = t_arr[0:3, r1b:r2b, c1b:c2b]
        ten1 = ten1.to(DEVICE).unsqueeze(0)
        ten_p = best_model.predict(ten1)
        ten_p = ten_p.squeeze(0)
        ten_p = ten_p.squeeze(0)
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
topo = rasterio.open("C:/Maxwell_Data/topo_work/data/img/KY_Offutt_709431_1954_24000_geo.tif")
profile1 = topo.profile.copy()
t_arr2 = topo.read()
topo.close()
profile1["driver"] = "GTiff"
profile1["dtype"] = "uint8"
profile1["count"] = 1

pr_out = p_arr.cpu().numpy().round().astype('uint8')

#Write out result========================================================
with rasterio.open("C:/Maxwell_Data/topo_work/test12.tif", "w", **profile1) as f:
    f.write(pr_out,1)
