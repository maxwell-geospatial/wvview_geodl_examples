#Based on this example: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb
#Import Libraries =======================================
from typing import Optional, List
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

#Read training and testing chip lists =======================================
train = pd.read_csv("C:/Maxwell_Data/wvlcDL_2/chips2/trainSet.csv")
val = pd.read_csv("C:/Maxwell_Data/wvlcDL_2/chips2/valSet.csv")

# Define Variables ========================================
MULTICLASS_MODE: str = "multiclass"
ENCODER = "resnet34"
ENCODER_WEIGHTS = None
CLASSES = ['background', 'barren', 'crop', 'forest', 'grass', 'imperv', 'mixdev', 'water']
ACTIVATION = None
DEVICE = 'cuda'

#Check if GPU is available ===================================
avail = torch.cuda.is_available()
devCnt = torch.cuda.device_count()
devName = torch.cuda.get_device_name(0)
print("Available: " + str(avail) + ", Count: " + str(devCnt) + ", Name: " + str(devName))

# Subclass and define custom dataset ===========================
class MultiClassSegDataset(Dataset):
    
    def __init__(self, df, classes=None, transform=None,):
        self.df = df
        self.classes = classes
        self.transform = transform
    
    def __getitem__(self, idx):
        
        image_name = self.df.iloc[idx, 2]
        mask_name = self.df.iloc[idx, 3]
        source = rasterio.open(image_name)
        image = source.read()
        source.close()
        image = image.astype('uint8')
        image = image.transpose(1,2,0)
        sourceM = rasterio.open(mask_name)
        mask = sourceM.read()
        mask = mask.transpose(1,2,0)
        sourceM.close()
        if(self.transform is not None):
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)   
            image = image.float()/255
            mask = mask.long()
            image = image.permute(2,0,1)
            mask = mask.permute(2,0,1)
        else: 
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)
            image = image.float()/255
            mask = mask.long()
            image = image.permute(2,0,1)
            mask = mask.permute(2,0,1)
        return image, mask  
        
    def __len__(self):
        return len(self.df)

#Define tranforms using Albumations =======================================
val_transform = A.Compose(
    [A.PadIfNeeded(min_height=512, min_width=512, border_mode=4), A.Resize(512, 512),]
)

train_transform = A.Compose(
    [
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=4),
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.MedianBlur(blur_limit=3, always_apply=False, p=0.1),
    ]
)

# Create the datasets   ================================================ 
trainDS = MultiClassSegDataset(train, classes=CLASSES, transform=train_transform)
valDS = MultiClassSegDataset(val, classes=CLASSES, transform=val_transform)
print("Number of Training Samples: " + str(len(trainDS)) + " Number of Validation Samples: " + str(len(valDS)))

# Define DataLoaders ==============================================
trainDL = torch.utils.data.DataLoader(trainDS, batch_size=4, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
valDL =  torch.utils.data.DataLoader(valDS, batch_size=4, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)

# Check Tensor shapes ======================================================
batch = next(iter(trainDL))
images, labels = batch
print(images.shape, labels.shape, type(images), type(labels), images.dtype, labels.dtype)

# Check first sample from patch =================================================
testImg = images[1]
testMsk = labels[1]
print(testImg.shape, testImg.dtype, type(testImg), testMsk.shape, 
testMsk.dtype, type(testMsk), testImg.min(), 
testImg.max(), testMsk.min(), testMsk.max())

# Plot example image =====================================
plt.imshow(testImg.permute(1,2,0))
plt.show()

# Plot exmaple mask ======================================
plt.imshow(testMsk.permute(1,2,0))
plt.show()

# Initiate UNet++ Model ======================================
model = smp.UnetPlusPlus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    decoder_use_batchnorm=True,
    in_channels=4,
    classes=8,
    activation=ACTIVATION,
)

#Define Loss and Metrics to Monitor (Make sure mode = "multiclass") ======================================
loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True, ignore_index=0)
loss.__name__ = 'Dice_loss'
#Will not monitor any metircs other than loss. 
metrics=[]

# Define Optimizer (Adam in this case) and learning rate ============================
optimizer = optim.Adam(params=model.parameters(), lr=0.0001)

# Define training epock =====================================
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss,
    metrics= metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

# Define testing epoch =====================================
valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

# train model for 10 epochs
for i in range(0, 10):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(trainDL)
    valid_logs = valid_epoch.run(valDL)
    torch.save(model, './best_model.pth')

