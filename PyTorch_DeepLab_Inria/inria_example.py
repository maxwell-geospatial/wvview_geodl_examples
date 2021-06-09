import sys

import numpy as np
import pandas as pd

import rasterio
from rasterio.windows import Window
from rasterio.errors import RasterioError, RasterioIOError

import torch 
from torchvision import transforms
from torch.utils.data.dataset import IterableDataset

from msai.dataloaders.StreamingDatasets import StreamingGeospatialDataset

import segmentation_models_pytorch as smp


trainLst = pd.read_csv("C:/Maxwell_Data/topo_data/topoDL/train_list.csv")
testLst = pd.read_csv("C:/Maxwell_Data/topo_data/topoDL/test_list.csv")
valLst = pd.read_csv("C:/Maxwell_Data/topo_data/topoDL/val_list.csv")

def image_transforms(img, group):
    img = np.rollaxis(img, 2, 0).astype(np.float32)
    img = torch.from_numpy(img)
    return img

def label_transforms(labels, group):
    return labels

def nodata_check(img, labels):
    return np.any(labels == 0) or np.any(np.sum(img == 0, axis=2) == 4)



train_data = StreamingGeospatialDataset(imagery_fns=trainLst.iloc[:,3], label_fns=trainLst.iloc[:,4], groups=None, chip_size=256, num_chips_per_tile=10, image_transform=image_transforms, label_transform=label_transforms, nodata_check=nodata_check)
test_data = StreamingGeospatialDataset(imagery_fns=testLst.iloc[:,3], label_fns=testLst.iloc[:,4], groups=None, chip_size=256, num_chips_per_tile=10, image_transform=image_transforms, label_transform=label_transforms, nodata_check=nodata_check)
val_data = StreamingGeospatialDataset(imagery_fns=valLst.iloc[:,3], label_fns=valLst.iloc[:,4], groups=None, chip_size=256, num_chips_per_tile=10, image_transform=image_transforms, label_transform=label_transforms, nodata_check=nodata_check)

trainDL = torch.utils.data.DataLoader(
        train_data,
        batch_size=10,
        num_workers=0,
        pin_memory=True,
    )
    
testDL = torch.utils.data.DataLoader(
        test_data,
        batch_size=10,
        num_workers=0,
        pin_memory=True,
    )

ENCODER="resnet18"
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ["not", "mine"]
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'
# In[2]:

model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

#%%

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.Fscore(beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None),
    smp.utils.metrics.Accuracy(threshold=0.5, activation=None, ignore_channels=None),
    smp.utils.metrics.Recall(eps=1e-7, threshold=0.5, activation=None, ignore_channels=None),
    smp.utils.metrics.Precision(eps=1e-7, threshold=0.5, activation=None, ignore_channels=None),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])

# In[54]:
# create epoch runners 
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

#%%

max_score = 0

for i in range(0, 10):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(trainDL)
    valid_logs = valid_epoch.run(testDL)
    
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['fscore']:
        max_score = valid_logs['fscore']
        torch.save(model, './best_model.pth')
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')


#%%

best_model = torch.load('./best_model.pth')
