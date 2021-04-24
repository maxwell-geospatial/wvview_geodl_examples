#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import os
import cv2
import torch
from torch.utils.data.dataset import Dataset
import albumentations as A
import segmentation_models_pytorch as smp

#%%
ENCODER="resnet18"
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ["mine"]
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'
# In[2]:

avail = torch.cuda.is_available()
devCnt = torch.cuda.device_count()
devName = torch.cuda.get_device_name(0)

print("Available: " + str(avail) + ", Count: " + str(devCnt) + ", Name: " + str(devName))

# In[3]:

set_train = pd.read_csv("C:/Maxwell_Data/topo_data/topo_dl_data/topo_dl_data/processing/train.csv")
set_val = pd.read_csv("C:/Maxwell_Data/topo_data/topo_dl_data/topo_dl_data/processing/val.csv")
set_test = pd.read_csv("C:/Maxwell_Data/topo_data/topo_dl_data/topo_dl_data/processing/test.csv")
set_train_sub = set_train.sample(1500)

#%%
testLst = []
for i in range(5195):
    if  (os.path.isfile(set_val.iloc[i, 1]) == True and os.path.isfile(set_val.iloc[i, 2]) == True):
        testLst.append("Exists")
    else:
        testLst.append("Missing")

set_val["Present"] = testLst
set_val2 = set_val[set_val["Present"] == 'Exists']

#%%
testLst = []
for i in range(26556):
    if  (os.path.isfile(set_train.iloc[i, 1]) == True and os.path.isfile(set_train.iloc[i, 2]) == True):
        testLst.append("Exists")
    else:
        testLst.append("Missing")

set_train["Present"] = testLst
set_train2 = set_train[set_train["Present"] == 'Exists']

#%%
#print(os.path.isfile(set_test.iloc[1,1])) 
print(set_test.iloc[1,1])
# In[39]:

class SegData(Dataset):
    
    def __init__(self, df, classes=None, transform=None):
        self.df = df
        self.transform = transform
        
    def __getitem__(self, idx):
        image_name = self.df.iloc[idx, 1]
        mask_name = self.df.iloc[idx, 2]
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_name, cv2.IMREAD_UNCHANGED)
        image = image.astype('uint8')
        mask = mask.astype('uint8')
        mask = np.expand_dims(mask, axis=2)
        
        if(self.transform is not None):
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)   
            image = image.permute(2, 0, 1)
            image = image.float()/255
            mask = mask.permute(2, 0, 1)
            mask = mask.float()
        else: 
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)
            image = image.permute(2, 0, 1)
            image = image.float()/255
            mask = mask.permute(2, 0, 1)
            mask = mask.float()
        return image, mask  
    def __len__(self):  # return count of sample we have
        return len(self.df)

# In[40]:

val_transform = A.Compose(
    [A.PadIfNeeded(min_height=256, min_width=256, border_mode=4), A.Resize(256, 256),]
)

# In[41]:
    
train_transform = A.Compose(
    [
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=4),
        A.Resize(256, 256),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.MedianBlur(blur_limit=3, always_apply=False, p=0.1),
    ]
)

# In[42]:
    
train = SegData(df=set_train_sub, classes=CLASSES, transform=train_transform)
val = SegData(df=set_val2, classes=CLASSES, transform=val_transform)

# In[43]:

print("Number of Training Samples: " + str(len(train)) + " Number of Validation Samples: " + str(len(val)))

# In[44]:

trainDL = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
valDL =  torch.utils.data.DataLoader(val, batch_size=10, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)

# In[46]:

batch = next(iter(trainDL))
images, labels = batch
print(images.shape, labels.shape, type(images), type(labels), images.dtype, labels.dtype)

# In[47]:

testImg = images[1]
testMsk = labels[1]
print(testImg.shape, testImg.dtype, type(testImg), testMsk.shape, testMsk.dtype, type(testMsk), testImg.min(), testImg.max(), testMsk.min(), testMsk.max())

# In[49]:

plt.imshow(testImg.permute(1,2,0))

# In[50]:
    
plt.imshow(testMsk.permute(1,2,0))


#%%
batch = next(iter(valDL))
images, labels = batch
print(images.shape, labels.shape, type(images), type(labels), images.dtype, labels.dtype)

#%%
testImg = images[1]
testMsk = labels[1]
print(testImg.shape, testImg.dtype, type(testImg), testMsk.shape, testMsk.dtype, type(testMsk), testImg.min(), testImg.max(), testMsk.min(), testMsk.max())

#%%
plt.imshow(testImg.permute(1,2,0))

#%%
plt.imshow(testMsk.permute(1,2,0))
#%%

model = smp.FPN(
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
    valid_logs = valid_epoch.run(valDL)
    
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

#%%


#%%
max_score = 0

for i in range(0, 10):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(trainDL)
    valid_logs = valid_epoch.run(valDL)
    
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5

#%%
test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

logs = test_epoch.run(valDL)


# evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

logs = test_epoch.run(valDL)



#%%

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

for i in range(5):
    n = np.random.choice(len(val))
    
    image_vis = val[n][0].astype('uint8')
    image, gt_mask = val[n]
    
    gt_mask = gt_mask.squeeze()
    
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        
    visualize(
        image=image_vis, 
        ground_truth_mask=gt_mask, 
        predicted_mask=pr_mask
    )

