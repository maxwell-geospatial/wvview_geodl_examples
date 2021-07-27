#Based on this example: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb
#Based on this example: https://towardsdatascience.com/super-convergence-with-just-pytorch-c223c0fc1e51
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
from sklearn.metrics import confusion_matrix

# Change directory
os.chdir("C:/Maxwell_Data/wvlcDL_2/")

#Read training and testing chip lists =======================================
train = pd.read_csv("C:/Maxwell_Data/wvlcDL_2/wvlcdl_5m/chips/trainSet.csv")
#train2 = train[0:100]
val = pd.read_csv("C:/Maxwell_Data/wvlcDL_2/wvlcdl_5m/chips/valSet.csv")
#val2 = val[0:100]

# Define Variables ========================================
MULTICLASS_MODE: str = "multiclass"
ENCODER = "resnet34"
ENCODER_WEIGHTS = None
CLASSES = ['background', 'barren', 'crop', 'forest', 'grass', 'imperv', 'mixdev', 'water']
ACTIVATION = None

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
    [A.PadIfNeeded(min_height=256, min_width=256, border_mode=4), A.Resize(256, 256),]
)

train_transform = A.Compose(
    [
        A.PadIfNeeded(min_height=256, min_width=256, border_mode=4),
        A.Resize(256, 256),
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
trainDL = torch.utils.data.DataLoader(trainDS, batch_size=10, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
valDL =  torch.utils.data.DataLoader(valDS, batch_size=10, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)

# Check Tensor shapes ======================================================
batch = next(iter(trainDL))
images, labels = batch
print(images.shape, labels.shape, type(images), type(labels), images.dtype, labels.dtype)

# Check first sample from batch =================================================
testImg = images[1]
testMsk = labels[1]
print(testImg.shape, testImg.dtype, type(testImg), testMsk.shape, 
testMsk.dtype, type(testMsk), testImg.min(), 
testImg.max(), testMsk.min(), testMsk.max())

# Plot example image =====================================
plt.imshow(testImg.permute(1,2,0))
plt.show()
plt.close()

# Plot exmaple mask ======================================
plt.imshow(testMsk.permute(1,2,0))
plt.show()
plt.close()

# Define function to calculate overall accuracy and all class precion, recal, and f1 scores from confusion matrix
def my_metrics(cm):
  oa = np.sum(np.diagonal(cm))/np.sum(cm)
  r_1 = cm[0][0]/np.sum(cm[:,0])
  r_2 = cm[1][1]/np.sum(cm[:,1])
  r_3 = cm[2][2]/np.sum(cm[:,2])
  r_4 = cm[3][3]/np.sum(cm[:,3])
  r_5 = cm[4][4]/np.sum(cm[:,4])
  r_6 = cm[5][5]/np.sum(cm[:,5])
  r_7 = cm[6][6]/np.sum(cm[:,6])
  p_1 = cm[0][0]/np.sum(cm[0,:])
  p_2 = cm[1][1]/np.sum(cm[1,:])
  p_3 = cm[2][2]/np.sum(cm[2,:])
  p_4 = cm[3][3]/np.sum(cm[3,:])
  p_5 = cm[4][4]/np.sum(cm[4,:])
  p_6 = cm[5][5]/np.sum(cm[5,:])
  p_7 = cm[6][6]/np.sum(cm[6,:])
  f_1 = (2*r_1*p_1)/(r_1+p_1)
  f_2 = (2*r_2*p_2)/(r_2+p_2)
  f_3 = (2*r_3*p_3)/(r_3+p_3)
  f_4 = (2*r_4*p_4)/(r_4+p_4) 
  f_5 = (2*r_5*p_5)/(r_5+p_5)
  f_6 = (2*r_6*p_6)/(r_6+p_6)
  f_7 = (2*r_7*p_7)/(r_7+p_7)
  met_out = pd.Series([oa, r_1, p_1, f_1, r_2, p_2, f_2, 
  r_3, p_3, f_3, r_4, p_4, f_4, 
  r_5, p_5, f_5, r_6, p_6, f_6, r_7, p_7, f_7], 
  index=["oa", "r_1", "p_1", "f_1", "r_2", "p_2", "f_2", 
  "r_3", "p_3", "f_3", "r_4", "p_4", "f_4", 
  "r_5", "p_5", "f_5", "r_6", "p_6", "f_6", "r_7", "p_7", "f_7"])
  return met_out

# Initiate UNet++ Model ======================================
model = smp.UnetPlusPlus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    decoder_use_batchnorm=True,
    in_channels=4,
    classes=8,
    activation=ACTIVATION,
).to(torch.device("cuda", 0))

# Define Loss and Metrics to Monitor (Make sure mode = "multiclass") ======================================
criterion = smp.losses.DiceLoss(mode="multiclass", from_logits=True, ignore_index=0)
#loss_fn.__name__ = 'Dice_loss'

# Define Optimizerand learning rate ============================
optimizer = torch.optim.SGD(model.parameters(), lr=2e-3)

# Set up scheduler for One Cycle Learning= ==========================
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-1, epochs=20, steps_per_epoch=len(trainDL))

# Make empty dataframes to save results to ==================================
all_metsTrain = pd.DataFrame(columns=["oa", "r_1", "p_1", "f_1", 
"r_2", "p_2", "f_2", "r_3", "p_3", "f_3", "r_4", "p_4", "f_4", 
"r_5", "p_5", "f_5", "r_6", "p_6", "f_6", "r_7", "p_7", "f_7", "loss"])
all_metsVal = pd.DataFrame(columns=["oa", "r_1", "p_1", "f_1", 
"r_2", "p_2", "f_2", "r_3", "p_3", "f_3", "r_4", "p_4", "f_4", 
"r_5", "p_5", "f_5", "r_6", "p_6", "f_6", "r_7", "p_7", "f_7", "loss"])

# Get number of batches in training set ==========================
size = len(trainDL.dataset)

# Define number of training epochs ========================
epochs = 20

# Define number of batches before applying optimization to update weights ==============
accum_iter = 20

# Define number of classes to differentiate ========================
n_classes = 7

# Define device =============================
device="cuda"

# Loop over epochs for training and validation =====================================
for t in range(epochs):
    # Define empty matrices
    cmTrain = np.zeros([7, 7], dtype=int)
    cmVal = np.zeros([7, 7], dtype=int)
    # Loop through training batches
    for batch_idx, (x_batch, y_batch) in enumerate(trainDL):
        # Send batches to GPU
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.set_grad_enabled(True):
          # Make prediction of data batch
          pred = model(x_batch)
          # Remove prediction for background class
          pred2 = pred[:, 1:n_classes+1, :, :]
          # Return class prediction with highest logit
          pred3 = torch.argmax(pred2, dim=1)
          # Send preddiction to CPU and flatten
          predNP = pred3.detach().cpu().numpy().flatten()
          # Add 1 so that codes match
          predNP = predNP + 1
          # Send reference data to CPU and flatten
          refNP = y_batch.detach().cpu().numpy().flatten()
          # Generate error matrix
          cmTB = confusion_matrix(refNP, predNP, labels=[1,2,3,4,5,6,7])
          # Calculate loss
          lossT = criterion(pred, y_batch)
          # Normalize for accumulation
          lossT = lossT/accum_iter
          #Backpropogate
          lossT.backward()
          # Step for one cycle learning
          scheduler.step()
          # Optimize after only 20 batches
          if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == size):
              optimizer.step()
              optimizer.zero_grad()
          # Add batch to error matrix
          cmTrain += cmTB
          # Same process for validation batches, minus backpropogation and optimization
    for batch_idx, (x_batch, y_batch) in enumerate(valDL):
        with torch.no_grad():
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            pred2 = pred[:, 1:n_classes+1, :, :]
            pred3= torch.argmax(pred2, dim=1)
            predNP = pred3.detach().cpu().numpy().flatten()
            predNP = predNP + 1
            refNP = y_batch.detach().cpu().numpy().flatten()
            cmVB = confusion_matrix(refNP, predNP, labels=[1,2,3,4,5,6,7])
            lossV = criterion(pred, y_batch)
            cmVal += cmVB
    # Use function to calculate metrics
    metsTrain = my_metrics(cmTrain)
    # Append loss to metrics
    metsTrain = metsTrain.append(pd.Series(lossT.detach().cpu().numpy(), index=["loss"]))
    # NA to 0
    metsTrain = metsTrain.fillna(0)
    # Append results to data frame
    all_metsTrain = all_metsTrain.append(metsTrain, ignore_index=True)
    # Repeat for validation data frame
    metsVal = my_metrics(cmVal)
    metsVal = metsVal.append(pd.Series(lossV.detach().cpu().numpy(), index=["loss"]))
    metsVal = metsVal.fillna(0)
    all_metsVal = all_metsVal.append(metsVal, ignore_index=True)
    # Save metrics at end of each epoch
    all_metsTrain.to_csv("train_epoch_metrics.csv")
    all_metsVal.to_csv("val_epoch_metrics.csv")
    # Save model at end of each epoch
    model_name = "model_out_" + str(t) + ".pth"
    torch.save(model.state_dict(), model_name)
    # Print metrics at end of each epoch
    print(f"Epoch {t+1}\nTrain Loss: {lossT}\nVal Loss: {lossV}\nTraining Metrics: {metsTrain}\nVal Metrics: {metsVal}")





