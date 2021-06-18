#Based on this example: https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb
#Import Libraries =======================================
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import os
import cv2
import torch
from torch.utils.data.dataset import Dataset
import albumentations as A
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split

# Change directory
os.chdir('C:/Maxwell_Data/inria')

# Define Variables ========================================
ENCODER = "resnet18"
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ["building"]
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'

#Check if GPU is available ===================================
avail = torch.cuda.is_available()
devCnt = torch.cuda.device_count()
devName = torch.cuda.get_device_name(0)
print("Available: " + str(avail) + ", Count: " + str(devCnt) + ", Name: " + str(devName))

#Read in chip lists =======================================
austin = pd.read_csv("C:/Maxwell_Data/inria/chips2/austin.csv")
vienna = pd.read_csv("C:/Maxwell_Data/inria/chips2/vienna.csv")
tyrol =  pd.read_csv("C:/Maxwell_Data/inria/chips2/tyrol.csv")

#Create training and test splits
austinTrain, austinTest = train_test_split(austin, test_size=0.3)
viennaTrain, viennaTest = train_test_split(austin, test_size=0.3)

# Subclass Dataset to create a training set ============================
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
            mask = mask.float()/255
        else: 
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)
            image = image.permute(2, 0, 1)
            image = image.float()/255
            mask = mask.permute(2, 0, 1)
            mask = mask.float()/255
        return image, mask  
    def __len__(self):  
        return len(self.df)

#Define tranforms using Albumations =======================================
test_transform = A.Compose(
    [A.PadIfNeeded(min_height=256, min_width=256, border_mode=4), A.Resize(256, 256),]
)

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

#===================Train on Austin Data===========================================

# Create the datasets   ================================================ 
austinTrainDS = SegData(austinTrain, classes=CLASSES, transform=train_transform)
austinTestDS = SegData(austinTest, classes=CLASSES, transform=test_transform)
print("Number of Training Samples: " + str(len(austinTrainDS)) + " Number of Validation Samples: " + str(len(austinTestDS)))

# Define DataLoaders ==============================================
trainDL = torch.utils.data.DataLoader(austinTrainDS, batch_size=10, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
testDL =  torch.utils.data.DataLoader(austinTestDS, batch_size=10, shuffle=False, sampler=None,
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

# Plot exmaple mask ======================================
plt.imshow(testMsk.permute(1,2,0))

# Initiate DeepLabv3+ Model ======================================
model = smp.DeepLabV3Plus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

#Define Loss and Metrics to Monitor ======================================
loss = smp.utils.losses.DiceLoss() + smp.utils.losses.BCELoss()
metrics = [
    smp.utils.metrics.Fscore(beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None),
    smp.utils.metrics.Accuracy(threshold=0.5, activation=None, ignore_channels=None),
    smp.utils.metrics.Recall(eps=1e-7, threshold=0.5, activation=None, ignore_channels=None),
    smp.utils.metrics.Precision(eps=1e-7, threshold=0.5, activation=None, ignore_channels=None),
]

# Define Optimizer (Adam in this case) and learning rate ============================
optimizer = torch.optim.Adam([ 
    dict(params=model.parameters()),
])

# Define training epock =====================================
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

# Define testing epoch =====================================
test_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

# Train model for 10 epochs ==================================
max_score = 0

for i in range(0, 10):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(trainDL)
    test_logs = test_epoch.run(testDL)
    
    # do something (save model, change lr, etc.)
    if max_score < test_logs['fscore']:
        max_score = test_logs['fscore']
        torch.save(model, './best_model_austin.pth')
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')

# ====================== Train on Vienna Data ======================================

# Create the datasets   ================================================ 
viennaTrainDS = SegData(viennaTrain, classes=CLASSES, transform=train_transform)
viennaTestDS = SegData(viennaTest, classes=CLASSES, transform=test_transform)
print("Number of Training Samples: " + str(len(austinTrainDS)) + " Number of Validation Samples: " + str(len(austinTestDS)))

# Define DataLoaders ==============================================
trainDL = torch.utils.data.DataLoader(viennaTrainDS, batch_size=10, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
testDL =  torch.utils.data.DataLoader(viennaTestDS, batch_size=10, shuffle=False, sampler=None,
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

# Plot exmaple mask ======================================
plt.imshow(testMsk.permute(1,2,0))

# Initiate UNet Model ======================================
model = smp.DeepLabV3Plus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

#Define Loss and Metrics to Monitor ======================================
loss = smp.utils.losses.DiceLoss() + smp.utils.losses.BCELoss()
metrics = [
    smp.utils.metrics.Fscore(beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None),
    smp.utils.metrics.Accuracy(threshold=0.5, activation=None, ignore_channels=None),
    smp.utils.metrics.Recall(eps=1e-7, threshold=0.5, activation=None, ignore_channels=None),
    smp.utils.metrics.Precision(eps=1e-7, threshold=0.5, activation=None, ignore_channels=None),
]

# Define Optimizer (Adam in this case) and learning rate ============================
optimizer = torch.optim.Adam([ 
    dict(params=model.parameters()),
])

# Define training epock =====================================
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

# Define testing epoch =====================================
test_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

# Train model for 10 epochs ==================================
max_score = 0

for i in range(0, 10):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(trainDL)
    test_logs = test_epoch.run(testDL)
    
    # do something (save model, change lr, etc.)
    if max_score < test_logs['fscore']:
        max_score = test_logs['fscore']
        torch.save(model, './best_model_vienna.pth')
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')

# Load saved model ============================================
best_model = torch.load('./best_model_vienna.pth')

#create validaiton dataset=======================================================
valDS = SegData(tyrol)

#Create validation dataloaer======================================================
valDL =  torch.utils.data.DataLoader(valDS, batch_size=10, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
           
# ===================== Validate Austin model on Tyrol data ============================

# Load saved model ============================================
best_model = torch.load('./best_model_austin.pth')

# Evaluate model on validation set
val_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

logs = val_epoch.run(valDL)
print(logs)

# ===================== Validate Vienna model on Tyrol data ============================

# Load saved model ============================================
best_model = torch.load('./best_model_vienna.pth')

# Evaluate model on validation set
val_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

logs = val_epoch.run(valDL)
print(logs)

#Visualize images, masks, and predictions=======================================
def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 10))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

for i in range(10):
    n = np.random.choice(len(valDS))
    
    image_vis = valDS[n][0].permute(1,2,0)
    image_vis = image_vis.numpy()*255
    image_vis = image_vis.astype('uint8')
    image, gt_mask = valDS[n]
    
    gt_mask = gt_mask.squeeze()
    
    x_tensor = image.to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        
    visualize(
        image=image_vis,
        ground_truth_mask=gt_mask, 
        predicted_mask=pr_mask
    )

