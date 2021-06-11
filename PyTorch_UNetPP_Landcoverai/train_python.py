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

# Change directory
os.chdir("C:/Maxwell_Data/landcover")

#Read in chip lists =======================================
train = pd.read_csv("C:/Maxwell_Data/landcover/train_chips.csv")
test = pd.read_csv("C:/Maxwell_Data/landcover/test_chips.csv")

# Define Variables ========================================
ENCODER = "resnet18"
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ["background", "building", "woodlands", "water"]
ACTIVATION = 'softmax'
DEVICE = 'cuda'

#Check if GPU is available ===================================
avail = torch.cuda.is_available()
devCnt = torch.cuda.device_count()
devName = torch.cuda.get_device_name(0)
print("Available: " + str(avail) + ", Count: " + str(devCnt) + ", Name: " + str(devName))


# Subclass Dataset to create a training set ============================
class SegData(Dataset):
    
    def __init__(self, df, classes=None, transform=None):
        self.df = df
        self.transform = transform
        
    def __getitem__(self, idx):
        image_name = self.df.iloc[idx, 2]
        mask_name = self.df.iloc[idx, 3]
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_name, cv2.IMREAD_UNCHANGED)
        image = image.astype('uint8')
        mask = mask.astype('uint8')
        mask = mask[:,:,0]
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
            mask = mask.long()
        else: 
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)
            image = image.permute(2, 0, 1)
            image = image.float()/255
            mask = mask.permute(2, 0, 1)
            mask = mask.long
        return image, mask  
    def __len__(self):  
        return len(self.df)

#Define tranforms using Albumations =======================================
test_transform = A.Compose(
    [A.PadIfNeeded(min_height=512, min_width=512, border_mode=4), A.Resize(512, 512),]
)

train_transform = A.Compose(
    [
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=4),
        A.Resize(512, 512),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.MedianBlur(blur_limit=3, always_apply=False, p=0.1),
    ]
)

# Create the datasets   ================================================ 
trainDS = SegData(train, classes=CLASSES, transform=train_transform)
testDS = SegData(test, classes=CLASSES, transform=test_transform)
print("Number of Training Samples: " + str(len(train)) + " Number of Validation Samples: " + str(len(test)))

# Define DataLoaders ==============================================
trainDL = torch.utils.data.DataLoader(trainDS, batch_size=8, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
testDL =  torch.utils.data.DataLoader(testDS, batch_size=8, shuffle=False, sampler=None,
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
model = smp.UnetPlusPlus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    in_channels=3,
    classes=len(CLASSES),
)

#Define Loss and Metrics to Monitor ======================================
loss = smp.utils.losses.CrossEntropyLoss()
metrics = [
    smp.utils.metrics.Accuracy(threshold=0.5, activation=None, ignore_channels=None)
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

for i in range(0, 20):
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(trainDL)
    test_logs = test_epoch.run(testDL)
    
    # do something (save model, change lr, etc.)
    if max_score < test_logs['Accuracy']:
        max_score = test_logs['Accuracy']
        torch.save(model, './best_model.pth')
        print('Model saved!')
        
    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')

# Load saved model ============================================
best_model = torch.load('./best_model.pth')


# Subclass Dataset to create a val set ============================
class SegDataVal(Dataset):
    
    def __init__(self, classes=None, transform=None):
        testP = pd.read_csv("C:/Maxwell_Data/topo_work/valP.csv")
        testB = pd.read_csv("C:/Maxwell_Data/topo_work/valB.csv")
        self.test = pd.concat([testP, testB])
        self.transform = transform
        
    def __getitem__(self, idx):
        image_name = self.test.iloc[idx, 1]
        mask_name = self.test.iloc[idx, 2]
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
        return len(self.test)
        
        
# Subclass Dataset to create a val set ============================
class SegDataVal2(Dataset):
    
    def __init__(self, classes=None, transform=None):
        self.test = pd.read_csv("C:/Maxwell_Data/topo_work/valP.csv")
        self.transform = transform
        
    def __getitem__(self, idx):
        image_name = self.test.iloc[idx, 1]
        mask_name = self.test.iloc[idx, 2]
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
        return len(self.test)

#create validaiton dataset=======================================================
valDS = SegDataVal2()

#Create validation dataloaer======================================================
valDL =  torch.utils.data.DataLoader(valDS, batch_size=10, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)


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
