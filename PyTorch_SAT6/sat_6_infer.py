# Based on the following example: https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/
# Based on the following example: https://heartbeat.fritz.ai/basics-of-image-classification-with-pytorch-2f8973c51864
# Based on the following example: https://www.kaggle.com/basu369victor/pytorch-tutorial-the-classification

#Read in libraries ===========================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt 
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torchvision
import torchvision.transforms as transforms
import rasterio

# Change directory
os.chdir('C:/Maxwell_Data/archive/chips2')

# Check if GPU is available ===================================
avail = torch.cuda.is_available()
devCnt = torch.cuda.device_count()
devName = torch.cuda.get_device_name(0)
print("Available: " + str(avail) + ", Count: " + str(devCnt) + ", Name: " + str(devName))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Read in CSV data =============================================================
testSet = pd.read_csv("C:/Maxwell_Data/archive/chips2/y_test.csv")
len(testSet)

# Define CNN model
class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(8),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(16),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear_layers = Sequential(
            Linear(16 * 7 * 7, 6)
        )
    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
        
# Initiate the model =========================
model = Net()

# Load saved model and weights ===========================
best_weights = torch.load('C:/Maxwell_Data/archive/chips2/sat6_model.pt')
model.load_state_dict(best_weights)

# Set model to eval model ==========================
model.eval()

# checking if GPU is available and push model to GPU =======================
if torch.cuda.is_available():
    model = model.cuda()
    
# Sumarize model ===========================================
print(model)

# Define image directory and CUDA device
directory = "C:/Maxwell_Data/archive/chips2/test_x/"
DEVICE = 'cuda'

#Initiate a tensor to save predictions to
result_class=torch.empty((1), dtype=torch.int64, device = 'cuda')


# Loop through and predict all testing images
for index, row in testSet.iterrows():
    source = rasterio.open(directory + str(testSet.iloc[index, 0]) + ".png")
    image = source.read()
    source.close()
    image = image.astype('uint8')
    image = torch.from_numpy(image)
    image = image.to(DEVICE)
    image = image.float()/255
    image = image.unsqueeze(0)
    ten_p = model(image)
    ten_p3 = torch.argmax(ten_p, dim=1)
    result_class = torch.cat((result_class, ten_p3))
    
# Merge predictions with reference data and save to CSV  
resultNP = result_class.cpu().numpy()
resultNP2 = resultNP[1:]
testSet["predicted"] = resultNP2
testSet.to_csv("C:/Maxwell_Data/archive/chips2/test_result4.csv")

