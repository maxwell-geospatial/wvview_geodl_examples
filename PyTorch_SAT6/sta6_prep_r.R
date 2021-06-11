library(dplyr)
library(stringr)
library(terra)
library(imager)
library(readr)

setwd("C:/Maxwell_Data/archive")

train_x <- read.csv("X_train_sat6.csv", header=FALSE)
train_y <- read.csv("y_train_sat6.csv", header=FALSE)
test_x <- read.csv("X_test_sat6.csv", header=FALSE)
test_y<- read.csv("y_test_sat6.csv", header=FALSE)

names(train_y) <- c("building", "barren", "trees", "grasslands", "road", "water")
names(test_y) <- c("building", "barren", "trees", "grasslands", "road", "water")
train_labels <- colnames(train_y)[max.col(train_y)]
test_labels <- colnames(test_y)[max.col(test_y)]


TrainTest <- cbind(train_labels, train_x)
names(TrainTest)[1] <- "class"
trainSet <- TrainTest %>% group_by(class) %>% sample_frac(.7, replace=FALSE)
valSet <- setdiff(TrainTest, trainSet)
testSet <- cbind(test_labels, test_x)

xData <- as.numeric(as.data.frame(trainSet[900, 2:ncol(trainSet)]))
b <- as.matrix(xData[1:(28*28)], nrow=28, ncol=28, byrow=TRUE)
g <- as.matrix(xData[((28*28)+1):(28*28*2)], nrow=28, ncol=28)
r <- as.matrix(xData[((28*28*2)+1):(28*28*3)], nrow=28, ncol=28)
n <- as.matrix(xData[((28*28*3)+1):(28*28*4)], nrow=28, ncol=28)
xArray <- array(xData, c(28,28,4))
xImg <- as.cimg(xArray, x=28, y=28, cc=4)

xRast <- rast(b)
plotRGB(xRast)
