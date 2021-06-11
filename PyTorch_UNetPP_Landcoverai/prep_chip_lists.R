library(dplyr)
library(stringr)
library(terra)
library(imager)

test_chips <- read.table("C:/Maxwell_Data/landcover/test.txt", header=FALSE)
train_chips <- read.table("C:/Maxwell_Data/landcover/train.txt", header=FALSE)
val_chips <- read.table("C:/Maxwell_Data/landcover/val.txt", header=FALSE)

names(test_chips) <- "name"
names(train_chips) <- "name"
names(val_chips) <- "name"

test_chips$img_path <- paste0("C:/Maxwell_Data/landcover/chips/img/", test_chips$name, ".png")
test_chips$mask_path <- paste0("C:/Maxwell_Data/landcover/chips/msk/", test_chips$name, ".png")

train_chips$img_path <- paste0("C:/Maxwell_Data/landcover/chips/img/", train_chips$name, ".png")
train_chips$mask_path <- paste0("C:/Maxwell_Data/landcover/chips/msk/", train_chips$name, ".png")

val_chips$img_path <- paste0("C:/Maxwell_Data/landcover/chips/img/", val_chips$name, ".png")
val_chips$mask_path <- paste0("C:/Maxwell_Data/landcover/chips/msk/", val_chips$name, ".png")

write.csv(test_chips, "C:/Maxwell_Data/landcover/test_chips.csv")
write.csv(train_chips, "C:/Maxwell_Data/landcover/train_chips.csv")
write.csv(val_chips, "C:/Maxwell_Data/landcover/val_chips.csv")