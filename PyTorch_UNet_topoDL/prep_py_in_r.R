library(reticulate)
use_python("C:/Users/amaxwel6/Anaconda3/envs/pytorchSB/python.exe", required=TRUE)
py_config()

library(dplyr)


train_imagesP <- list.files("C:/Maxwell_Data/topo_work/chips/train/images/positive",full.names=TRUE)
train_masksP <- list.files("C:/Maxwell_Data/topo_work/chips/train/masks/positive", full.names=TRUE)

test_imagesP <- list.files("C:/Maxwell_Data/topo_work/chips/test/images/positive", full.names=TRUE)
test_masksP <- list.files("C:/Maxwell_Data/topo_work/chips/test/masks/positive", full.names=TRUE)

val_imagesP <- list.files("C:/Maxwell_Data/topo_work/chips/val/images/positive", full.names=TRUE)
val_masksP <- list.files("C:/Maxwell_Data/topo_work/chips/val/masks/positive", full.names=TRUE)

train_imagesB <- list.files("C:/Maxwell_Data/topo_work/chips/train/images/background", full.names=TRUE)
train_masksB <- list.files("C:/Maxwell_Data/topo_work/chips/train/masks/background", full.names=TRUE)

test_imagesB <- list.files("C:/Maxwell_Data/topo_work/chips/test/images/background", full.names=TRUE)
test_masksB <- list.files("C:/Maxwell_Data/topo_work/chips/test/masks/background", full.names=TRUE)

val_imagesB <- list.files("C:/Maxwell_Data/topo_work/chips/val/images/background", full.names=TRUE)
val_masksB <- list.files("C:/Maxwell_Data/topo_work/chips/val/masks/background", full.names=TRUE)


train_chipsP <- data.frame(Images = train_imagesP, Masks = train_masksP)
test_chipsP <- data.frame(Images = test_imagesP, Masks = test_masksP)
val_chipsP <- data.frame(Images = val_imagesP, val = val_masksP)
train_chipsB <- data.frame(Images = train_imagesB, Masks = train_masksB)
test_chipsB <- data.frame(Images = test_imagesB, Masks = test_masksB)
val_chipsB <- data.frame(Images = val_imagesB, val = val_masksB)

write.csv(train_chipsP, "C:/Maxwell_Data/topo_work/trainP.csv")
write.csv(test_chipsP, "C:/Maxwell_Data/topo_work/testP.csv")
write.csv(val_chipsP, "C:/Maxwell_Data/topo_work/valP.csv")
write.csv(train_chipsB, "C:/Maxwell_Data/topo_work/trainB.csv")
write.csv(test_chipsB, "C:/Maxwell_Data/topo_work/testB.csv")
write.csv(val_chipsB, "C:/Maxwell_Data/topo_work/valB.csv")