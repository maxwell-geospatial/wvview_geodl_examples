library(dplyr)
library(stringr)
library(terra)
library(imager)

images <- list.files("C:/Maxwell_Data/topo_work/data/img", pattern="\\.tif$")
masks <- list.files("C:/Maxwell_Data/topo_work/data/msk", pattern="\\.tif$")

input_topos <- data.frame(Images = images, Masks = masks)

input_topos$img_full <- paste0("C:/Maxwell_Data/topo_work/data/img/", input_topos$Images)

input_topos$msk_full <- paste0("C:/Maxwell_Data/topo_work/data/msk/", input_topos$Masks)

#Loop to extract components of file paths to columns
topo_prep <- data.frame()
for(i in 1:nrow(input_topos)) {
  ky_All <- str_split(input_topos[i, 1], "_", simplify=TRUE)
  topo_prep <- rbind(topo_prep, ky_All)
}

names(topo_prep) <- c("STATE", "NAME", "SCANID", "YEAR", "SCALE", "GEO")

input_topos2 <- cbind(input_topos, topo_prep)
input_topos2$STATE <- as.factor(input_topos2$STATE)

ky_topos <- input_topos2 %>% dplyr::filter(STATE == "KY")


#List all topo names
quad_names <- as.data.frame(levels(as.factor(ky_topos$NAME)))
names(quad_names) <- "NAME"

set.seed(42)
topos_train <- quad_names %>% sample_n(70)
topos_remaining <- setdiff(quad_names, topos_train)
set.seed(43)
topos_val <- topos_remaining %>% sample_frac(.5)
topos_test <- setdiff(topos_remaining, topos_val)
topos_train$select <- 1
topos_val$select <- 2
topos_test$select <- 3
topos_combined <- rbind(topos_train, topos_val, topos_test)

#Join sampling results back to folder list
ky_topos2 <- left_join(ky_topos, topos_combined, by="NAME")

#Separate into training, validation, and testing splits
train_topos <- ky_topos2 %>% filter(select==1)
val_topos <- ky_topos2 %>% filter(select==2)
test_topos <- ky_topos2 %>% filter(select==3)

write.csv(val_topos, "C:/Maxwell_Data/topo_work/val_topos.csv")


for(t in 1:nrow(train_topos)){
  chipIt(image= train_topos[t, c("img_full")], 
         mask= train_topos[t, c("msk_full")],
         n_channels=3,
         size=256, stride_x=256, stride_y=256, 
         outDir= "C:/Maxwell_Data/topo_work/chips/train", 
         mode="Divided")
}

for(t in 1:nrow(test_topos)){
  chipIt(image= test_topos[t, c("img_full")], 
         mask= test_topos[t, c("msk_full")],
         n_channels=3,
         size=256, stride_x=256, stride_y=256, 
         outDir= "C:/Maxwell_Data/topo_work/chips/test", 
         mode="Divided")
}

for(t in 1:nrow(val_topos)){
  chipIt(image= val_topos[t, c("img_full")], 
         mask= val_topos[t, c("msk_full")],
         n_channels=3,
         size=256, stride_x=256, stride_y=256, 
         outDir= "C:/Maxwell_Data/topo_work/chips/val", 
         mode="Divided")
}


