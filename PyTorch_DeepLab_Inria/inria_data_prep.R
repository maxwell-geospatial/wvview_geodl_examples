library(dplyr)
library(stringr)
library(terra)
library(imager)

#List out all images and masks
images <- list.files("C:/Maxwell_Data/inria/NEW2-AerialImageDataset/AerialImageDataset/train/images", pattern="\\.tif$")
masks <- list.files("C:/Maxwell_Data/inria/NEW2-AerialImageDataset/AerialImageDataset/train/gt", pattern="\\.tif$")

#Build data frame of images and masks
input_cities <- data.frame(Images = images, Masks = masks)

#Add city column
input_cities$city <- str_replace_all(c(input_cities$Images), "[:digit:]", "")
input_cities$city <- substr(input_cities$city,1,nchar(input_cities$city)-4)

#Get count of images for each city
input_cities %>% group_by(city) %>% count()

#Create columns with full path
input_cities$img_full <- paste0("C:/Maxwell_Data/inria/NEW2-AerialImageDataset/AerialImageDataset/train/images/", input_cities$Images)
input_cities$msk_full <- paste0("C:/Maxwell_Data/inria/NEW2-AerialImageDataset/AerialImageDataset/train/gt/", input_cities$Masks)

#Create separate data frames for each city
austin <- input_cities %>% filter(city == "austin")
chicago <- input_cities %>% filter(city == "chicago")
kitsap <- input_cities %>% filter(city == "kitsap")
tyrol <- input_cities %>% filter(city == "tyrol-w")
vienna <- input_cities %>% filter(city == "vienna")

write.csv(tyrol, "C:/Maxwell_Data/inria/tyrol_images.csv")

#Crate chips for each city
for(t in 1:nrow(austin)){
  chipIt(image= austin[t, c("img_full")], 
         mask= austin[t, c("msk_full")],
         n_channels=3,
         size=256, stride_x=256, stride_y=256, 
         outDir= "C:/Maxwell_Data/inria/chips2/austin", 
         mode="All")
}

for(t in 1:nrow(chicago)){
  chipIt(image= chicago[t, c("img_full")], 
         mask= chicago[t, c("msk_full")],
         n_channels=3,
         size=256, stride_x=256, stride_y=256, 
         outDir= "C:/Maxwell_Data/inria/chips2/chicago", 
         mode="All")
}

for(t in 1:nrow(kitsap)){
  chipIt(image= kitsap[t, c("img_full")], 
         mask= kitsap[t, c("msk_full")],
         n_channels=3,
         size=256, stride_x=256, stride_y=256, 
         outDir= "C:/Maxwell_Data/inria/chips2/kitsap", 
         mode="All")
}

for(t in 1:nrow(tyrol)){
  chipIt(image= tyrol[t, c("img_full")], 
         mask= tyrol[t, c("msk_full")], 
         n_channels=3,
         size=256, stride_x=256, stride_y=256, 
         outDir= "C:/Maxwell_Data/inria/chips2/tyrol", 
         mode="All")
}

for(t in 1:nrow(vienna)){
  chipIt(image= vienna[t, c("img_full")], 
         mask= vienna[t, c("msk_full")],
         n_channels=3,
         size=256, stride_x=256, stride_y=256, 
         outDir= "C:/Maxwell_Data/inria/chips2/vienna", 
         mode="All")
}

#List out all images and masks for each city
austin_img <- list.files("C:/Maxwell_Data/inria/chips2/austin/images", pattern="\\.png$", full.names=TRUE)
austin_masks <- list.files("C:/Maxwell_Data/inria/chips2/austin/masks", pattern="\\.png$",full.names=TRUE)

vienna_img <- list.files("C:/Maxwell_Data/inria/chips2/vienna/images", pattern="\\.png$", full.names=TRUE)
vienna_masks <- list.files("C:/Maxwell_Data/inria/chips2/vienna/masks", pattern="\\.png$", full.names=TRUE)

tyrol_img <- list.files("C:/Maxwell_Data/inria/chips2/tyrol/images", pattern="\\.png$", full.names=TRUE)
tyrol_masks <- list.files("C:/Maxwell_Data/inria/chips2/tyrol/masks", pattern="\\.png$", full.names=TRUE)

kitsap_img <- list.files("C:/Maxwell_Data/inria/chips2/kitsap/images", pattern="\\.png$", full.names=TRUE)
kitsap_masks <- list.files("C:/Maxwell_Data/inria/chips2/kitsap/masks", pattern="\\.png$", full.names=TRUE)

#Make data frame of image and and mask chip paths per city
austin_chips <- data.frame(Images = austin_img, Masks = austin_masks)
vienna_chips <- data.frame(Images = vienna_img, Masks = vienna_masks)
tyrol_chips <- data.frame(Images = tyrol_img, Masks = tyrol_masks)
kitsap_chips <- data.frame(Images = kitsap_img, Masks = kitsap_masks)

#Write out to CSV
write.csv(austin_chips, "C:/Maxwell_Data/inria/chips2/austin.csv")
write.csv(vienna_chips, "C:/Maxwell_Data/inria/chips2/vienna.csv")
write.csv(tyrol_chips, "C:/Maxwell_Data/inria/chips2/tyrol.csv")
write.csv(kitsap_chips, "C:/Maxwell_Data/inria/chips2/kitsap.csv")
