library(dplyr)

#Make dataframe of image names and path to image and masks========================================
images <- list.files("C:/Maxwell_Data/wvlcDL_2/chips3/images", pattern="\\.tif$")
chips <- data.frame(Images = images)
chips$img_full <- paste0("C:/Maxwell_Data/wvlcDL_2/chips3/images/", chips$Images)
chips$msk_full <- paste0("C:/Maxwell_Data/wvlcDL_2/chips3/masks/", chips$Images)


#Split into training and validation sets========================================================
train_chips <- chips %>% sample_frac(0.80, replace=FALSE)
val_chips <- setdiff(chips, train_chips)

#Write out to CSV===============================================================================
write.csv(train_chips, "C:/Maxwell_Data/wvlcDL_2/chips3/trainSet.csv")
write.csv(val_chips, "C:/Maxwell_Data/wvlcDL_2/chips3/valSet.csv")

