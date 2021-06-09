library(dplyr)
library(stringr)
library(terra)
library(imager)

library(reticulate)
use_python("C:/Users/amaxwel6/Anaconda3/envs/pytorchSB/python.exe", required=TRUE)
py_config()

library(dplyr)
library(stringr)


images <- list.files("C:/Maxwell_Data/inria/NEW2-AerialImageDataset/AerialImageDataset/train/images")
masks <- list.files("C:/Maxwell_Data/inria/NEW2-AerialImageDataset/AerialImageDataset/train/gt")

input_cities <- data.frame(Images = images, Masks = masks)

input_cities$city <- str_replace_all(c(input_cities$Images), "[:digit:]", "")
input_cities$city <- substr(input_cities$city,1,nchar(input_cities$city)-4)

input_cities %>% group_by(city) %>% count()

input_cities$img_full <- paste0("C:/Maxwell_Data/inria/NEW2-AerialImageDataset/AerialImageDataset/train/images/", input_cities$Images)

input_cities$msk_full <- paste0("C:/Maxwell_Data/inria/NEW2-AerialImageDataset/AerialImageDataset/train/gt/", input_cities$Masks)

austin <- input_cities %>% filter(city == "austin")
chicago <- input_cities %>% filter(city == "chicago")
kitsap <- input_cities %>% filter(city == "kitsap")
tyrol <- input_cities %>% filter(city == "tyrol-w")
vienna <- input_cities %>% filter(city == "vienna")

chipIt <- function(image, mask, size=256, stride_x=256, stride_y=256, outDir, mode="All"){
  if(mode == "All"){
    topo1 <- rast(image)
    mask1 <- rast(mask)
    
    fName = basename(image)
    
    if (file.exists(paste0(outDir, "/images"))){
      print("Using Existing Folders")
    } else {
      dir.create(paste0(outDir, "/images"))
      dir.create(paste0(outDir, "/masks"))
    }
    
    across_cnt = ncol(topo1)
    down_cnt = nrow(topo1)
    tile_size_across = size
    tile_size_down = size
    overlap_across = stride_x
    overlap_down = stride_y
    across <- ceiling(across_cnt/overlap_across)
    down <- ceiling(down_cnt/overlap_down)
    across_add <- (across*overlap_across)-across_cnt 
    across_seq <- seq(0, across-1, by=1)
    down_seq <- seq(0, down-1, by=1)
    across_seq2 <- (across_seq*overlap_across)+1
    down_seq2 <- (down_seq*overlap_down)+1
    
    #Loop through row/column combinations to make predictions for entire image 
    for (c in across_seq2){
      for (r in down_seq2){
        c1 <- c
        r1 <- r
        c2 <- c + (stride_x-1)
        r2 <- r + (stride_y-1)
        if(c2 <= across_cnt && r2 <= down_cnt){ #Full chip
          chip_data <- topo1[r1:r2, c1:c2, 1:3]
          mask_data <- mask1[r1:r2, c1:c2, 1]
        }else if(c2 > across_cnt && r2 <= down_cnt){ # Last column
          c1b <- across_cnt - (size-1)
          c2b <- across_cnt
          chip_data <- topo1[r1:r2, c1b:c2b, 1:3]
          mask_data <- mask1[r1:r2, c1b:c2b, 1]
        }else if(c2 <= across_cnt && r2 > down_cnt){ #Last row
          r1b <- down_cnt - (size-1)
          r2b <- down_cnt
          chip_data <- topo1[r1b:r2b, c1:c2, 1:3]
          mask_data <- mask1[r1b:r2b, c1:c2, 1]
        }else{ # Last row, last column
          c1b <- across_cnt - (size -1)
          c2b <- across_cnt
          r1b <- down_cnt - (size -1)
          r2b <- down_cnt
          chip_data <- topo1[r1b:r2b, c1b:c2b, 1:3]
          mask_data <- mask1[r1b:r2b, c1b:c2b, 1]
        }
        names(chip_data) <- c("R", "G", "B")
        R <- as.vector(chip_data$R)
        G <- as.vector(chip_data$G)
        B <- as.vector(chip_data$B)
        chip_data2 <- c(R, G, B)
        chip_array <- array(chip_data2, c(256,256,3))
        img1 <- as.cimg(chip_array)
        imager::save.image(img1, paste0(outDir, "/images/", substr(fName, 1, nchar(image)-4), "_", c1, "_", r1, ".png"))
        names(mask_data) <- c("C")
        Cx <- as.vector(mask_data$C)
        mask_array <- array(Cx, c(256,256,1))
        msk1 <- as.cimg(mask_array)
        imager::save.image(msk1, paste0(outDir, "/masks/", substr(fName, 1, nchar(image)-4), "_", c1, "_", r1, ".png"))
      }
    }
  }else if(mode == "Positive"){
    topo1 <- rast(image)
    mask1 <- rast(mask)
    
    fName = basename(image)
    
    if (file.exists(paste0(outDir, "/images"))){
      print("Using Existing Folders")
    } else {
      dir.create(paste0(outDir, "/images"))
      dir.create(paste0(outDir, "/masks"))
    }
    
    across_cnt = ncol(topo1)
    down_cnt = nrow(topo1)
    tile_size_across = size
    tile_size_down = size
    overlap_across = stride_x
    overlap_down = stride_y
    across <- ceiling(across_cnt/overlap_across)
    down <- ceiling(down_cnt/overlap_down)
    across_add <- (across*overlap_across)-across_cnt 
    across_seq <- seq(0, across-1, by=1)
    down_seq <- seq(0, down-1, by=1)
    across_seq2 <- (across_seq*overlap_across)+1
    down_seq2 <- (down_seq*overlap_down)+1
    
    #Loop through row/column combinations to make predictions for entire image 
    for (c in across_seq2){
      for (r in down_seq2){
        c1 <- c
        r1 <- r
        c2 <- c + (stride_x-1)
        r2 <- r + (stride_y-1)
        if(c2 <= across_cnt && r2 <= down_cnt){ #Full chip
          chip_data <- topo1[r1:r2, c1:c2, 1:3]
          mask_data <- mask1[r1:r2, c1:c2, 1]
        }else if(c2 > across_cnt && r2 <= down_cnt){ # Last column
          c1b <- across_cnt - (size-1)
          c2b <- across_cnt
          chip_data <- topo1[r1:r2, c1b:c2b, 1:3]
          mask_data <- mask1[r1:r2, c1b:c2b, 1]
        }else if(c2 <= across_cnt && r2 > down_cnt){ #Last row
          r1b <- down_cnt - (size-1)
          r2b <- down_cnt
          chip_data <- topo1[r1b:r2b, c1:c2, 1:3]
          mask_data <- mask1[r1b:r2b, c1:c2, 1]
        }else{ # Last row, last column
          c1b <- across_cnt - (size -1)
          c2b <- across_cnt
          r1b <- down_cnt - (size -1)
          r2b <- down_cnt
          chip_data <- topo1[r1b:r2b, c1b:c2b, 1:3]
          mask_data <- mask1[r1b:r2b, c1b:c2b, 1]
        }
        names(chip_data) <- c("R", "G", "B")
        R <- as.vector(chip_data$R)
        G <- as.vector(chip_data$G)
        B <- as.vector(chip_data$B)
        chip_data2 <- c(R, G, B)
        chip_array <- array(chip_data2, c(256,256,3))
        img1 <- as.cimg(chip_array)
        names(mask_data) <- c("C")
        Cx <- as.vector(mask_data$C)
        mask_array <- array(Cx, c(256,256,1))
        msk1 <- as.cimg(mask_array)
        if(max(mask_array) > 0){
          imager::save.image(img1, paste0(outDir, "/images/", substr(fName, 1, nchar(image)-4), "_", c1, "_", r1, ".png"))
          imager::save.image(msk1, paste0(outDir, "/masks/", substr(fName, 1, nchar(image)-4), "_", c1, "_", r1, ".png"))
        }
      }
    }
  }else{
    topo1 <- rast(image)
    mask1 <- rast(mask)
    
    fName = basename(image)
    
    if (file.exists(paste0(outDir, "/images"))){
      print("Using Existing Folders")
    } else {
      dir.create(paste0(outDir, "/images"))
      dir.create(paste0(outDir, "/masks"))
      
      dir.create(paste0(outDir, "/images/positive"))
      dir.create(paste0(outDir, "/images/background"))
      dir.create(paste0(outDir, "/masks/positive"))
      dir.create(paste0(outDir, "/masks/background"))
    }
    
    across_cnt = ncol(topo1)
    down_cnt = nrow(topo1)
    tile_size_across = size
    tile_size_down = size
    overlap_across = stride_x
    overlap_down = stride_y
    across <- ceiling(across_cnt/overlap_across)
    down <- ceiling(down_cnt/overlap_down)
    across_add <- (across*overlap_across)-across_cnt 
    across_seq <- seq(0, across-1, by=1)
    down_seq <- seq(0, down-1, by=1)
    across_seq2 <- (across_seq*overlap_across)+1
    down_seq2 <- (down_seq*overlap_down)+1
    
    #Loop through row/column combinations to make predictions for entire image 
    for (c in across_seq2){
      for (r in down_seq2){
        c1 <- c
        r1 <- r
        c2 <- c + (stride_x-1)
        r2 <- r + (stride_y-1)
        if(c2 <= across_cnt && r2 <= down_cnt){ #Full chip
          chip_data <- topo1[r1:r2, c1:c2, 1:3]
          mask_data <- mask1[r1:r2, c1:c2, 1]
        }else if(c2 > across_cnt && r2 <= down_cnt){ # Last column
          c1b <- across_cnt - (size-1)
          c2b <- across_cnt
          chip_data <- topo1[r1:r2, c1b:c2b, 1:3]
          mask_data <- mask1[r1:r2, c1b:c2b, 1]
        }else if(c2 <= across_cnt && r2 > down_cnt){ #Last row
          r1b <- down_cnt - (size-1)
          r2b <- down_cnt
          chip_data <- topo1[r1b:r2b, c1:c2, 1:3]
          mask_data <- mask1[r1b:r2b, c1:c2, 1]
        }else{ # Last row, last column
          c1b <- across_cnt - (size -1)
          c2b <- across_cnt
          r1b <- down_cnt - (size -1)
          r2b <- down_cnt
          chip_data <- topo1[r1b:r2b, c1b:c2b, 1:3]
          mask_data <- mask1[r1b:r2b, c1b:c2b, 1]
        }
        names(chip_data) <- c("R", "G", "B")
        R <- as.vector(chip_data$R)
        G <- as.vector(chip_data$G)
        B <- as.vector(chip_data$B)
        chip_data2 <- c(R, G, B)
        chip_array <- array(chip_data2, c(256,256,3))
        img1 <- as.cimg(chip_array)
        names(mask_data) <- c("C")
        Cx <- as.vector(mask_data$C)
        mask_array <- array(Cx, c(256,256,1))
        msk1 <- as.cimg(mask_array)
        if(max(mask_array) > 0){
          imager::save.image(img1, paste0(outDir, "/images/positive/", substr(fName, 1, nchar(image)-4), "_", c1, "_", r1, ".png"))
          imager::save.image(msk1, paste0(outDir, "/masks/positive/", substr(fName, 1, nchar(image)-4), "_", c1, "_", r1, ".png"))
        }else{
          imager::save.image(img1, paste0(outDir, "/images/background/", substr(fName, 1, nchar(image)-4), "_", c1, "_", r1, ".png"))
          imager::save.image(msk1, paste0(outDir, "/masks/background/", substr(fName, 1, nchar(image)-4), "_", c1, "_", r1, ".png"))
        }
      }
    }
  }
}


for(t in 1:nrow(austin)){
  chipIt(image= austin[t, c("img_full")], 
         mask= austin[t, c("msk_full")],
         size=256, stride_x=256, stride_y=256, 
         outDir= "C:/Maxwell_Data/chip_inria/austin", 
         mode="All")
}

for(t in 1:nrow(chicago)){
  chipIt(image= chicago[t, c("img_full")], 
         mask= chicago[t, c("msk_full")],
         size=256, stride_x=256, stride_y=256, 
         outDir= "C:/Maxwell_Data/chip_inria/chicago", 
         mode="All")
}

for(t in 1:nrow(kitsap)){
  chipIt(image= kitsap[t, c("img_full")], 
         mask= kitsap[t, c("msk_full")],
         size=256, stride_x=256, stride_y=256, 
         outDir= "C:/Maxwell_Data/chip_inria/kitsap", 
         mode="All")
}

for(t in 1:nrow(tyrol)){
  chipIt(image= tyrol[t, c("img_full")], 
         mask= tyrol[t, c("msk_full")], 
         size=256, stride_x=256, stride_y=256, 
         outDir= "C:/Maxwell_Data/chip_inria/tyrol", 
         mode="All")
}

for(t in 1:nrow(vienna)){
  chipIt(image= vienna[t, c("img_full")], 
         mask= vienna[t, c("msk_full")],
         size=256, stride_x=256, stride_y=256, 
         outDir= "C:/Maxwell_Data/chip_inria/vienna", 
         mode="All")
}
