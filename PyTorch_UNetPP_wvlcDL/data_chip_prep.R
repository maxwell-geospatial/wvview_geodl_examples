# Read in libraries =====================================
library(dplyr)
library(terra)
library(sf)

# List all quarter quads ==================================
naip_imgs <- list.files(path="F:/NAIP_2016/wv_60cm_2016", pattern = "\\.tif$", recursive = TRUE)
#polys <- vect("C:/Maxwell_Data/wvlcDL_2/training_polys.shp")

# Create subdirectories in mask folder ============================
subDirs <- list.dirs('F:/NAIP_2016/wv_60cm_2016', full.names=FALSE, recursive=FALSE)
for (s in 1:length(subDirs)) {
  dir.create(paste0("C:/Maxwell_Data/wvlcDL_2/masks/", subDirs[s]))
}

# Define mask generation function ===============================  
makeMasks <- function(image, features, field, background, outMask){
  imgData <- rast(image)
  featData <- vect(features)
  extData <- as.polygons(ext(imgData))
  featData2 <- crop(featData, extData)
  if (length(featData2) > 0) {
    mskR <- rasterize(featData2, imgData, field=field, background=background)
    writeRaster(mskR, outMask)
  }
}

# Generate masks for all quarter quads ==========================================
for (i in 1:length(naip_imgs)) {
  makeMasks(image=paste0("F:/NAIP_2016/wv_60cm_2016/", naip_imgs[i]), 
            features="C:/Maxwell_Data/wvlcDL_2/training_polys.shp", 
            field="classvalue", background=0, 
            outMask=paste0("C:/Maxwell_Data/wvlcDL_2/masks/", naip_imgs[i]))
}

# Define chip generation function =================================================
chipIt <- function(image, mask, fname=imgNm, n_channels=3, size=256, stride_x=256, stride_y=256, outDir){
  require(terra)
  require(imager)
  img1 <- image
  mask1 <- mask
    
  fName = basename(imgNm)
    
  across_cnt = ncol(img1)
  down_cnt = nrow(img1)
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
      chip_data <- data.frame()
      mask_data <- data.frame()
      if(c2 <= across_cnt && r2 <= down_cnt){ #Full chip
        chip_data <- img1[r1:r2, c1:c2, 1:n_channels]
        mask_data <- mask1[r1:r2, c1:c2, 1]
      }else if(c2 > across_cnt && r2 <= down_cnt){ # Last column
        c1b <- across_cnt - (size-1)
        c2b <- across_cnt
        chip_data <- img1[r1:r2, c1b:c2b, 1:n_channels]
        mask_data <- mask1[r1:r2, c1b:c2b, 1]
      }else if(c2 <= across_cnt && r2 > down_cnt){ #Last row
        r1b <- down_cnt - (size-1)
        r2b <- down_cnt
        chip_data <- img1[r1b:r2b, c1:c2, 1:n_channels]
        mask_data <- mask1[r1b:r2b, c1:c2, 1]
      }else{ # Last row, last column
        c1b <- across_cnt - (size -1)
        c2b <- across_cnt
        r1b <- down_cnt - (size -1)
        r2b <- down_cnt
        chip_data <- img1[r1b:r2b, c1b:c2b, 1:n_channels]
        mask_data <- mask1[r1b:r2b, c1b:c2b, 1]
      }
      chip_data2 <- c(stack(chip_data)[,1])
      chip_array <- array(chip_data2, c(size,size,n_channels))
      image1 <- rast(chip_array)
      names(mask_data) <- c("C")
      Cx <- as.vector(mask_data$C)
      mask_array <- array(Cx, c(size,size,1))
      msk1 <- rast(mask_array)
      if(max(mask_array) > 0){
        writeRaster(image1, paste0(outDir, "/images/", substr(fName, 1, nchar(fName)-4), "_", c1, "_", r1, ".tif"))
        writeRaster(msk1, paste0(outDir, "/masks/", substr(fName, 1, nchar(fName)-4), "_", c1, "_", r1, ".tif"))
      }
    }
  }
}

# List all masks =================================
naip_msks <- list.files(path="C:/Maxwell_Data/wvlcDL_2/masks", pattern = "\\.tif$", recursive = TRUE)


# Loop to generate chips for quater quads and masks ============================
for (i in 1:length(naip_msks)) {
  imgIn <- paste0("C:/Maxwell_Data/wv_60cm_2016/", naip_msks[i])
  maskIn <- paste0("C:/Maxwell_Data/wvlcDL_2/masks/", naip_msks[i])
  imgInR <- rast(imgIn)
  maskInR <- rast(maskIn)
  imgNm <- imgIn
  chipIt(image=imgInR,mask=maskInR, fname=imgNm,
         n_channels=4, size=512, stride_x=512, stride_y=512, 
         outDir="C:/Maxwell_Data/wvlcDL_2/chips2")
}




