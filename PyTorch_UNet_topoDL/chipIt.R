chipIt <- function(image, mask, n_channels=3, size=256, stride_x=256, stride_y=256, outDir, mode="All"){
  require(terra)
  require(imager)
  if(mode == "All"){
    img1 <- rast(image)
    mask1 <- rast(mask)
    
    fName = basename(image)
    
    dir.create(paste0(outDir, "/images"))
    dir.create(paste0(outDir, "/masks"))
    
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
        image1 <- as.cimg(chip_array, x=size, y=size, cc=n_channels)
        imager::save.image(image1, paste0(outDir, "/images/", substr(fName, 1, nchar(image)-4), "_", c1, "_", r1, ".png"))
        names(mask_data) <- c("C")
        Cx <- as.vector(mask_data$C)
        mask_array <- array(Cx, c(size,size,1))
        msk1 <- as.cimg(mask_array, x=size, y=size, cc=1)
        imager::save.image(msk1, paste0(outDir, "/masks/", substr(fName, 1, nchar(image)-4), "_", c1, "_", r1, ".png"))
      }
    }
  }else if(mode == "Positive"){
    img1 <- rast(image)
    mask1 <- rast(mask)
    
    fName = basename(image)
    
    dir.create(paste0(outDir, "/images"))
    dir.create(paste0(outDir, "/masks"))
    
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
        image1 <- as.cimg(chip_array, x=size, y=size, cc=n_channels)
        names(mask_data) <- c("C")
        Cx <- as.vector(mask_data$C)
        mask_array <- array(Cx, c(size,size,1))
        msk1 <- as.cimg(mask_array, x=size, y=size, cc=1)
        if(max(mask_array) > 0){
          imager::save.image(image1, paste0(outDir, "/images/", substr(fName, 1, nchar(image)-4), "_", c1, "_", r1, ".png"))
          imager::save.image(msk1, paste0(outDir, "/masks/", substr(fName, 1, nchar(image)-4), "_", c1, "_", r1, ".png"))
        }
      }
    }
  }else if(mode=="Divided") {
    img1 <- rast(image)
    mask1 <- rast(mask)
    
    fName = basename(image)
    
    dir.create(paste0(outDir, "/images"))
    dir.create(paste0(outDir, "/masks"))
    
    dir.create(paste0(outDir, "/images/positive"))
    dir.create(paste0(outDir, "/images/background"))
    dir.create(paste0(outDir, "/masks/positive"))
    dir.create(paste0(outDir, "/masks/background"))
    
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
        image1 <- as.cimg(chip_array, x=size, y=size, cc=n_channels)
        names(mask_data) <- c("C")
        Cx <- as.vector(mask_data$C)
        mask_array <- array(Cx, c(size,size,1))
        msk1 <- as.cimg(mask_array, x=size, y=size, cc=1)
          if(max(mask_array) > 0){
          imager::save.image(image1, paste0(outDir, "/images/positive/", substr(fName, 1, nchar(image)-4), "_", c1, "_", r1, ".png"))
          imager::save.image(msk1, paste0(outDir, "/masks/positive/", substr(fName, 1, nchar(image)-4), "_", c1, "_", r1, ".png"))
          }else{
          imager::save.image(image1, paste0(outDir, "/images/background/", substr(fName, 1, nchar(image)-4), "_", c1, "_", r1, ".png"))
          imager::save.image(msk1, paste0(outDir, "/masks/background/", substr(fName, 1, nchar(image)-4), "_", c1, "_", r1, ".png"))
        }
      }
    }
  } else {
    print("Invalid Mode Provided.")
  }
}




