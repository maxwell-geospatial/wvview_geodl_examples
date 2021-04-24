chipIt <- function(image, size, stride_x, sride_y, inDirImg, inDirMask, outDirMask, outDirImg){
  
  topo1 <- rast(paste0(inDirImg, image))
  mask1 <- rast(paste0(inDirMask, image))

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
      imager::save.image(img1, paste0(outDirImg, substr(image, 1, nchar(image)-4), "_", c1, "_", r1, ".png"))
      names(mask_data) <- c("C")
      Cx <- as.vector(mask_data$C)
      mask_array <- array(Cx, c(256,256,1))
      msk1 <- as.cimg(mask_array)
      imager::save.image(msk1, paste0(outDirMask, substr(image, 1, nchar(image)-4), "_", c1, "_", r1, ".png"))
    }
  }
}


chipIt(image, size, stride_x, sride_y, inDirImg, inDirMask, outDirMask, outDirImg)


chipIt <- function(image, size, stride_x, sride_y, inDir, outDirImg, mode){
  
  topo1 <- rast(paste0(inDirImg, image))
  mask1 <- rast(paste0(inDirMask, image))
  
  across_cnt = ncol(topo1)
  down_cnt = nrow(topo1)
  tile_size_across = size
  tile_size_down = size
  overlap_across = stride_x
  overlap_down = stride_y
  across <- ceiling(across_cnt/overlap_across)
  down <- ceiling(down_cnt/overlap_down)
  across_add <- (across*overlap_across)-across_cnt 
  across_seq <- seq(0, across-2, by=1)
  down_seq <- seq(0, down-2, by=1)
  across_seq2 <- (across_seq*overlap_across)+1
  down_seq2 <- (down_seq*overlap_down)+1
  
  locate <- c()
  
  for (c in across_seq2){
    for (r in down_seq2){
      c1 <- c
      r1 <- r
      c2 <- c + (stride_x-1)
      r2 <- r + (stride_y-1)
      if(c2 <= across_cnt && r2 <= down_cnt){ #Full chip
        locate <- c(locate, paste0(c1, "_", r2))
      }else if(c2 > across_cnt && r2 <= down_cnt){ # Last column
        locate <- c(locate, paste0(c1, "_", r2))
      }else if(c2 <= across_cnt && r2 > down_cnt){ #Last row
        locate <- c(locate, paste0(c1, "_", r2))
      }else{ # Last row, last column
        locate <- c(locate, "Last Column/Row")
      }
    }
  }
  return(locate)
}


ctest <- chipIt(image, mask, size, stride_x, sride_y, inDirImg, inDirMask, outDirMask, outDirImg)





chipIt <- function(image, mask, size=256, stride_x=256, sride_y=256, outDir, mode="All"){
  if(mode == "All"){
    topo1 <- rast(image)
    mask1 <- rast(mask)
    
    fName = basename(image)
    
    dir.create(paste0(outDir, "/images"))
    dir.create(paste0(outDir, "/masks"))
    
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
    
    dir.create(paste0(outDir, "/images"))
    dir.create(paste0(outDir, "/masks"))
    
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
    
    dir.create(paste0(outDir, "/images"))
    dir.create(paste0(outDir, "/masks"))
    
    dir.create(paste0(outDir, "/images/positive"))
    dir.create(paste0(outDir, "/images/background"))
    dir.create(paste0(outDir, "/masks/positive"))
    dir.create(paste0(outDir, "/masks/background"))
    
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


chipIt(image= "C:/Maxwell_Data/topo_data/topoDL/topos/KY_Adams_708051_1971_24000_geo.tif", 
                   mask="C:/Maxwell_Data/topo_data/topoDL/masks/KY_Adams_708051_1971_24000_geo.tif",
                   size=256, stride_x=256, sride_y=256, 
                   outDir= "C:/Maxwell_Data/chip_test", 
                   mode="Divided")