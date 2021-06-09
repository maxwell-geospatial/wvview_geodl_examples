library(dplyr)
library(stringr)
library(terra)
library(imager)

makeMasks <- function(image, features, crop=FALSE, extent, field, background, outImage, outMask, mode="Both"){
  require(terra)
  
  imgData <- rast(image)
  imgData <- clamp(imgData, 0, 254, values=TRUE)
  featData <- vect(features)
  
  featCRS <- project(featData, imgData)
  if (crop==TRUE){
    extData <- vect(extent)
    extCRS <- project(extData, imgData)
    imgData <- crop(imgData, extCRS)
  }
  mskR <- rasterize(featCRS, imgData, field=field, background=background)
  if (mode=="Both") {
    writeRaster(imgData, outImage)
    writeRaster(mskR, outMask)
  } else if (mode=="Mask") {
    writeRaster(mskR, outMask)
  } else{
    print("Invalid Mode.")
  }
}


images <- list.files("C:/Maxwell_Data/topo_work/topo_dl_data/topo_dl_data/ky_topos",
                     full.names= FALSE, 
                     pattern="\\.tif$")

imgDF <- data.frame(img = substr(images, 1, nchar(images)-4))
imgDF$img_path <- paste0("C:/Maxwell_Data/topo_work/topo_dl_data/topo_dl_data/ky_topos/", imgDF$img, ".tif")
imgDF$msk_path <- paste0("C:/Maxwell_Data/topo_work/topo_dl_data/topo_dl_data/ky_mines/", imgDF$img, ".shp")
imgDF$ext_path <- paste0("C:/Maxwell_Data/topo_work/topo_dl_data/topo_dl_data/ky_quads/", imgDF$img, ".shp")
imgDF$img_out <- paste0("C:/Maxwell_Data/topo_work/data/", "img/", imgDF$img, ".tif")
imgDF$msk_out <- paste0("C:/Maxwell_Data/topo_work/data/", "msk/", imgDF$img, ".tif")

for(i in 1:nrow(imgDF)) {
  makeMasks(image=imgDF[i, "img_path"], features=imgDF[i, "msk_path"], 
            crop=TRUE, extent = imgDF[i, "ext_path"], field=254, background=0,
            outImage=imgDF[i, "img_out"], outMask=imgDF[i, "msk_out"], 
            mode="Both")
}


