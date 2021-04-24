makeMasks <- function(image, features, extent, field, background){
  
    imgData <- rast(image)
    extData <- vect(extent)
    featData <- vect(feastures)
    
    extCRS <- project(extData, imgData)
    featCRS <- project(featData, imgData)
    
    img_crop <- crop(imgData, extCRS)
    
    minesCRS$code <- 255
    
    mineR <- rasterize(minesCRS, topo_crop, field=field, background=background)
    
    writeRaster(topo_crop, paste0(out_topo, t))
    writeRaster(mineR, paste0(out_mask, t))
}