library(terra)
library(sf)

evalClassification <- function(reference, predicted, truth_dtype="Vector", 
                               codes="codes", background = 0, mappings){
  
  require(terra)
  require(caret)
  require(diffeR)
  require(rfUtilities)

  if(truth_dtype =="Vector"){
    predG <- predicted
    ref2 <- vect(reference)
    ref3 <- terra::project(ref2, predG)
    refG <- terra::rasterize(ref3, predG, field=codes, background=background)
  }else{
    predG <- predicted
    refG <- reference
  }
  stacked <- c(predG, refG)
  ctab <- crosstab(stacked, useNA=FALSE)
  colnames(ctab) <- mappings
  rownames(ctab) <- mappings
  dimnames(ctab) <- setNames(dimnames(ctab),c("Predicted", "Reference"))
  cm <- caret::confusionMatrix(ctab, mode="everything")
  rfu <- rfUtilities::accuracy(ctab)
  Error <- overallDiff(ctab)/sum(ctab)
  Allocation <- overallAllocD(ctab)/sum(ctab)
  Quantity <- overallQtyD(ctab)/sum(ctab)
  Exchange <- overallExchangeD(ctab)/sum(ctab)
  Shift <- overallShiftD(ctab)/sum(ctab)
  
  final_metrics <- list(ConfusionMatrix=cm$table, 
                        overallAcc=cm$overall, 
                        ClassAcc=cm$byClass, 
                        UsersAcc = rfu$users.accuracy/100,
                        ProducersAcc = rfu$producers.accuracy/100,
                        Pontius = data.frame(Error, Allocation, Quantity, Exchange, Shift),
                        Classes=mappings)
  return(final_metrics)
}

evalOut = evalClassification(reference=rast("C:/Maxwell_Data/landcover/masks/M-34-56-A-b-1-4.tif"), predicted=rast("C:/Maxwell_Data/landcover/predictions/test.tif"), truth_dtype="Raster", 
                               codes="codes", background = 0, mappings=c("Background", "Building", "Woodland", "Water"))
