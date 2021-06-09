
library(terra)
library(sf)

reference <- "Reference Data"
predicted <- "Predicted Data"
predicted <- rast(predicted) #Load prediction raster
reference <- st_read(reference) #Load reference vector/use rast() for raster

evalClassification <- function(reference, predicted, truth_dtype="Vector", 
                               codes="codes", background = 0, mappings,
                               positive_case=mappings[1]){
  
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
  cm <- caret::confusionMatrix(ctab, mode="everything", positive=positive_case)
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
                        Classes=mappings,
                        PostiveCase=cm$positive)
  return(final_metrics)
}

func_test <- evalClassification(reference=reference, 
                        predicted=predicted, 
                        truth_dtype="Vector", 
                        codes="codes", 
                        background = 0, 
                        mappings=c("Not Mining", "Mining"),
                        positive_case="Mining")

val_set <- read.csv("C:/Maxwell_Data/topo_work/val_topos.csv")


func_test <- evalClassification(reference=rast(val_set[1, c("msk_full")]), 
                                predicted=rast(paste0("C:/Maxwell_Data/topo_work/data/predictions/", val_set[1, c("Images")])),
                                truth_dtype="Raster",
                                codes="codes", 
                                background = 0, 
                                mappings=c("Not Mining", "Mining"),
                                positive_case="Mining")


results_table <- data.frame(name=as.character(), acc=as.numeric(), kap=as.numeric(), f1s=as.numeric(), prec=as.numeric(), recall=as.numeric(), spec=as.numeric(), npv =as.numeric())

for(i in 1:nrow(val_set)){
  met_out <- evalClassification(reference=rast(val_set[i, c("msk_full")]), 
                                  predicted=rast(paste0("C:/Maxwell_Data/topo_work/data/predictions/", val_set[i, c("Images")])),
                                  truth_dtype="Raster",
                                  codes="codes", 
                                  background = 0, 
                                  mappings=c("Not Mining", "Mining"),
                                  positive_case="Mining")
  name <- val_set[i, c("Images")]
  acc <- met_out$overallAcc[1]
  kap <- met_out$overallAcc[2]
  f1s <- met_out$ClassAcc[7]
  prec <- met_out$ClassAcc[5]
  recall <- met_out$ClassAcc[6]
  spec <- met_out$ClassAcc[2]
  npv <- met_out$ClassAcc[4]
  met_out2 <- data.frame(name, acc, kap, f1s, prec, recall, spec, npv)
  results_table <- rbind(results_table, met_out2)
}
