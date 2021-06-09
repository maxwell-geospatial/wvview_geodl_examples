library(dplyr)
library(stringr)
library(fasterize)
library(terra)
library(sf)

topos_preds <- list.files("C:/Maxwell_Data/Dropbox/predictions3/", pattern="*.tif")
topos_ref_path <- "C:/Maxwell_Data/topo_data/topo_dl_data/topo_dl_data/all_mines"

truth <- paste0(topos_ref_path, "/", substr(topos_preds[1], 1, nchar(topos_preds[1])-4), ".shp")
predicted <- paste0("C:/Maxwell_Data/Dropbox/predictions3/", topos_preds[1])

Positive_Case <- 1
Background <- 0
threshold <- .5


#Generate blank data frame to store assessment metrics
evalBinary <- function(reference, predicted, crop=0, truth_dtype="Vector", Positive_Case = 1, Background = 0){
  metrics_df <- data.frame(set=character(), 
                           quad=character(), 
                           tp=numeric(), 
                           tn=numeric(),
                           fp=numeric(), 
                           fn=numeric(),
                           acc=numeric(), 
                           recall=numeric(), 
                           precision=numeric(), 
                           f1=numeric(), 
                           specificity=numeric(), 
                           npv=numeric())
  if(truth_dtype =="Vector"){
    predG <- rast(predicted) > threshold
    blankG <- predG
    blankG[] <- NA
    refV <- st_read(reference)
    refV$code <- Positive_Case
    ref2 <- vect(refV)
    refG <- rasterize(ref2, blankG, field="code", background=Background)
  }else{
    predG <- rast(predicted)
    refG <- rast(reference)
  }
  refGb <- (refG+1)*10
  comp1 <- predG+refGb
  comp2 <- comp1[crop:nrow(comp1)-crop, crop:ncol(comp1)-crop]
  names(comp2) <- "cells"
  table1 <- comp2 %>% group_by(as.factor(cells)) %>% count()
  tp <- table1[4,2]
  tn <- table1[1,2]
  fn <- table1[3,2]
  fp <- table1[2,2]
  acc <- (tp+tn)/(tp+tn+fp+fn)
  precision <- tp/(tp+fp)
  recall <- tp/(tp+fn)
  specificity <- tn/(tn+fp)
  npv <- tn/(tn+fn)
  f1 <- (2*precision*recall)/(precision+recall)
  
  case_metrics <- data.frame(tp = tp,
                             tn = tn,
                             fn = fn,
                             fp = fp,
                             acc = acc,
                             f1 = f1,
                             precision = precision,
                             recall = recall, 
                             specificity = specificity,
                             npv = npv)
  names(case_metrics) <- c("TP", "TN", "FN", "FP", "OA", "F1", "Precision", "Recall", "Specificity", "NPV")
  return(case_metrics)
}

func_test <- evalBinary(reference=truth, predicted=predicted, crop=128, truth_dtype="Vector", Positive_Case = 1, Background = 0)

case_metrics2 <- data.frame(name = character(),
                           tp = numeric(),
                           tn = numeric(),
                           fn = numeric(),
                           fp = numeric(),
                           acc = numeric(),
                           f1 = numeric(),
                           precision = numeric(),
                           recall = numeric(),
                           specificity = numeric(),
                           npv = numeric())

for(t in 1:length(topos_preds)){
  truth <- paste0(topos_ref_path, "/", substr(topos_preds[t], 1, nchar(topos_preds[t])-4), ".shp")
  predicted <- paste0("C:/Maxwell_Data/Dropbox/predictions3/", topos_preds[t])
  topo_name <- t[t]
  
  Positive_Case <- 1
  Background <- 0
  threshold <- .5
  
  met_out <- evalBinary(reference=truth, predicted=predicted, crop=128, truth_dtype="Vector", Positive_Case = Positive_Case, Background = Background)
  
  out_frame <- cbind(topo_name, met_out)
  case_metrics2 <- rbind(case_metrics2, out_frame)
}

write.csv(case_metrics2,  "C:/Maxwell_Data/Dropbox/predictions3/metricsNew.csv")



