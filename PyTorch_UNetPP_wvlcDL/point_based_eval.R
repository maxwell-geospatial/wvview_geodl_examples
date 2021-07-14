# Read in libraries ===========================
library(dplyr)
library(terra)
library(caret)
library(rfUtilities)

# Read in data ===================================
lc <- rast("C:/Users/amaxwel6/Downloads/accuracy_assessment/accuracy_assessment/WV_Spectral_classes_NAIP_2016.tif")
val_pnts <- vect("C:/Users/amaxwel6/Downloads/accuracy_assessment/accuracy_assessment/validation_points.shp")

# Extract classification from raster at validation point locations ====================
lc_ext <- extract(lc, val_pnts, factor=FALSE)
lc_ext2 <- as.factor(as.character(lc_ext[,2]))

# Make data frame with references and predictions
val_df <- as.data.frame(val_pnts)
val_df$predicted <- lc_ext2
val_df$GrndTruth <- as.factor(as.character(val_df$GrndTruth))

# Generate confusion matrix with caret
cfOut <- confusionMatrix(data=val_df$predicted, reference=val_df$GrndTruth)

# Generate assessment metrics with rfUtilities
accOut <- accuracy(val_df$predicted, val_df$GrndTruth)


