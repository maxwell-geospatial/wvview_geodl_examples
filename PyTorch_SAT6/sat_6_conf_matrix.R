library(dplyr)
library(caret)

# Read in results CSV
result <- read.csv("C:/Maxwell_Data/archive/chips2/test_result4.csv")

# Set reference and predicted columns to factors
result$class <- as.factor(result$class)
result$predicted <- as.factor(result$predicted)

# Use caret to create confusion matrix
cm <- confusionMatrix(data=result$predicted, reference=result$class, mode="everything")

# Print confusion matrix
cm
