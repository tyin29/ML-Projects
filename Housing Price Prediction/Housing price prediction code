#####################  Step 0: Load necessary libraries ##################### 

library(glmnet)
library(xgboost)

##################### Step 1: Preprocess training data  #####################

train <- read.csv("train.csv", stringsAsFactors = FALSE)

missing.n = sapply(names(train), function(x) length(which(is.na(train[, x])))) # check missing values
which(missing.n != 0)  # only the 60th col "Garage_Yr_Blt" has missing values
id = which(is.na(train$Garage_Yr_Blt))
length(id) # number of missing values in total
train$Garage_Yr_Blt[is.na(train$Garage_Yr_Blt)] <- 0 # replace the missing values as 0
missing.n = sapply(names(train), function(x) length(which(is.na(train[, x]))))
id = which(is.na(train$Garage_Yr_Blt))
which(missing.n != 0)  # no missing values now

train.x <- train[, -83]
train.y <- log(train$Sale_Price)

remove.var <- c('Street', 'Utilities', 'Condition_2', 'Roof_Matl', 'Heating', 'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude','Latitude') 
# the variables removed, which include imbalanced categorical variables and non-interpretable predictors (Longitude and Latitude)

### Apply winsorization on some numerical variables
winsor.vars <- c("Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF", "Second_Flr_SF", 'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF", "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val")

quan.value <- 0.95
for(var in winsor.vars){
  tmp <- train.x[, var]
  myquan <- quantile(tmp, probs = quan.value, na.rm = TRUE)
  tmp[tmp > myquan] <- myquan
  train.x[, var] <- tmp
}

train.x = train.x[,!(names(train.x) %in% remove.var)]
categorical.vars <- colnames(train.x)[which(sapply(train.x, function(x) mode(x)=="character"))]
train.matrix <- train.x[, !colnames(train.x) %in% categorical.vars, drop=FALSE]
n.train <- nrow(train.matrix)

var_level = list()
for(var in categorical.vars){
  mylevels <- sort(unique(train.x[, var]))
  var_level[[var]] = c(mylevels) ### save the variable levels
  m <- length(mylevels)
  m <- ifelse(m>2, m, 1)  ### generate m binary categorical variables when m > 2
  tmp.train <- matrix(0, n.train, m)
  col.names <- NULL
  
  for(j in 1:m){
    tmp.train[train.x[, var]==mylevels[j], j] <- 1
    col.names <- c(col.names, paste(var, '_', mylevels[j], sep=''))
  }
  colnames(tmp.train) <- col.names
  train.matrix <- cbind(train.matrix, tmp.train)
}

###### linear model linear regression
set.seed(9569)
cv.out <- cv.glmnet(as.matrix(train.matrix[-1]), train.y, alpha = 1)
sel.vars <- predict(cv.out, type="nonzero", s = cv.out$lambda.min)$s1
cv.out <- cv.glmnet(as.matrix(train.matrix[-1][, sel.vars]), train.y, alpha = 0)

###### boosting tree
set.seed(9569)
xgb.model <- xgboost(data = as.matrix(train.matrix[-1]), label = train.y, max_depth = 6,
                     eta = 0.05, nrounds = 5000, subsample = 0.5, verbose = FALSE)

##################### Step 2: Preprocess test data ############################

test <- read.csv("test.csv", stringsAsFactors = FALSE)

missing.n = sapply(names(test), function(x) length(which(is.na(test[, x])))) # check missing values
which(missing.n != 0)  # only the 60th col "Garage_Yr_Blt" has missing values
id = which(is.na(test$Garage_Yr_Blt))
length(id) # number of missing values in total
test$Garage_Yr_Blt[is.na(test$Garage_Yr_Blt)] <- 0 # replace the missing values as 0
missing.n = sapply(names(test), function(x) length(which(is.na(test[, x]))))
which(missing.n != 0)  
id = which(is.na(test$Garage_Yr_Blt))
length(id) # no missing values now

test.x <- test

quan.value <- 0.95
for(var in winsor.vars){
  tmp <- test.x[, var]
  myquan <- quantile(tmp, probs = quan.value, na.rm = TRUE)
  tmp[tmp > myquan] <- myquan
  test.x[, var] <- tmp
}

test.x = test.x[,!(names(test.x) %in% remove.var)]
categorical.vars <- colnames(test.x)[which(sapply(test.x, function(x) mode(x)=="character"))]
test.matrix <- test.x[, !colnames(test.x) %in% categorical.vars, drop=FALSE]
n.test <- nrow(test.matrix)

for(var in categorical.vars){
  mylevels <- var_level[[var]] ### use the variable levels saved
  m <- length(mylevels)
  m <- ifelse(m>2, m, 1)
  tmp.test <- matrix(0, n.test, m)
  col.names <- NULL
  
  for(j in 1:m){
    tmp.test[test.x[, var]==mylevels[j], j] <- 1
    col.names <- c(col.names, paste(var, '_', mylevels[j], sep=''))
  }
  colnames(tmp.test) <- col.names
  test.matrix <- cbind(test.matrix, tmp.test)
}

##################### Prediction and Write Files #####################

options(digits = 7) # set all digits to 7 decimals

####### linear regression
pred.y_linear <- predict(cv.out, s = cv.out$lambda.min, newx = as.matrix(test.matrix[-1][, sel.vars]))
pred.y_linear <- cbind(test.x[1], exp(pred.y_linear))
colnames(pred.y_linear)[2] <- "Sale_Price"
write.table(pred.y_linear, file = "mysubmission1.txt", append = FALSE, sep = ",", dec = ".", row.names = FALSE,col.names = TRUE, quote = FALSE)

####### boosting tree
pred.y_tree <- as.data.frame(predict(xgb.model, as.matrix(test.matrix[-1])))
pred.y_tree <- cbind(test.x[1], exp(pred.y_tree))
colnames(pred.y_tree)[2] <- "Sale_Price"
write.table(pred.y_tree, file = "mysubmission2.txt", append = FALSE, sep = ",", dec = ".", row.names = FALSE,col.names = TRUE, quote = FALSE)

