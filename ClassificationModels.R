library(caret)
library(FactoMineR)
library(mice)
library(Hmisc)
library(factoextra)
library(gridExtra)
library(grid)
library(ggpubr)
library(MASS)
library(ggplot2)
library(magrittr)
library(VIM)
library(knitr)
library(pROC)

#setwd("/Users/sasi/Desktop/Final Project/APS-Failure/dataset")

# Read Data
df_train<- read.csv("aps_failure_training_set.csv",
                    header=T, na.strings=c("na","NA"), stringsAsFactors = FALSE, strip.white = TRUE)

df_test <- read.csv("aps_failure_test_set.csv",
                    header=T, na.strings=c("na","NA"), stringsAsFactors = FALSE, strip.white = TRUE)

df_train$class[df_train$class == "neg"] <- 0
df_train$class[df_train$class == "pos"] <- 1

df_test$class[df_test$class == "neg"] <- 0
df_test$class[df_test$class == "pos"] <- 1


# check and count total missing values
sum(is.na(df_train)) # 850015
sum(is.na(df_test)) # 228680

# check data str
str(df_train)

# create a copy
df<- df_train

# Visualize the imbalanced data
df %>%
  dplyr::filter(!is.na(class)) %>%
  ggplot(aes(x=class))+
  geom_bar()+
  ggtitle("Imbalanced class distribution")+
  labs(x="class", y="failure count")+
  theme_bw()+
  theme(panel.border = element_rect(colour = "black", fill=NA, size=1))

# Data cleaning and save the class var
x<- df_train$class
y<- df_test$class

# Dimensional reduction steps for training data
df<- df_train

# Check for variables with more than 75% data is missing
miss_cols<- lapply(df, function(col){sum(is.na(col))/length(col)})
df<- df[, !(names(df) %in% names(miss_cols[lapply(miss_cols, function(x) x) > 0.75]))]  # 6 cols with more than 75% missing data

# Check for variables with more than 80% values are zero
zero_cols<- lapply(df, function(col){length(which(col==0))/length(col)})

# Zero_cols<- as.data.frame(zero_cols)
df<- df[, !(names(df) %in% names(zero_cols[lapply(zero_cols, function(x) x) > 0.8]))]

# As all independent variables are continuous in nature, so check for variables where the standard deviation is zero
# Remove columns where the standard derivation is zero
std_zero_col <- lapply(df, function(col){sd(col, na.rm = TRUE)})
df <- df[, !(names(std_zero_col) %in% names(std_zero_col[lapply(std_zero_col, function(x) x) == 0]))] # only 1 variable with std dev is zero

# Check for near zero variance cols
badCols<- nearZeroVar(df)
names(df[, badCols]) # 9 cols in aps_train data with near zero variance property. removing them
df<- df[, -badCols]

# Missing data visualization
aggr(df, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, 
     labels=names(df), cex.axis=.7, gap=3, 
     ylab=c("Histogram of missing data","Pattern"))

# Missing data imputation
imputed<- impute(df,imputation=1, list.class=TRUE, pr=FALSE, check=FALSE)

# convert the list to dataframe
imputed.data <- as.data.frame(do.call(cbind,imputed))

# arrange the columns accordingly
imputed.data <- imputed.data[, colnames(df), drop = FALSE]

# check for missing data
sum(is.na(imputed.data))
names(imputed.data)

# PCA correlation treatment 
df_pca<-PCA(imputed.data, scale.unit = TRUE, ncp = 5,  graph = FALSE)

#Scree plot to visualize the PCA's
screeplot<-fviz_screeplot(df_pca, addlabels = TRUE,
                          barfill = "gray", barcolor = "black",
                          ylim = c(0, 50), xlab = "Principal Component (PC) for continuous variables", ylab = "Percentage of explained variance",
                          main = "(A) Scree plot: Factors affecting APS ",
                          ggtheme = theme_minimal()
)

# Determine Variable contributions to the principal axes
# Contributions of variables to PC1
pc1<-fviz_contrib(df_pca, choice = "var", 
                  axes = 1, top = 10, sort.val = c("desc"),
                  ggtheme= theme_minimal())+
  labs(title="(B) PC-1")

# Contributions of variables to PC2
pc2<-fviz_contrib(df_pca, choice = "var", axes = 2, top = 10,
                  sort.val = c("desc"),
                  ggtheme = theme_minimal())+
  labs(title="(C) PC-2")

fig1<- grid.arrange(arrangeGrob(screeplot), 
                    arrangeGrob(pc1, ncol=1), ncol=2, widths=c(2,1)) 
annotate_figure(fig1
                ,top = text_grob("Principal Component Analysis (PCA): training data", color = "black", face = "bold", size = 14)
                ,bottom = text_grob("Data source: \n APS\n", color = "brown",
                                    hjust = 1, x = 1, face = "italic", size = 10)
)

# Add a black border around the 2x2 grid plot
grid.rect(width = 1.00, height = 0.99, 
          gp = gpar(lwd = 2, col = "black", fill=NA))
grid.newpage()


# Extract the Principal Components
# select PC1 components only
eig.val<- get_eigenvalue(df_pca)

# A simple method to extract the results, for variables, from a PCA classput is to use the function get_pca_var() [factoextra package].
imp_vars<-factoextra::fviz_contrib(df_pca, choice = "var", 
                                   axes = c(1), top = 10, sort.val = c("desc"))

#save data from contribution plot
dat<-imp_vars$data

#filter class ID's that are higher than 1

r<-rownames(dat[dat$contrib>1,])

# extract these from your original data frame into a new data frame for further analysis
df_imputed_impvars<-imputed.data[r] # 50 variables showing maximum variance
df_imputed_impvars$class<- x
train_data_clean<- df_imputed_impvars
train_data_clean$class<- as.factor(train_data_clean$class)
str(train_data_clean)

# Select numeric columns
numeric_cols <- sapply(train_data_clean, is.numeric)
train_data_numeric <- train_data_clean[, numeric_cols]

# Normalize numeric columns
train_data_normalized <- as.data.frame(scale(train_data_numeric))
train_data_normalized <- cbind(train_data_normalized, class = train_data_clean$class)
View(train_data_normalized)


# PREDICTIVE MODELLING ON IMBALANCED TRAINING DATA
# Run algorithms using 3-fold cross validation
set.seed(3)
index <- createDataPartition(train_data_normalized$class, p = 0.75, list = FALSE, times = 1)
train_data <- train_data_normalized[index, ]
test_data  <- train_data_normalized[-index, ]

calculate_metrics <- function(cm) {
  TP <- cm$table[2, 2]
  FP <- cm$table[1, 2]
  TN <- cm$table[1, 1]
  FN <- cm$table[2, 1]
  
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  F1 <- 2 * precision * recall / (precision + recall)
  
  return(list(precision = precision, recall = recall, F1 = F1))
}


# Logistic Regression
glm_model <- glm(class ~ ., data = train_data, family = "binomial")
glm_pred <- predict(glm_model, newdata = test_data, type = "response")
glm_pred_factor <- factor(ifelse(glm_pred > 0.5, "1", "0"), levels = levels(factor(test_data$class)))
glm_cm <- confusionMatrix(data = glm_pred_factor, reference = factor(test_data$class))
glm_roc <- roc(test_data$class, glm_pred)
glm_auc <- auc(glm_roc)

glm_metrics <- calculate_metrics(glm_cm)
glm_precision <- glm_metrics$precision
glm_recall <- glm_metrics$recall
glm_F1 <- glm_metrics$F1
glm_cost <- 500*glm_cm$table[1,2] + 10*glm_cm$table[2,1]


# Random Forest
rf_model <- randomForest::randomForest(class ~ ., data = train_data, ntree = 500, importance = TRUE)
rf_pred <- predict(rf_model, newdata = test_data)
rf_cm <- confusionMatrix(data = rf_pred, reference = test_data$class)
rf_roc <- roc(test_data$class, as.numeric(rf_pred))
rf_auc <- auc(rf_roc)

rf_metrics   <- calculate_metrics(rf_cm)
rf_precision <- rf_metrics$precision
rf_recall    <- rf_metrics$recall
rf_F1        <- rf_metrics$F1
rf_cost <- 500*rf_cm$table[1,2] + 10*rf_cm$table[2,1]

# KNN
library(class)
k <- 5
knn_model <- knn(train = train_data[-1], test = test_data[-1], cl = train_data$class, k = k)
knn_cm <- confusionMatrix(knn_model, test_data$class)
knn_roc <- roc(test_data$class, as.numeric(knn_model))
knn_auc <- auc(knn_roc)

knn_metrics   <- calculate_metrics(knn_cm)
knn_precision <- knn_metrics$precision
knn_recall    <- knn_metrics$recall
knn_F1        <- knn_metrics$F1
knn_cost <- 500*knn_cm$table[1,2] + 10*knn_cm$table[2,1]

# Classification Trees
ct_model <- rpart::rpart(class ~ ., data = train_data, method = "class")
ct_pred <- predict(ct_model, newdata = test_data, type = "class")
ct_cm <- confusionMatrix(data = ct_pred, reference = test_data$class)
ct_roc <- roc(test_data$class, as.numeric(ct_pred))
ct_auc <- auc(ct_roc)

ct_metrics   <- calculate_metrics(ct_cm)
ct_precision <- ct_metrics$precision
ct_recall    <- ct_metrics$recall
ct_F1        <- ct_metrics$F1
ct_cost <- 500*ct_cm$table[1,2] + 10*ct_cm$table[2,1]

# Support Vector Machines
library(e1071)
svm_model <- svm(class ~ ., data = train_data, type = "C-classification", kernel = "linear")
svm_pred <- predict(svm_model, test_data)
svm_cm <- confusionMatrix(data = svm_pred, reference = test_data$class)
svm_roc <- roc(test_data$class, as.numeric(svm_pred))
svm_auc <- auc(svm_roc)

svm_metrics   <- calculate_metrics(svm_cm)
svm_precision <- svm_metrics$precision
svm_recall    <- svm_metrics$recall
svm_F1        <- svm_metrics$F1
svm_cost <- 500*svm_cm$table[1,2] + 10*svm_cm$table[2,1]

# Decision Tree
dt_model <- rpart::rpart(class ~ ., data = train_data)
dt_pred <- predict(dt_model, newdata = test_data, type = "class")
dt_cm <- confusionMatrix(data = dt_pred, reference = test_data$class)
dt_roc <- roc(test_data$class, as.numeric(dt_pred))
dt_auc <- auc(dt_roc)

dt_metrics   <- calculate_metrics(dt_cm)
dt_precision <- dt_metrics$precision
dt_recall    <- dt_metrics$recall
dt_F1        <- dt_metrics$F1
dt_cost <- 500*dt_cm$table[1,2] + 10*dt_cm$table[2,1]

# Bagging with Random Forest
bag_rf_model <- randomForest::randomForest(class ~ ., data = train_data, ntree = 500, importance = TRUE, mtry = ncol(train_data)-1, replace = TRUE, bag=TRUE)
bag_rf_pred <- predict(bag_rf_model, newdata = test_data)
bag_rf_cm <- confusionMatrix(data = bag_rf_pred, reference = test_data$class)
bag_rf_roc <- roc(test_data$class, as.numeric(bag_rf_pred))
bag_rf_auc <- auc(bag_rf_roc)

bag_rf_metrics   <- calculate_metrics(bag_rf_cm)
bag_rf_precision <- bag_rf_metrics$precision
bag_rf_recall    <- bag_rf_metrics$recall
bag_rf_F1        <- bag_rf_metrics$F1
bag_rf_cost <- 500*bag_rf_cm$table[1,2] + 10*bag_rf_cm$table[2,1]

# Boosting with Random Forest
boost_rf_model <- randomForest::randomForest(class ~ ., data = train_data, ntree = 500, mtry = sqrt(ncol(train_data)-1), importance = TRUE,  do.trace = 100, classwt = c(1, 10),replace = TRUE)
boost_rf_pred <- predict(boost_rf_model, newdata = test_data)
boost_rf_cm <- confusionMatrix(data = boost_rf_pred, reference = test_data$class)
boost_rf_roc <- roc(test_data$class, as.numeric(boost_rf_pred))
boost_rf_auc <- auc(boost_rf_roc)

boost_rf_metrics   <- calculate_metrics(boost_rf_cm)
boost_rf_precision <- boost_rf_metrics$precision
boost_rf_recall    <- boost_rf_metrics$recall
boost_rf_F1        <- boost_rf_metrics$F1
boost_rf_cost <- 500*boost_rf_cm$table[1,2] + 10*boost_rf_cm$table[2,1]

# LDA
library(MASS)
lda_model <- lda(class ~ ., data = train_data)
lda_pred <- predict(lda_model, newdata = test_data)
lda_cm <- confusionMatrix(data = lda_pred$class, reference = test_data$class)
lda_roc <- roc(test_data$class, lda_pred$posterior[, 2])
lda_auc <- auc(lda_roc)

lda_metrics   <- calculate_metrics(lda_cm)
lda_precision <- lda_metrics$precision
lda_recall    <- lda_metrics$recall
lda_F1        <- lda_metrics$F1
lda_cost <- 500*lda_cm$table[1,2] + 10*lda_cm$table[2,1]

# Create a data frame of compare model performances
modelsss <- data.frame(
  model = c("Random Forest", "Support Vector Machines", "Decision Tree", "Bagging", "Boosting","Linear Discriminant Analysis"),
  AUC = c(rf_auc, svm_auc, dt_auc, bag_rf_auc, boost_rf_auc, lda_auc),
  Accuracy = c(rf_cm$overall['Accuracy'], svm_cm$overall['Accuracy'], dt_cm$overall['Accuracy'], bag_rf_cm$overall['Accuracy'], boost_rf_cm$overall['Accuracy'], lda_cm$overall['Accuracy']),
  Precision = c(rf_precision, svm_precision, dt_precision,bag_rf_precision, boost_rf_precision,lda_precision),
  Recall = c(rf_recall, svm_recall, dt_recall, bag_rf_recall, boost_rf_recall, lda_recall),
  F1 = c(rf_F1, svm_F1, dt_F1, bag_rf_F1, boost_rf_F1, lda_F1),
  Cost = c(rf_cost, svm_cost, dt_cost, bag_rf_cost, boost_rf_cost, lda_cost)
)

modelsss <- modelsss[order(modelsss$Accuracy, decreasing = TRUE), ]

# Display the table
kable(modelsss, row.names = FALSE)

library(ggplot2)

# Define a function to create plots for each metric
create_plots <- function(metric) {
  # Create a plot for each model
  plots <- lapply(modelsss$model, function(model) {
    ggplot(modelsss, aes(x = model, y = .data[[metric]], fill = model)) +
      geom_col(fill = "steelblue",show.legend = FALSE) +
      ggtitle(paste0(metric)) +
      ylab(metric) +
      xlab("Model") +
      theme_bw()
  })
  return(plots)
}

# Create plots for each metric
plots_auc <- create_plots("AUC")
plots_acc <- create_plots("Accuracy")
plots_prec <- create_plots("Precision")
plots_rec <- create_plots("Recall")
plots_f1 <- create_plots("F1")
plots_cost <- create_plots("Cost")


library(ggplot2)

# Create the data frame for the cost metric
cost_data <- data.frame(
  model = c("Logistic Regression","Random Forest", "KNN", "Support Vector Machines", "Decision Tree", "Bagging", "Boosting","Linear Discriminant Analysis"),
  cost = c(glm_cost, rf_cost, knn_cost, svm_cost, dt_cost, bag_rf_cost, boost_rf_cost, lda_cost)
)

# Create the line plot
ggplot(cost_data, aes(x = model, y = cost)) +
  geom_line(color = "blue") +
  geom_point(color = "blue", size = 3) +
  labs(title = "Cost Metric for Each Model",
       x = "Model",
       y = "Cost") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))


# Data cleaning for the testing dataset
# Dimensional reduction steps
df<- df_test

# check for variables with more than 75% data is missing
miss_cols<- lapply(df, function(col){sum(is.na(col))/length(col)})
df<- df[, !(names(df) %in% names(miss_cols[lapply(miss_cols, function(x) x) > 0.75]))]  # 6 cols with more than 75% missing data

# check for variables with more than 80% values are zero
zero_cols<- lapply(df, function(col){length(which(col==0))/length(col)})

#zero_cols<- as.data.frame(zero_cols)
df<- df[, !(names(df) %in% names(zero_cols[lapply(zero_cols, function(x) x) > 0.8]))]

# as all independent variables are continuous in nature, so check for variables where the standard deviation is zero
# remove columns where the standard derivation is zero
std_zero_col <- lapply(df, function(col){sd(col, na.rm = TRUE)})
df <- df[, !(names(std_zero_col) %in% names(std_zero_col[lapply(std_zero_col, function(x) x) == 0]))] # only 1 variable with std dev is zero

# check for near zero variance cols
badCols<- nearZeroVar(df)
names(df[, badCols]) # 8 cols in aps_train data with near zero variance property. removing them
df<- df[, -badCols]

# Missing data visualization
aggr(df, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, 
     labels=names(df), cex.axis=.7, gap=3, 
     ylab=c("Histogram of missing data","Pattern"))


# missing data imputation
imputed<- impute(df,imputation=1, list.class=TRUE, pr=FALSE, check=FALSE)

# convert the list to dataframe
imputed.data <- as.data.frame(do.call(cbind,imputed))

# arrange the columns accordingly
imputed.data <- imputed.data[, colnames(df), drop = FALSE]

# check for missing data
sum(is.na(imputed.data))

# PCA correlation treatment 
df_pca<-PCA(imputed.data, scale.unit = TRUE, ncp = 5,  graph = FALSE)

#Scree plot to visualize the PCA's
screeplot<-fviz_screeplot(df_pca, addlabels = TRUE,
                          barfill = "gray", barcolor = "black",
                          ylim = c(0, 50), xlab = "Principal Component (PC) for continuous variables", ylab = "Percentage of explained variance",
                          main = "(A) Scree plot: Factors affecting APS ",
                          ggtheme = theme_minimal()
)

# Determine Variable contributions to the principal axes
# Contributions of variables to PC1
pc1<-fviz_contrib(df_pca, choice = "var", 
                  axes = 1, top = 10, sort.val = c("desc"),
                  ggtheme= theme_minimal())+
  labs(title="(B) PC-1")

# Contributions of variables to PC2
pc2<-fviz_contrib(df_pca, choice = "var", axes = 2, top = 10,
                  sort.val = c("desc"),
                  ggtheme = theme_minimal())+
  labs(title="(C) PC-2")

fig2<- grid.arrange(arrangeGrob(screeplot), 
                    arrangeGrob(pc1,pc2, ncol=1), ncol=2, widths=c(2,1)) 
annotate_figure(fig2
                ,top = text_grob("Principal Component Analysis (PCA): test data", color = "black", face = "bold", size = 14)
                ,bottom = text_grob("Data source: \n APS\n", color = "brown",
                                    hjust = 1, x = 1, face = "italic", size = 10)
)

# Add a black border around the 2x2 grid plot
grid.rect(width = 1.00, height = 0.99, 
          gp = gpar(lwd = 2, col = "black", fill=NA))
grid.newpage()

# Extract the Principal Components
# select PC1 components only
eig.val<- get_eigenvalue(df_pca)

# A simple method to extract the results, for variables, from a PCA classput is to use the function get_pca_var() [factoextra package].
imp_vars<-factoextra::fviz_contrib(df_pca, choice = "var", 
                                   axes = c(1), top = 10, sort.val = c("desc"))

#save data from contribution plot
dat<-imp_vars$data

#filter class ID's that are higher than 1
r<-rownames(dat[dat$contrib>1,])

#extract these from your original data frame into a new data frame for further analysis
df_imputed_impvars_test<-imputed.data[r] # 49 variables showing maximum variance
df_imputed_impvars_test$class<- y
df_imputed_impvars_test$class<- as.factor(df_imputed_impvars_test$class)
test_data_clean<-df_imputed_impvars_test 
dim(test_data_clean)

# Select numeric columns
numeric_cols <- sapply(test_data_clean, is.numeric)
test_data_numeric <- test_data_clean[, numeric_cols]

# Normalize numeric columns
test_data_normalized <- as.data.frame(scale(test_data_numeric))
test_data_normalized <- cbind(test_data_normalized, class = test_data_clean$class)
View(test_data_normalized)


# PREDICTIVE MODELLING ON IMBALANCED TRAINING DATA
# Run algorithms using 3-fold cross validation
set.seed(3)
index <- createDataPartition(test_data_normalized$class, p = 0.75, list = FALSE, times = 1)
train_data <- train_data_normalized[index, ]
test_data  <- train_data_normalized[-index, ]

calculate_metrics <- function(cm) {
  TP <- cm$table[2, 2]
  FP <- cm$table[1, 2]
  TN <- cm$table[1, 1]
  FN <- cm$table[2, 1]
  
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  F1 <- 2 * precision * recall / (precision + recall)
  
  return(list(precision = precision, recall = recall, F1 = F1))
}


# Logistic Regression
glm_model <- glm(class ~ ., data = train_data, family = "binomial")
glm_pred <- predict(glm_model, newdata = test_data, type = "response")
glm_pred_factor <- factor(ifelse(glm_pred > 0.5, "1", "0"), levels = levels(factor(test_data$class)))
glm_cm <- confusionMatrix(data = glm_pred_factor, reference = factor(test_data$class))
glm_roc <- roc(test_data$class, glm_pred)
glm_auc <- auc(glm_roc)

glm_metrics <- calculate_metrics(glm_cm)
glm_precision <- glm_metrics$precision
glm_recall <- glm_metrics$recall
glm_F1 <- glm_metrics$F1
glm_cost <- 500*glm_cm$table[1,2] + 10*glm_cm$table[2,1]


# Random Forest
rf_model <- randomForest::randomForest(class ~ ., data = train_data, ntree = 500, importance = TRUE)
rf_pred <- predict(rf_model, newdata = test_data)
rf_cm <- confusionMatrix(data = rf_pred, reference = test_data$class)
rf_roc <- roc(test_data$class, as.numeric(rf_pred))
rf_auc <- auc(rf_roc)

rf_metrics   <- calculate_metrics(rf_cm)
rf_precision <- rf_metrics$precision
rf_recall    <- rf_metrics$recall
rf_F1        <- rf_metrics$F1
rf_cost <- 500*rf_cm$table[1,2] + 10*rf_cm$table[2,1]

# KNN
library(class)
k <- 5
knn_model <- knn(train = train_data[-1], test = test_data[-1], cl = train_data$class, k = k)
knn_cm <- confusionMatrix(knn_model, test_data$class)
knn_roc <- roc(test_data$class, as.numeric(knn_model))
knn_auc <- auc(knn_roc)

knn_metrics   <- calculate_metrics(knn_cm)
knn_precision <- knn_metrics$precision
knn_recall    <- knn_metrics$recall
knn_F1        <- knn_metrics$F1
knn_cost <- 500*knn_cm$table[1,2] + 10*knn_cm$table[2,1]

# Classification Trees
ct_model <- rpart::rpart(class ~ ., data = train_data, method = "class")
ct_pred <- predict(ct_model, newdata = test_data, type = "class")
ct_cm <- confusionMatrix(data = ct_pred, reference = test_data$class)
ct_roc <- roc(test_data$class, as.numeric(ct_pred))
ct_auc <- auc(ct_roc)

ct_metrics   <- calculate_metrics(ct_cm)
ct_precision <- ct_metrics$precision
ct_recall    <- ct_metrics$recall
ct_F1        <- ct_metrics$F1
ct_cost <- 500*ct_cm$table[1,2] + 10*ct_cm$table[2,1]

# Support Vector Machines
library(e1071)
svm_model <- svm(class ~ ., data = train_data, type = "C-classification", kernel = "linear")
svm_pred <- predict(svm_model, test_data)
svm_cm <- confusionMatrix(data = svm_pred, reference = test_data$class)
svm_roc <- roc(test_data$class, as.numeric(svm_pred))
svm_auc <- auc(svm_roc)

svm_metrics   <- calculate_metrics(svm_cm)
svm_precision <- svm_metrics$precision
svm_recall    <- svm_metrics$recall
svm_F1        <- svm_metrics$F1
svm_cost <- 500*svm_cm$table[1,2] + 10*svm_cm$table[2,1]

# Decision Tree
dt_model <- rpart::rpart(class ~ ., data = train_data)
dt_pred <- predict(dt_model, newdata = test_data, type = "class")
dt_cm <- confusionMatrix(data = dt_pred, reference = test_data$class)
dt_roc <- roc(test_data$class, as.numeric(dt_pred))
dt_auc <- auc(dt_roc)

dt_metrics   <- calculate_metrics(dt_cm)
dt_precision <- dt_metrics$precision
dt_recall    <- dt_metrics$recall
dt_F1        <- dt_metrics$F1
dt_cost <- 500*dt_cm$table[1,2] + 10*dt_cm$table[2,1]

# Bagging with Random Forest
bag_rf_model <- randomForest::randomForest(class ~ ., data = train_data, ntree = 500, importance = TRUE, mtry = ncol(train_data)-1, replace = TRUE, bag=TRUE)
bag_rf_pred <- predict(bag_rf_model, newdata = test_data)
bag_rf_cm <- confusionMatrix(data = bag_rf_pred, reference = test_data$class)
bag_rf_roc <- roc(test_data$class, as.numeric(bag_rf_pred))
bag_rf_auc <- auc(bag_rf_roc)

bag_rf_metrics   <- calculate_metrics(bag_rf_cm)
bag_rf_precision <- bag_rf_metrics$precision
bag_rf_recall    <- bag_rf_metrics$recall
bag_rf_F1        <- bag_rf_metrics$F1
bag_rf_cost <- 500*bag_rf_cm$table[1,2] + 10*bag_rf_cm$table[2,1]

# Boosting with Random Forest
boost_rf_model <- randomForest::randomForest(class ~ ., data = train_data, ntree = 500, mtry = sqrt(ncol(train_data)-1), importance = TRUE,  do.trace = 100, classwt = c(1, 10),replace = TRUE)
boost_rf_pred <- predict(boost_rf_model, newdata = test_data)
boost_rf_cm <- confusionMatrix(data = boost_rf_pred, reference = test_data$class)
boost_rf_roc <- roc(test_data$class, as.numeric(boost_rf_pred))
boost_rf_auc <- auc(boost_rf_roc)

boost_rf_metrics   <- calculate_metrics(boost_rf_cm)
boost_rf_precision <- boost_rf_metrics$precision
boost_rf_recall    <- boost_rf_metrics$recall
boost_rf_F1        <- boost_rf_metrics$F1
boost_rf_cost <- 500*boost_rf_cm$table[1,2] + 10*boost_rf_cm$table[2,1]

# LDA
library(MASS)
lda_model <- lda(class ~ ., data = train_data)
lda_pred <- predict(lda_model, newdata = test_data)
lda_cm <- confusionMatrix(data = lda_pred$class, reference = test_data$class)
lda_roc <- roc(test_data$class, lda_pred$posterior[, 2])
lda_auc <- auc(lda_roc)

lda_metrics   <- calculate_metrics(lda_cm)
lda_precision <- lda_metrics$precision
lda_recall    <- lda_metrics$recall
lda_F1        <- lda_metrics$F1
lda_cost <- 500*lda_cm$table[1,2] + 10*lda_cm$table[2,1]

# Create a data frame of compare model performances
modelsss <- data.frame(
  model = c("Random Forest", "Support Vector Machines", "Decision Tree", "Bagging", "Boosting","Linear Discriminant Analysis"),
  AUC = c(rf_auc, svm_auc, dt_auc, bag_rf_auc, boost_rf_auc, lda_auc),
  Accuracy = c(rf_cm$overall['Accuracy'], svm_cm$overall['Accuracy'], dt_cm$overall['Accuracy'], bag_rf_cm$overall['Accuracy'], boost_rf_cm$overall['Accuracy'], lda_cm$overall['Accuracy']),
  Precision = c(rf_precision, svm_precision, dt_precision,bag_rf_precision, boost_rf_precision,lda_precision),
  Recall = c(rf_recall, svm_recall, dt_recall, bag_rf_recall, boost_rf_recall, lda_recall),
  F1 = c(rf_F1, svm_F1, dt_F1, bag_rf_F1, boost_rf_F1, lda_F1),
  Cost = c(rf_cost, svm_cost, dt_cost, bag_rf_cost, boost_rf_cost, lda_cost)
)

modelsss <- modelsss[order(modelsss$Accuracy, decreasing = TRUE), ]

# Display the table
kable(modelsss, row.names = FALSE)

library(ggplot2)

# Define a function to create plots for each metric
create_plots <- function(metric) {
  # Create a plot for each model
  plots <- lapply(modelsss$model, function(model) {
    ggplot(modelsss, aes(x = model, y = .data[[metric]], fill = model)) +
      geom_col(fill = "steelblue",show.legend = FALSE) +
      ggtitle(paste0(metric)) +
      ylab(metric) +
      xlab("Model") +
      theme_bw()
  })
  return(plots)
}

# Create plots for each metric
plots_auc <- create_plots("AUC")
plots_acc <- create_plots("Accuracy")
plots_prec <- create_plots("Precision")
plots_rec <- create_plots("Recall")
plots_f1 <- create_plots("F1")
plots_cost <- create_plots("Cost")


library(ggplot2)

# Create the data frame for the cost metric
cost_data <- data.frame(
  model = c("Logistic Regression","Random Forest", "KNN", "Support Vector Machines", "Decision Tree", "Bagging", "Boosting","Linear Discriminant Analysis"),
  cost = c(glm_cost, rf_cost, knn_cost, svm_cost, dt_cost, bag_rf_cost, boost_rf_cost, lda_cost)
)

# Create the line plot
ggplot(cost_data, aes(x = model, y = cost)) +
  geom_line(color = "blue") +
  geom_point(color = "blue", size = 3) +
  labs(title = "Cost Metric for Each Model",
       x = "Model",
       y = "Cost") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
                                                                                                                                
