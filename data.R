library(tidyverse)
library(caret)
library(corrplot)
library(RWeka)
library(PerformanceAnalytics)
library(car)
library(psych)
library(caretEnsemble)
library(doParallel)
setwd("C:/Users/awe/OneDrive - Queensland University of Technology/qut units/2021 sem 1/IFN704")
data <- read.csv("diabetes_data_upload.csv")
hehe
data$class <- factor(data$class)

corrplot(cor(data), method = 'square')

boxplot(data, col = 'orange', main = 'Features Boxplot')

hist(data$Age)
hist(data$Gender)
hist(data$class)

set.seed(123)
data_rand <- data[sample(1:nrow(data)), ]
dim(data_rand)

X = data_rand %>% select(-class)
y = data_rand %>% select(class)

part.index <- createDataPartition(data_rand$class, 
                                  p = 0.75,                         
                                  list = FALSE)
X_train <- X[part.index, ]
X_test <- X[-part.index, ]
y_train <- y[part.index]
y_test <- y[-part.index]

str(X_train)
str(X_test)
str(y_train)
str(y_test)


################################


getDoParWorkers()
set.seed(123)
my_control <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE,
                           savePredictions = "final")

##############################

set.seed(222)
model_list <- caretList(X_train,
                        y_train,
                        trControl = my_control,
                        methodList = c('svmRadial', 'rf', 
                                       'xgbTree', 'xgbLinear','gbm'),
                        tuneList = NULL,
                        continue_on_fail = FALSE, 
                        preProcess = NULL)


resamples <- resamples(model_list)
dotplot(resamples, metric = 'Accuracy')
#######################################
pred_svm <- predict.train(model_list$svmRadial, newdata = X_test)
pred_rf <- predict.train(model_list$rf, newdata = X_test)
pred_xgbT <- predict.train(model_list$xgbTree, newdata = X_test)
pred_xgbL <- predict.train(model_list$xgbLinear, newdata = X_test)
pred_gbm <- predict.train(model_list$gbm, newdata = X_test)

y_test
y_test_factor <- factor(y_test)


confusionMatrix(pred_svm,y_test_factor)
confusionMatrix(pred_xgbT,y_test_factor,positive="1")
confusionMatrix(pred_gbm,y_test_factor,positive="1")

check <- confusionMatrix(pred_gbm,y_test_factor,positive="1")
#######################################

set.seed(123)
xgbTree_model <- train(X_train,
                       y_train,
                       trControl = my_control,
                       method = 'xgbTree',
                       metric = 'Accuracy',
                       preProcess = NULL,
                       importance = TRUE)
plot(varImp(xgbTree_model))

randomfor <- train(X_train,
                       y_train,
                       trControl = my_control,
                       method = 'rf',
                       metric = 'Accuracy',
                       preProcess = NULL,
                       importance = TRUE)
plot(varImp(randomfor))


xgbTree_model
