#Data preparation
rain = read.csv("weatherAUS.csv")
rain = data.frame(rain)
head(rain)
print(c(nrow(rain), ncol(rain)))
summary(rain)
str(rain)
df_null = is.na(rain)
colSums(df_null)
rain = rain[,c(-2,-6, -7, -18,-19)]
head(rain)
summary(rain)
df_null = is.na(rain)
colSums(df_null)

rain$MinTemp[is.na(rain$MinTemp)]=mean(rain$MinTemp,na.rm=T)
rain$MaxTemp[is.na(rain$MaxTemp)]=mean(rain$MaxTemp,na.rm=T)
rain$Rainfall[is.na(rain$Rainfall)]=mean(rain$Rainfall,na.rm=T)
rain$WindGustSpeed[is.na(rain$WindGustSpeed)]=mean(rain$WindGustSpeed,na.rm=T)
rain$WindSpeed9am[is.na(rain$WindDir9am)]=mean(rain$WindSpeed9am,na.rm=T)
rain$WindSpeed3pm[is.na(rain$WindDir3pm)]=mean(rain$WindSpeed3pm,na.rm=T)
rain$Humidity9am[is.na(rain$Humidity9am)]=mean(rain$Humidity9am,na.rm=T)
rain$Humidity3pm[is.na(rain$Humidity3pm)]=mean(rain$Humidity3pm,na.rm=T)
rain$Pressure9am[is.na(rain$Pressure9am)]=mean(rain$Pressure9am,na.rm=T)
rain$Pressure3pm[is.na(rain$Pressure3pm)]=mean(rain$Pressure3pm,na.rm=T)
rain$Temp9am[is.na(rain$Temp9am)]=mean(rain$Temp9am,na.rm=T)
rain$Temp3pm[is.na(rain$Temp3pm)]=mean(rain$Temp3pm,na.rm=T)

library(tidyr)
rain$WindGustDir <- replace_na(rain$WindGustDir, replace = "W")
rain$WindDir9am <- replace_na(rain$WindDir9am, replace = "N")
rain$WindDir3pm <- replace_na(rain$WindDir3pm, replace = "SE")

rain$RainToday<-replace_na(rain$RainToday, replace = "Yes")
rain$RainTomorrow<-replace_na(rain$RainTomorrow, replace = "Yes")

df_null = is.na(rain)
colSums(df_null)

rain_corr = rain
rain_corr
rain_corr$WindGustDir <-as.integer(rain_corr$WindGustDir)
rain_corr$WindDir9am <- as.integer(rain_corr$WindDir9am)
rain_corr$WindDir3pm <- as.integer(rain_corr$WindDir3pm)
rain_corr$RainToday <- as.integer(rain_corr$RainToday)
rain_corr$RainTomorrow <- as.integer(rain_corr$RainTomorrow)
rain_corr <- rain_corr[,-1]
rain_corr


library(ggplot2)
library(GGally)
ggcorr(rain_corr)
ggheatmap +
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 4)+
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    legend.direction = "horizontal") +
  guides(fill = guide_colourbar(title.position = "top", title.hjust = 0.5))
    

rain_int = rain_corr 
rain_int
rain_int = rain_int[, c(-10, -14,-15 )]
rain_int$RainTomorrow = rain_int$RainTomorrow-1
rain_int$RainToday = rain_int$RainToday-1
rain_int
rain_logi = rain_int
rain_logi$WindGustDir <- as.factor(rain.train_logi$WindGustDir)
rain_logi$WindDir9am <- as.factor(rain.train_logi$WindDir9am)
rain_logi$WindDir9am
rain_logi$WindDir3pm <- as.factor(rain.train_logi$WindDir3pm)
rain_logi$RainToday <- as.factor(rain.train_logi$RainToday)
rain_logi$RainTomorrow <- as.factor(rain_logi$RainTomorrow)
#split train and test dataset
train = sample(1:nrow(rain_logi), nrow(rain_logi)/2)
rain.train = rain_logi[train, ]
rain.test = rain_logi[-train, ]
RainTomorrow.test = rain_logi$RainTomorrow[-train]
rain.train
#logistic model

logi_model<-glm(RainTomorrow~., family = "binomial",data = rain.train)
summary(logi_model)
library(MASS)
step.model<-stepAIC(logi_model, direction = "both", trace = FALSE)
summary(step.model)
logi_model1<-glm(RainTomorrow~.-WindGustDir-WindDir9am-WindDir3pm-RainToday, family = "binomial",data = rain.train)
summary(logi_model1)
rain.pred = predict(logi_model1, rain.test, type = "response")
rain.pred.class <-ifelse(rain.pred>0.5, 1,0)
mean(rain.pred.class == rain.test$RainTomorrow)
# SVM
library(e1071)
library(ROCR)
library(tree)
library(gbm)
library(randomForest)
library(ISLR)

nrow(rain.train)
train_svm = sample(1:nrow(rain.train), 10000)
rain.train_svm = rain.train[train_svm,]
svmfit = svm(RainTomorrow~., data = rain.train_svm, kernel = "linear", cost = 10, scale = FALSE) 


svmfit
summary(svmfit)
plot(svmfit, rain.train_svm)
svmfit$index    # the index of the support vectors
summary(svmfit)

ypred = predict(svmfit, rain.test)
table(predict = ypred, truth = rain.test$RainTomorrow)


#cross-validation on different cost
set.seed(1)
tune.out = tune(svm, RainTomorrow~., data= rain.train_svm, kernel = "linear", ranges = list(cost=c(0.01, 10, 100)))
tune.out    # output best parameters and best performance, best performance means the error of the best parameter
summary(tune.out)
bestmod = tune.out$best.model
summary(bestmod)


svmfit_poly = svm(RainTomorrow~., data = rain.train_svm, kernel = "polynomial", cost = 10, scale = FALSE) 

summary(svmfit_poly)
plot(svmfit_poly, rain.train_svm)
svmfit_poly$index    # the index of the support vectors


ypred_poly = predict(svmfit_poly, rain.test)
table(predict = ypred_poly, truth = rain.test$RainTomorrow)





#freestyle part
library(xgboost)
library(Matrix)
rain.train
train_matrix <- sparse.model.matrix(RainTomorrow~.-1, data = rain.train)
test_matrix <- sparse.model.matrix(RainTomorrow~.-1, data = rain.test)
train_label <- as.numeric(rain.train$RainTomorrow ==1)
test_label <- as.numeric(rain.test$RainTomorrow == 1)
train_fin <-list(data = train_matrix, label = train_label)
test_fin <- list(data = train_matrix, label = test_label)
dtrain <- xgb.DMatrix(data = train_fin$data, label = train_fin$label)
dtest <- xgb.DMatrix(data = test_fin$data, label = test_fin$label)
xgb<- xgboost(data = dtrain, max_depth = 15, eta = 0.5, objective = 'binary:logistic', nround = 25)
library(Ckmeans.1d.dp)
importance <- xgb.importance(train_matrix@Dimnames[[2]], model = xgb)  
head(importance)
xgb.ggplot.importance(importance)



pre_xgb = round(predict(xgb, newdata = dtest))
tt <- table(test_label, pre_xgb, dnn = c("true","pre"))
(tt[1,1]+tt[2,2])/sum(tt)    #the rate of correct predictions
1-(tt[1,1]+tt[2,2])/sum(tt)
library(pROC)
xgboost_roc <- roc(test_label,as.numeric(pre_xgb))
plot(xgboost_roc, print.auc=TRUE, auc.polygon=TRUE, 
     grid=c(0.1, 0.2),grid.col=c("green", "red"), 
     max.auc.polygon=TRUE,auc.polygon.col="skyblue", 
     print.thres=TRUE,main='ROC curve')


























