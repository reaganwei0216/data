# 
# https://www.youtube.com/watch?v=qkivJzjyHoA
# 
# https://kanchengzxdfgcv.blogspot.com/2016/04/r_9.html 18-3-2
# https://rstudio-pubs-static.s3.amazonaws.com/203423_6183225937f946d997034dbd273564fe.html  18-3-2
# https://kanchengzxdfgcv.blogspot.com/2016/05/r_93.html 21 ¼Ò«¬¶EÂ_
# https://kanchengzxdfgcv.blogspot.com/2016/05/r_20.html 20  ¼s¸q½u©Ê¼Ò«¬
# #  23-5  xgboost
# 
# R »y¨¥¨Ï¥ÎªÌªº Python ¾Ç²ßµ§°O

createtable <- matrix(c(118,15,51,30),ncol=2,byrow=TRUE)
colnames(createtable) <- c(0,1)
rownames(createtable) <- c(0,1)
createtable <- as.table(createtable)
createtable
caret::confusionMatrix(createtable)

¶iÀ»ªº¸ê®Æ¬ì¾Ç
https://www.datainpoint.com/data-science-in-action/
  
  ct = trainControl(method = "repeatedcv", number = 10, repeats = 2)
grid_rf = expand.grid(.mtry = c(2, 3, 6))
set.seed(1)
cl = makePSOCKcluster(4)
registerDoParallel(cl)
tr.cvrfclass = train(excellent~., data = train,
                     method = 'rf',
                     metric = "Kappa",
                     trControl = ct,
                     tuneGrid = grid_rf)
stopCluster(cl)
save(tr.cvrfclass, file = "~/Downloads/wine_train_cvrfclass.RData")
load(file = "~/Downloads/wine_train_cvrfclass.RData"); tr.cvrfclass

C:\Users\reaganwei\Documents\cd_R»y¨¥¸ê®Æ¤ÀªR»PÀ³¥Î

ordinal logistic
multinomial logistic regression (question 1)
In this case we can use Multinomial logistic regression as response variable (quality) is a nominal categorical variable with more than 2 levels.
Ordinal logistic regression

https://wangcc.me/LSHTMlearningnote/ordinal-logistic-regression.html

Âå¾Ç²Î­p¾Ç ¤ý¶W¨° Chaochen Wang
https://wangcc.me/LSHTMlearningnote/
  
  https://www.r-bloggers.com/how-to-perform-ordinal-logistic-regression-in-r/
  https://stats.idre.ucla.edu/r/dae/ordinal-logistic-regression/
  
  ¦h¶µÅÞ¿è¼Ò«¬¡]Multinomial logistic regression¡^¡A¥¦¬OLogistic regressionªºÂX¥R¡A¸ÑÄÀ¤èªk³£¤@¼Ë¡C
°ß¤@¤£¦P¤§³B¦b©ó­n±N¨ÌÅÜ¶µ¨ä¤¤¤@­ÓÃþ§O³]¬°¡u°Ñ·Ó²Õ¡v¡]Baseline category / Reference group¡^¡A
°²³]¨ÌÅÜ¶µ¦³¤TÃþ¡A¨º»ò°jÂk«Y¼Æ¸ÑÅª¬°¡u·í¦ÛÅÜ¶µ¼W¥[¤@­Ó³æ¦ì¡A¨ÌÅÜ¶µA¬Û¹ï¨ÌÅÜ¶µCªº¾÷²v·|¼W¥[´X­¿¡v
https://dasanlin888.pixnet.net/blog/post/34468457

http://www.r-web.com.tw/stat/step1.php?method=g_logit
reaganwei/ r-webr-web



# histogram ª½¤è¹Ï
hist(people$age,col=rainbow(15),xlab="¦~ÄÖ",ylab="¤H¼Æ")
hist(people$age)
# ªø±ø¹Ï
barplot(table(people$rank))
barplot(table(people$rank),col=rainbow(7))

idx <- caret::createDataPartition(winequality_white$quality, p = 0.8, list = FALSE)
trainset <- winequality_white[idx,]
testset <- winequality_white[-idx,]
nrow(trainset);prop.table(table(trainset$quality))
nrow(testset);prop.table(table(testset$quality))


churnTrain <- churnTrain[ , ! names(churnTrain) %in% c('state', 'account_length', 'area_code')]
people2 = subset(people,select=-c(name,note)) # ¥h°£©m¦W¸ò³Æµù


people2$rank<- ordered(people2$rank, levels = c(1, 2, 3)) ## Create as Ordered Factor).


https://github.com/PacktPublishing/Machine-Learning-with-R-Cookbook/tree/master/Chapter05

idx <- sample.int(2, nrow(churnTrain), replace=TRUE, prob= c(0.7, 0.3))
trainset <- churnTrain[idx == 1, ]
testset  <- churnTrain[idx == 2, ]

# ÅÞ¿è°jÂk¤ÀªR¹ê°µ-----
C:\Users\reaganwei\Documents\tibame\TibaMe ¾Ç²ßºô_R»y¨¥¸ê®Æ¬ì¾Ç®aºë­×¯Z\Lab 6-2.mp4
https://github.com/PacktPublishing/Machine-Learning-with-R-Cookbook/blob/master/Chapter05/chapter5.Rmd
fit <- glm(churn ~ ., data = trainset, family=binomial)
summary(fit)
pred <- predict(fit, testset, type= 'response') # 'arg' should be one of ¡§link¡¨, ¡§response¡¨, ¡§terms¡¨


library(party)
fit <- party::ctree(Species ~ ., data = trainset)
summary(fit)
pred <- predict(fit, testset, type= 'response') # 'arg' should be one of ¡§link¡¨, ¡§response¡¨, ¡§terms¡¨

library(e1071)
fit <- e1071::naiveBayes(Species ~ ., data = trainset)
summary(fit)
pred <- predict(fit, testset, type= 'class')

C:\Users\reaganwei\Documents\tibame\TibaMe ¾Ç²ßºô_R»y¨¥¸ê®Æ¬ì¾Ç®aºë­×¯Z\Lab 6-3.mp4
library(e1071)
model.tuned <- svm(churn~., data = trainset, gamma = 10^-2, cost = 100)
summary(model.tuned)
svm.tuned.pred <- predict(model.tuned, testset[, !names(testset) %in% c("churn")])

library(rpart)
fit <- rpart::rpart(Species ~ ., data = iris)
summary(fit)
# rpart¦³±MÄÝªºÃ¸¹Ï®M¥órpart.plot¡A¨ç¦¡¬Oprp()
# ³Ì¤U­±¸`ÂIªº¼Æ¦r¡A¥Nªí¡Gnumber of correct classifications / number of observations in that node
require(rpart.plot) 
prp(fit,                # ¼Ò«¬
    faclen=0,           # §e²{ªºÅÜ¼Æ¤£­nÁY¼g
    fallen.leaves=TRUE, # Åý¾ðªK¥H««ª½¤è¦¡§e²{
    shadow.col="gray",  # ³Ì¤U­±ªº¸`ÂI¶î¤W³±¼v
    # number of correct classifications / number of observations in that node
    extra=2)  
fit$$variable.importance
pred <- predict(fit, testset, type= 'class')

library(caret)
tb <- table(testtag, pred)
cm <- confusionMatrix(tb)
cm

library(caret)
control = caret::trainControl(method="repeatedcv", number=10, repeats=3)
#?train
# ?caret::getModelInfo
#names(getModelInfo())
model = caret::train(churn~., data=trainset, method="rpart", preProcess="scale", trControl=control)
model


# ROC Curve-----
#¨Ï¥ÎROCR
#install.packages('ROCR')
library(ROCR)

#tip  end----



# createtable----
createtable <- matrix(c(118,15,51,30),ncol=2,byrow=TRUE)
colnames(createtable) <- c(0,1)
rownames(createtable) <- c(0,1)
createtable <- as.table(createtable)
createtable
caret::confusionMatrix(createtable)


mean(c(1,2,3,NA),na.rm = TRUE)
summary(c(1,2,3,NA),na.rm = TRUE)
summary(c(1,2,3,NA))


# Q1.1----
# http://rstudio-pubs-static.s3.amazonaws.com/438329_edfaab4011ce44a59fb9ae2d216d8dea.html #6. Variable interaction and variable selection

library(readr)
winequality_white <- read_delim("C:/Users/reaganwei/Documents/bank_test/winequality-white.csv",";", escape_double = FALSE, trim_ws = TRUE)
#Check for missing values
colSums(is.na(winequality_white))
head(winequality_white)
table(winequality_white$quality)

library(corrplot)
cor.white <- cor(winequality_white)
corrplot:corrplot(cor.white, method = 'number')

library(caret)
idx <- caret::createDataPartition(winequality_white$quality, p = 0.8, list = FALSE)
trainset <- winequality_white[idx,]
testset <- winequality_white[-idx,]
nrow(trainset);prop.table(table(trainset$quality))
nrow(testset);prop.table(table(testset$quality))


tr.lm.interract = lm(quality~ .^2, data = trainset)
summary(tr.lm.interract)
anova(tr.lm.interract)
AIC(tr.lm.interract)
BIC(tr.lm.interract)

# str(trainset$quality)
# trainset$quality <- as.factor(trainset$quality)
# tr.glm.interract = glm(quality~ .^2, data = trainset,family = binomial)
# summary(tr.glm.interract)
# anova(tr.glm.interract)
# AIC(tr.glm.interract)
# BIC(tr.glm.interract)
# pred <- predict(tr.glm.interract, testset, type= 'response')



Ypred = predict(tr.lm.interract, newdata=testset,type='response')
Ypred
tr.lm.interract.pred2 = predict(tr.lm.interract, testset)
tr.lm.interract.pred = predict(tr.lm.interract, testset[, 1:12])
identical(tr.lm.interract.pred2,tr.lm.interract.pred)


# q1.2------- 
# http://rstudio-pubs-static.s3.amazonaws.com/438329_edfaab4011ce44a59fb9ae2d216d8dea.html #4. Decision Tree
# install.packages("C50")
library(C50)
library(readr)
winequality_white <- read_delim("C:/Users/reaganwei/Documents/bank_test/winequality-white.csv",";", escape_double = FALSE, trim_ws = TRUE)
#Check for missing values
colSums(is.na(winequality_white))
head(winequality_white)
table(winequality_white$quality)
winequality_white$quality <- as.factor(winequality_white$quality)
str(winequality_white$quality)


idx <- createDataPartition(winequality_white$quality, p = 0.8, list = FALSE)
trainset <- winequality_white[idx,]
testset <- winequality_white[-idx,]
nrow(trainset);prop.table(table(trainset$quality))
nrow(testset);prop.table(table(testset$quality))


# for (i in c(2,3,4,5,6,7,8,9,10,11)) {
#   print(i)
#   tr.dt = C5.0(trainset[,1:i], trainset$quality)
#   tr.dt.pred = predict(tr.dt, testset[,1:i])
#   confMat = confusionMatrix(tr.dt.pred, testset$quality, positive="1")
#   tr.dt.eval = list(auc = NA, confusionMatrix = confMat);
#   print(tr.dt.eval$confusionMatrix$overall[1])
# }



#decision tree
tr.dt = C5.0(trainset[,1:9], trainset$quality)
tr.dt


tr.dt.pred = predict(tr.dt, testset[,1:9])
CconfMat2 = confusionMatrix(tr.dt.pred, testset$quality)
confMat = confusionMatrix(tr.dt.pred, testset$quality, positive="1")
# confMat$overall[1]


tr.dt.eval = list(auc = NA, confusionMatrix = confMat)
tr.dt.eval


tr.dt.eval$confusionMatrix$overall[1]
# Accuracy : 0.59 




lm0 = lm(quality ~ 1, data = trainset)
tr.lm.interract.step = step(lm0, ~ (fixed_acidity + volatile_acidity + 
                                      citric_acid + residual_sugar +  chlorides + free_sulfur_dioxide +
                                      total_sulfur_dioxide + density + pH + sulphates + alcohol)^2, 
                            direction = "both", trace = 0)
summary(tr.lm.interract.step)




#no1  multinomial regression using R -----
# https://rpubs.com/HashimotoArina/453177

# install.packages("tree")

library(dplyr)

confmatrix = function(Y,Ypred)
{
  t1 = table(Y,Ypredict=Ypred)  
  print(t1)
  p = sum(diag(t1))/sum(t1)*100
  cat("\n\n¹w´ú¥¿½T²v = ",p,"% \n")
}
head(iris[,c(1,2,3,4)])

# C:\Users\reaganwei\Dropbox\R³nÅé¨t¦C½Òµ{\RDM_2015\DM_codes.txt

# CART¡Gtree ®M¥ó tree ¨ç¼Æ -----
library(tree)
str(iris$Species)
result = tree(Species ~ . , data=iris)
p1 = predict(result,iris[,c(1,2,3,4)],type="class")
p1 = predict(result,type="class")

confmatrix(iris$Species, p1)
table(iris$Species, p1)
table(iris$Species, p1) %>% diag() %>% sum() 
table(iris$Species, p1)  %>% sum() 

# CART¡Grpart ®M¥ó rpart ¨ç¼Æ -----

library(rpart)
result2 = rpart(Species ~ . ,data=iris); 
plot(result2)
text(result2)
Ypred2 = predict(result2,type="class")
confmatrix(iris$Species, Ypred2)
result2$variable.importance

# https://rpubs.com/allan811118/R_programming_08 ----
require(rpart)
# ¥ý§â¸ê®Æ°Ï¤À¦¨ train=0.8, test=0.2 
set.seed(22)
iris.index <- sample(x=1:nrow(iris), size=ceiling(0.8*nrow(iris) ))
trainset <- iris[iris.index, ]
dim(trainset)
testset <- iris[-iris.index, ]
dim(testset)
# CARTªº¼Ò«¬
cart.model<- rpart(trainset$Species ~. ,data=trainset)
# ¿é¥X¦U¸`ÂIªº²Ó³¡¸ê°T(§e²{¦bconsoleµøµ¡)
cart.model
# rpart¦³±MÄÝªºÃ¸¹Ï®M¥órpart.plot¡A¨ç¦¡¬Oprp()
# ³Ì¤U­±¸`ÂIªº¼Æ¦r¡A¥Nªí¡Gnumber of correct classifications / number of observations in that node
require(rpart.plot) 
prp(cart.model,         # ¼Ò«¬
    faclen=0,           # §e²{ªºÅÜ¼Æ¤£­nÁY¼g
    fallen.leaves=TRUE, # Åý¾ðªK¥H««ª½¤è¦¡§e²{
    shadow.col="gray",  # ³Ì¤U­±ªº¸`ÂI¶î¤W³±¼v
    # number of correct classifications / number of observations in that node
    extra=2)  
# ¥Î¥t¤@­ÓÃ¸¹Ï®M¥ópartykit¡A¨ç¦¡¬Oas.party()©Mplot()
require(partykit)   
rparty.tree <- as.party(cart.model) # Âà´«cart¨Mµ¦¾ð
rparty.tree # ¿é¥X¦U¸`ÂIªº²Ó³¡¸ê°T
plot(rparty.tree) 

# ¹w´ú¥Î predict()
pred <- predict(cart.model, newdata=testset, type="class")
# ¥Îtable¬Ý¹w´úªº±¡ªp
table(real=testset$Species, predict=pred)
# ­pºâ¹w´ú·Ç½T²v = ¹ï¨¤½uªº¼Æ¶q/Á`¼Æ¶q
confus.matrix <- table(real=testset$Species, predict=pred)
sum(diag(confus.matrix))/sum(confus.matrix) # ¹ï¨¤½uªº¼Æ¶q/Á`¼Æ¶q
# ÁÙ¦³´£¤Éªº¥i¯à¶Ü¡H§Ú­ÌÄ~Äò¹ï¼Ò«¬¶i¦æ­×¾ð~
printcp(cart.model) # ¥ýÆ[¹î¥¼­×°Åªº¾ð¡ACPÄæ¦ì¥Nªí¾ðªº¦¨¥»½ÆÂø«×°Ñ¼Æ
plotcp(cart.model) # µe¹ÏÆ[¹î¥¼­×°Åªº¾ð

# §Q¥Î¯à¨Ï¨Mµ¦¾ð¨ã¦³³Ì¤p»~®tªºCP¨Ó­×°Å¾ð
prunetree_cart.model <- prune(cart.model, cp = cart.model$cptable[which.min(cart.model$cptable[,"xerror"]),"CP"])
# ­×°Å§¹¨Mµ¦¾ð¤§«á¡AÅý§Ú­Ì­«·s«Øºc¤@¦¸¹w´ú¼Ò«¬
prunetree_pred <- predict(prunetree_cart.model, newdata=testset, type="class")
# ¥Îtable¬Ý¹w´úªº±¡ªp
table(real=testset$Species, predict=prunetree_pred)
prunetree_confus.matrix <- table(real=testset$Species, predict=prunetree_pred)
sum(diag(prunetree_confus.matrix))/sum(prunetree_confus.matrix) # ¹ï¨¤½uªº¼Æ¶q/Á`¼Æ¶q
# ¬°¤FÁ×§K¼Ò«¬¹L«×ÀÀ¦X(overfitting)¡A¬G­n§Q¥ÎK-fold Cross-Validationªº¤èªk¶i¦æ¥æ¤eÅçÃÒ¡A¨Ï¥Îcaret³o­Ó®M¥ó¡A¦ÓK¥ý³]©w¬°10¦¸

require(caret)
require(e1071)
train_control <- trainControl(method="cv", number=10)
train_control.model <- train(Species~., data=trainset, method="rpart", trControl=train_control)
train_control.model



#  randomForest -----
library(randomForest)
set.seed(71)
result = randomForest(Species ~ . , data=iris , importance=TRUE, proximity=TRUE)
print(result)

round(importance(result), 2)
names(result)
result$predicted
result$importance
result$confusion

t = result$confusion
t
sum(diag(t))/sum(t)


# ¥Î randomForest ¨Ó§@¤À¸s¡G¥u¥Î³Ì«e­± 4 ­ÓÅÜ¼Æ¡A¤£¨Ï¥Î²Ä 5 ­ÓÅÜ¼Æ Species
set.seed(17)
result2 = randomForest(iris[, -5])
result2
MDSplot(result2, iris$Species, palette=rep(1, 3), pch=as.numeric(iris$Species))


# http://www.cc.ntu.edu.tw/chinese/epaper/0034/20150920_3410.html
install.packages("C50")
library(C50)
iris.C5=C5.0(Species~ . ,data=iris)
summary(iris.tree)
plot(iris.tree)

# rpubs.com/allan811118/R_programming_00
# CART¨Mµ¦¾ð¨Ó½m²ß¡A¹ïÀ³ªº®M¥ó¬Orpart-----

# https://rstudio-pubs-static.s3.amazonaws.com/237448_25448d1a60d24e599e9531bf76c39f20.html
# Basic exploratory data analysis to get a sense of data
library(readr)
winequality_white <- read_delim("C:/Users/reaganwei/Documents/bank_test/winequality-white.csv",";", escape_double = FALSE, trim_ws = TRUE)
#Check for missing values
colSums(is.na(winequality_white))
str(winequality_white)
winequality_white$quality <- as.factor(winequality_white$quality)
table(winequality_white$quality)

idx <- sample.int(2, nrow(winequality_white), replace=TRUE, prob= c(0.8, 0.2))
trainset <- winequality_white[idx == 1, ]
testset  <- winequality_white[idx == 2, ]


idx <- createDataPartition(winequality_white$quality, p = 0.8, list = FALSE)
trainset <- winequality_white[idx,]
testset <- winequality_white[-idx,]
nrow(trainset);prop.table(table(trainset$quality))
nrow(testset);prop.table(table(testset$quality))

cor_mat=cor(winequality_white[,! names(winequality_white) %in% c("quality")])
high=findCorrelation(cor_mat,cutoff = 0.60)
names(winequality_white)[high]
# caret::createDataPartition() ???


library(rminer)
model <- fit(quality~.,trainset,model="rpart")
VariableImportance <- rminer::Importance(model,trainset,method="sensv")
VariableImportance
L=list(runs=1,sen=t(VariableImportance$imp),sresponses=VariableImportance$sresponses)
L$sen
L$sresponses
mgraph(L,graph="IMP",leg=names(trainset),col="gray",Grid=10)
names(winequality_white)

M=rminer::fit(quality~.,trainset,model="cv.glmnet",task="class") # pure classes



control = caret::trainControl(method="repeatedcv", number=10, repeats=3)
#?train
# ?caret::getModelInfo
#names(getModelInfo())
model = caret::train(quality~., data=winequality_white, method="rpart", preProcess="scale", trControl=control)
model = caret::train(quality~ alcohol+sulphates+total_sulfur_dioxide+density+chlorides+residual_sugar+pH+volatile_acidity, data=winequality_white, method="rpart", preProcess="scale", trControl=control)
model = caret::train(quality~ alcohol+volatile_acidity, data=winequality_white, method="rpart", preProcess="scale", trControl=control)

model = caret::train(quality~., data=winequality_white, method="svmLinear", preProcess="scale", trControl=control)
model = caret::train(quality~ alcohol+volatile_acidity, data=winequality_white, method="svmLinear", preProcess="scale", trControl=control)

summary(model)
pred <- predict(model,newdata = testset, type = 'raw')
pred2 <- predict(model, newdata=testset, type= 'prob')
model$modelInfo
plot(model)
model$results
model$finalModel
plot(model$finalModel,margin = 0.1)
text(model$finalModel)
model

varImp(model,scale = FALSE)

tb <- table(real=testset$quality, pred=pred)
cm <- caret::confusionMatrix(tb)
cm
# Accuracy : 0.5155 rpart all
# Accuracy : 0.5174 rpart
# Accuracy : 0.5123  svmLinear all

#install.packages("corrplot")
library(corrplot)  
wine_corr <- cor(winequality_white)
summary(wine_corr)
corrplot(wine_corr,method="number")
# ³q¹Lpsych¥]¤¤ªºdescribe()¨ç¼Æ­pºâ´y­z©Ê²Î­p¶q
# Skewness and kurtosis
# http://estat.ncku.edu.tw/topic/desc_stat/base/Skewness.html   #Skewness
# http://estat.ncku.edu.tw/topic/desc_stat/base/Kurtosis.html   #kurtosis
#install.packages("psych")
#library("psych")
describe(winequality_white)
# Histogram and table to get frquency of different qualities
hist_1 <- hist(winequality_white$quality)
hist_1
#print(hist_1$breaks)
# Change number of breaks and add labels
hist_2 <- hist(winequality_white$quality, breaks = 5, col ="lightblue", xlab = "Quality ", main= " Histogram of Quality rating")
table(winequality_white$quality)

# split the data into test and train data
set.seed(1235)
wine_redwhite <- winequality_white[sample(nrow(winequality_white)),]
split <- floor(nrow(winequality_white)/2)
wine_train <- winequality_white[0:split,]
wine_test <- winequality_white[(split+1):(nrow(winequality_white)-1),]
View(wine_train)
View(wine_test)
summary(wine_train)
str(wine_train)
str(wine_test)
plot(wine_test)
library(nnet)
wine_train$quality <- as.factor(wine_train$quality)
mlogit_model <- multinom(quality ~. ,data =wine_train, maxit = 1000) 
mlogit_model
mlogit_output <- summary(mlogit_model)

mlogit_output
names(mlogit_output)
mlogit_output$coefficients
mlogit_output$standard.errors
mlogit_output$AIC
z <- mlogit_output$coefficients/mlogit_output$standard.errors
p <- (1-pnorm(abs(z),0,1))*2 # I am using two-tailed z test
print(z, digits =2)
print(p, digits =2)
# The p-Value for quality tells us that "free_sulfur_dioxide" and "total_sulfur_dioxide"  are not significant. 
Pquality5 <- rbind(mlogit_output$coefficients[2, ],mlogit_output$standard.errors[2, ],z[2, ],p[2, ])
Pquality5
rownames(Pquality5) <- c("Coefficient","Std. Errors","z stat","p value")
knitr::kable(Pquality5)

oddsML <- exp(coef(mlogit_output))
print(oddsML, digits =2)
str(wine_test)
predictedML <- predict(mlogit_model,wine_test,na.action =na.pass, type="probs")
predicted_classML <- predict(mlogit_model,wine_test)
predicted_classML2 <- predict(mlogit_model,wine_test,type="class")
table(predicted_classML==predicted_classML2)
#wine_train$quality
# confusion matrix
cm=table(read=as.factor(wine_test$quality),pred=as.factor(predicted_classML))
library(caret)
confusionMatrix(cm)

caret::confusionMatrix(as.factor(wine_test$quality),as.factor(predicted_classML) )
str(as.factor(predicted_classML))
str(as.factor(wine_test$quality))
mean(as.character(predicted_classML) != as.character(wine_test$quality))


# winequality_white rpart------
# https://datasciencebeginners.com/2018/12/20/multinomial-logistic-regression-using-r/
# Loading the dplyr package
library(dplyr)
library(readr)
winequality_white <- read_delim("C:/Users/reaganwei/Documents/bank_test/winequality-white.csv", 
                                ";", escape_double = FALSE, trim_ws = TRUE)
winequality_white$quality <- as.factor(winequality_white$quality)
str(winequality_white$quality)
head(winequality_white)
summary(winequality_white)
# Using sample_frac to create 80 - 20 slipt into test and train
train <- sample_frac(winequality_white, 0.8)
sample_id <- as.numeric(rownames(train)) # rownames() returns character so as.numeric
test <- winequality_white[-sample_id,]
# Setting the basline 
train$quality <- relevel(train$quality, ref = "9")
str(train$quality)
table(train$quality)
table(winequality_white$quality)
# Loading the nnet package
require(nnet)
# Training the multinomial model
names(winequality_white)
multinom.fit <- multinom(quality ~ fixed_acidity + volatile_acidity  + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density + pH + sulphates + alcohol -1, data = train)

# Checking the model
summary(multinom.fit)

## extracting coefficients from the model and exponentiate
exp(coef(multinom.fit))

head(probability.table <- fitted(multinom.fit))

dim(train)
dim(test)

# Predicting the values for train dataset
train$precticed <- predict(multinom.fit, newdata = train, "class")

# Building classification table
ctable <- table(real=train$quality, pred=train$precticed)

# Calculating accuracy - sum of diagonal elements divided by total obs
round((sum(diag(ctable))/sum(ctable))*100,2)

# winequality_white rpart------
library(readr)
winequality_white <- read_delim("C:/Users/reaganwei/Documents/bank_test/winequality-white.csv", 
                                ";", escape_double = FALSE, trim_ws = TRUE)
head(winequality_white)
str(winequality_white)
winequality_white$quality <- as.factor(winequality_white$quality)

require(rpart)
# ¥ý§â¸ê®Æ°Ï¤À¦¨ train=0.8, test=0.2 
set.seed(22)
winequality_white.index <- sample(x=1:nrow(winequality_white), size=ceiling(0.8*nrow(winequality_white) ))
trainset <- winequality_white[winequality_white.index, ]
dim(trainset)
testset <- winequality_white[-winequality_white.index, ]
dim(testset)
# CARTªº¼Ò«¬
cart.model<- rpart(trainset$quality ~. ,data=trainset)
# ¿é¥X¦U¸`ÂIªº²Ó³¡¸ê°T(§e²{¦bconsoleµøµ¡)
cart.model
# rpart¦³±MÄÝªºÃ¸¹Ï®M¥órpart.plot¡A¨ç¦¡¬Oprp()
# ³Ì¤U­±¸`ÂIªº¼Æ¦r¡A¥Nªí¡Gnumber of correct classifications / number of observations in that node
require(rpart.plot) 
prp(cart.model,         # ¼Ò«¬
    faclen=0,           # §e²{ªºÅÜ¼Æ¤£­nÁY¼g
    fallen.leaves=TRUE, # Åý¾ðªK¥H««ª½¤è¦¡§e²{
    shadow.col="gray",  # ³Ì¤U­±ªº¸`ÂI¶î¤W³±¼v
    # number of correct classifications / number of observations in that node
    extra=2)  
# ¥Î¥t¤@­ÓÃ¸¹Ï®M¥ópartykit¡A¨ç¦¡¬Oas.party()©Mplot()
require(partykit)   
rparty.tree <- as.party(cart.model) # Âà´«cart¨Mµ¦¾ð
rparty.tree # ¿é¥X¦U¸`ÂIªº²Ó³¡¸ê°T
plot(rparty.tree) 
# ¹w´ú¥Î predict()
pred <- predict(cart.model, newdata=testset, type="class")
# ¥Îtable¬Ý¹w´úªº±¡ªp
table(real=testset$quality, predict=pred)
# ­pºâ¹w´ú·Ç½T²v = ¹ï¨¤½uªº¼Æ¶q/Á`¼Æ¶q
confus.matrix <- table(real=testset$quality, predict=pred)
sum(diag(confus.matrix))/sum(confus.matrix) # ¹ï¨¤½uªº¼Æ¶q/Á`¼Æ¶q
# ÁÙ¦³´£¤Éªº¥i¯à¶Ü¡H§Ú­ÌÄ~Äò¹ï¼Ò«¬¶i¦æ­×¾ð~
printcp(cart.model) # ¥ýÆ[¹î¥¼­×°Åªº¾ð¡ACPÄæ¦ì¥Nªí¾ðªº¦¨¥»½ÆÂø«×°Ñ¼Æ
plotcp(cart.model) # µe¹ÏÆ[¹î¥¼­×°Åªº¾ð

# §Q¥Î¯à¨Ï¨Mµ¦¾ð¨ã¦³³Ì¤p»~®tªºCP¨Ó­×°Å¾ð
prunetree_cart.model <- prune(cart.model, cp = cart.model$cptable[which.min(cart.model$cptable[,"xerror"]),"CP"])
# ­×°Å§¹¨Mµ¦¾ð¤§«á¡AÅý§Ú­Ì­«·s«Øºc¤@¦¸¹w´ú¼Ò«¬
prunetree_pred <- predict(prunetree_cart.model, newdata=testset, type="class")
# ¥Îtable¬Ý¹w´úªº±¡ªp
table(real=testset$quality, predict=prunetree_pred)
prunetree_confus.matrix <- table(real=testset$quality, predict=prunetree_pred)
sum(diag(prunetree_confus.matrix))/sum(prunetree_confus.matrix) # ¹ï¨¤½uªº¼Æ¶q/Á`¼Æ¶q
# ¬°¤FÁ×§K¼Ò«¬¹L«×ÀÀ¦X(overfitting)¡A¬G­n§Q¥ÎK-fold Cross-Validationªº¤èªk¶i¦æ¥æ¤eÅçÃÒ¡A¨Ï¥Îcaret³o­Ó®M¥ó¡A¦ÓK¥ý³]©w¬°10¦¸
require(caret)
require(e1071)
train_control <- trainControl(method="cv", number=10)
train_control.model <- train(quality~., data=trainset, method="rpart", trControl=train_control)
train_control.model









# install.packages("mlogit")
suppressMessages(library("mlogit")) # multinomial logit
library("caret")


# https://datasciencebeginners.com/2018/12/20/multinomial-logistic-regression-using-r/


install.packages("rattle.data")
# Loading the library
library(rattle.data)
# Loading the wine data
data(wine)
# Checking the structure of wine dataset
str(wine)
# Loading the dplyr package
library(dplyr)

# Using sample_frac to create 70 - 30 slipt into test and train
# train <- sample_frac(wine, 0.7)
train <- sample_frac(wine, 1)
sample_id <- as.numeric(rownames(train)) # rownames() returns character so as.numeric
test <- wine[-sample_id,]
# Setting the basline 
train$Type <- relevel(train$Type, ref = "")


# Fit the model
model <- nnet::multinom(Species ~., data = train.data)
# Summarize the model
summary(model)
# Make predictions
predicted.classes <- model %>% predict(test.data)
head(predicted.classes)
# Model accuracy
mean(predicted.classes == test.data$Species)

# https://github.com/dataspelunking/MLwR/blob/master/Machine%20Learning%20with%20R%20(2nd%20Ed.)/Chapter%2006/MLwR_v2_06.r
library(rpart)
library(caret)
# ??ˆæ?Šè?‡æ?™å?€??†æ?? train=0.8, test=0.2
intrain<-createDataPartition(y=winequality_white$quality,p=0.8,list=FALSE)
training<-winequality_white[intrain,]
testing<-winequality_white[-intrain,]
# CART??„æ¨¡???
# http://www.rpubs.com/skydome20/R-Note6-Apriori-DecisionTree
m.rpart <- rpart(quality ~ ., data = training)
m.rpart
require(rpart.plot) 
prp(m.rpart,         # æ¨¡å??
    faclen=0,           # ??ˆç¾??„è?Šæ•¸ä¸è?ç¸®å¯?
    fallen.leaves=TRUE, # è®“æ¨¹??ä»¥??‚ç›´?–¹å¼å?ˆç¾
    shadow.col="gray",  # ??€ä¸‹é¢??„ç?€é»žå?—ä?Šé™°å½?
    # number of correct classifications / number of observations in that node
    extra=1) 
pred <- predict(m.rpart, newdata=testing)
# ?”¨table??‹é?æ¸¬??„æ?…æ??
table(real=testing$quality, predict=pred)
# è¨ˆç?—é?æ¸¬æº–ç¢º??? = å°è?’ç?šç?„æ•¸???/ç¸½æ•¸???
confus.matrix <- table(real=testing$quality, predict=pred)
sum(diag(confus.matrix))/sum(confus.matrix) # å°è?’ç?šç?„æ•¸???/ç¸½æ•¸???


# get basic information about the tree
m.rpart

# get more detailed information about the tree
summary(m.rpart)

# use the rpart.plot package to create a visualization
library(rpart.plot)

# a basic decision tree diagram
rpart.plot(m.rpart, digits = 3)

# a few adjustments to the diagram
rpart.plot(m.rpart, digits = 4, fallen.leaves = TRUE, type = 3, extra = 101)

## Step 4: Evaluate model performance ----

# generate predictions for the testing dataset
p.rpart <- predict(m.rpart, wine_test)

# compare the distribution of predicted values vs. actual values
summary(p.rpart)
summary(wine_test$quality)


# winequality.white <- read.csv("~/bank_test/winequality-white.csv", sep=";")
library(tidyverse)
library(caret)
library(nnet)
# Load the data
data("iris")
# Inspect the data
sample_n(iris, 3)
# Split the data into training and test set
set.seed(123)
training.samples <- iris$Species %>% 
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- iris[training.samples, ]
test.data <- iris[-training.samples, ]
test.data$Species %>%table()
??nnet
# Fit the model
model <- nnet::multinom(Species ~., data = train.data)
# Summarize the model
summary(model)

# Make predictions
predicted.classes <- model %>% predict(test.data)
head(predicted.classes)
# Model accuracy
mean(predicted.classes == test.data$Species)


# randomForest-----
library(caret)
library(randomForest)
idx <- createDataPartition(wine$quality, p = 0.8, list = FALSE)
trainset <- wine[idx,]
testset <- wine[-idx,]
nrow(trainset);prop.table(table(trainset$quality))
nrow(testset);prop.table(table(testset$quality))

set.seed(71)			# ³]©wÀH¾÷¼Æ²£¥Í¾¹ªì­Èset.seed(111)			# ³]©wÀH¾÷¼Æ²£¥Í¾¹ªì­È
wine.rf=randomForest(quality~.,data=trainset,importance=TRUE,proximity=TRUE,ntree=500)	 # «Øºc¨Mµ¦¾ð¬°500´ÊªºÀH¾÷´ËªL¼Ò«¬

str(trainset)
pred=predict(wine.rf,testset)		# ®Ú¾Ú¼Ò«¬wine.rf¹ïx¸ê®Æ¶i¦æ¹w´ú
cm <- table(testset$quality,pred)
caret::confusionMatrix(cm)

importance(wine.rf)

importanceFactor <- as.data.frame(importance(wine.rf))
importanceFactor <-importanceFactor[order(importanceFactor$MeanDecreaseAccuracy,decreasing = TRUE),]
rownames(importanceFactor)

library(C50)
tr.dt = C5.0(trainset[,c("alcohol","volatile_acidity","free_sulfur_dioxide")], trainset$quality)
tr.dt = C5.0(trainset[,c("alcohol","volatile_acidity")], trainset$quality)
tr.dt
plot(tr.dt)

tr.dt.pred = predict(tr.dt, testset[,c("alcohol","volatile_acidity","free_sulfur_dioxide")])
tr.dt.pred = predict(tr.dt, testset[,c("alcohol","volatile_acidity")])
                                    
confMat2 = confusionMatrix(tr.dt.pred, testset$quality)
confMat = confusionMatrix(tr.dt.pred, testset$quality, positive="1")
# confMat$overall[1]
Accuracy : 0.5307




                                    
                                    
control = caret::trainControl(method="repeatedcv", number=10, repeats=3)
#?train
# ?caret::getModelInfo
#names(getModelInfo())
model = caret::train(quality~alcohol+volatile_acidity, data=trainset, method="rpart", preProcess="scale", trControl=control)
model
pred=predict(model,testset)		# ®Ú¾Ú¼Ò«¬wine.rf¹ïx¸ê®Æ¶i¦æ¹w´ú
cm <- table(testset$quality,pred)
caret::confusionMatrix(cm)
# Accuracy : 0.5164






#no2-----
# è³‡æ?™ä?Œï?šç?“åŽ»è­˜åˆ¥??–ç?„å¸³??™äº¤??“è?‡æ??(txhist.csvï¼Œå…±50?¬ç­†è?‡æ??)
# è³‡æ?™ä?‰ï?šåŒ¯??‡è?‡æ??(FXRATE.csvï¼Œå?„ç¨®å¹??ˆ¥å°å°å¹???„åŒ¯??‡è?‡æ??)
library(readr)
library(sqldf)
txhist <- read_csv("bank_test/txhist.csv", 
                   col_types = cols(AMT = col_double()))
View(txhist)
str(txhist)
FXRATE <- read_csv("bank_test/FXRATE.csv")
str(FXRATE)
??base::merge
# http://rstudio-pubs-static.s3.amazonaws.com/13602_96265a9b3bac4cb1b214340770aa18a1.html
df=base::merge(x=txhist,y=FXRATE,by='CUR_CODE',all.x = TRUE)
df$NT <- df$AMT*df$FXR_TWD
df2<-sqldf("SELECT ID,DR,sum(NT) from df group by ID,DR ",drv="SQLite")
head(df2,20)
??sqldf
df_DX<-sqldf("SELECT ID,DR,sum(NT) AS D from df  group by ID,DR  ",drv="SQLite")
ID_UNIQ<-sqldf("SELECT DISTINCT(ID) AS ID from df",drv="SQLite")

df_D=df2[df_DX$DR=='D',]
df_C=df2[df_DX$DR=='C',]
head(df_D,10)
names(df_D)[names(df_D)=="sum(NT)"] <- "sum(D)"
names(df_C)[names(df_C)=="sum(NT)"] <- "sum(C)"

head(df_C,10) #è²¸æ–¹
head(df_D,10) #?€Ÿæ–¹


df_D$`sum(D)` %>% summary()
Q1 <- summary(df_D$`sum(D)`) [2]
Q1
# å®¢æˆ¶?€Ÿæ–¹ç¸½é?‘é?ä¸­??„ç¬¬ä¸€??›å?†ä?æ•¸(Q1)??„é?‘é?è?‡å?æ?‰å®¢?ˆ¶ID----
df_D[df_D$`sum(D)`==Q1,]
dim(df_D[df_D$`sum(D)`==Q1,])[1]
# å®¢æˆ¶?€Ÿæ–¹ç¸½é?‘é?ä¸­??„ä¸­ä½æ•¸(Q2)??„é?‘é?è?‡å?æ?‰å®¢?ˆ¶ID----
Q2 <- summary(df_D$`sum(D)`) [3]
Q2
df_D[df_D$`sum(D)`==Q2,]

head(ID_UNIQ,10)
df3 <- base::merge(x=ID_UNIQ,y=df_C,by='ID',all.x = TRUE)
head(df3)
df4 <- base::merge(x=df3,y=df_D,by='ID',all.x = TRUE)
head(df4)
is.na(head(df4))
is.na(df4$`sum(C)`) 
is.na(df4$`sum(D)`) 
head(df4)
is.na(df4$`sum(C)`) %>% head()
df4$`sum(C)`[is.na(df4$`sum(C)`)] <- 0
df4$`sum(D)`[is.na(df4$`sum(D)`)] <- 0
head(df4)
df4$D_c <- df4$`sum(D)`-df4$`sum(C)`
head(df4)
df5 <- df4[order(df4$D_c,decreasing = TRUE),]
# (2)	[?€Ÿæ–¹ç¸½é?‘é?? ?€? è²¸æ–¹ç¸½é?‘é?]??„æ?€å¤§å€¼è?‡å?æ?‰å®¢?ˆ¶ID----
df5[1,c('ID','D_c')]


