


# Chapter 2



# summarise_all {dplyr}
# ts {stats}
??vars




# EXAMPLE 2.3.1 SCORECARD DEVELOPMENT
# Let us consider a retail loan portfolio database. The goal is to show how to build a scorecard
# through the following steps:
# 1. Default flag definition and data preparation,
# 2. Univariate analysis,
# 3. Multivariate analysis, and
# 4. Stepwise regression.
# 1. Default flag definition and data preparation
# 1.1. Import data
# oneypd<- read.csv('chap2oneypd.csv') [,2:45]
chap2oneypd <- read.csv('C:\\Users\\bigdata\\Documents\\chap2oneypd.csv', stringsAsFactors=FALSE)
oneypd<-chap2oneypd [,2:45]
library(dplyr)
# 1.1.1. Data overview: data content and format
dplyr::glimpse(oneypd)
# $ id <int> 6670001, 9131199...
# $ vintage_year <int> 2005, 2006...
# $ monthly_installment <dbl> 746.70, 887.40...
# $ loan_balance <dbl> 131304.44, 115486.51...
# $ bureau_score <int> 541, 441...
# ...
# 1.1.2. Date format
library(vars)
oneypd <- dplyr::mutate_at(oneypd, vars(contains('date')),funs(as.Date))
class(chap2oneypd$origination_date)
class(chap2oneypd$maturity_date)
class(oneypd$origination_date)
class(oneypd$maturity_date)
# 1.1.3. Round arrears count fields
oneypd$max_arrears_12m<- round(oneypd$max_arrears_12m,4)
oneypd$arrears_months<- round(oneypd$arrears_months,4)
head(chap2oneypd$max_arrears_12m)
head(oneypd$max_arrears_12m)
head(chap2oneypd$arrears_months)
head(oneypd$arrears_months)


# 1.2. Default flag definition
oneypd<- dplyr::mutate(oneypd,default_event = if_else(oneypd$arrears_event == 1 |oneypd$term_expiry_event == 1 |oneypd$bankrupt_event == 1, 1,0))
head(oneypd$arrears_event)
head(oneypd$term_expiry_event)
head(oneypd$bankrupt_event)
head(oneypd$default_event)

??dplyr::if_else

# 1.3. Database split in train and test samples
# Recode default event variables for more convenient use
# 0-default, 1-non-default
oneypd$default_flag<-  dplyr::if_else(oneypd$default_event == 1,0,1)
# Perform a stratified sampling: 70% train and 30% test
library(caret)
set.seed(2122)
train.index <- caret::createDataPartition(oneypd$default_event,p = .7, list = FALSE)
train <- oneypd[ train.index,]
test <- oneypd[-train.index,]


??smbinning.sumiv
??smbinning.sumiv.plot
# 2. Univariate analysis
# Information Value (IV) assessment
library(smbinning)
iv_analysis<- smbinning.sumiv(df=train,y='default_flag')
summary(iv_analysis)

# iv_analysis$Char[!is.na(iv_analysis$IV)] %>% as.character()  %>% class()



# Plot IV summary table
par(mfrow=c(1,1))
smbinning.sumiv.plot(iv_analysis,cex=1)

??smbinning.plot


# 3. Multivariate analysis
# Compute Spearman rank correlation based on variables' WOE
# based on Table 2.2 binning scheme
woe_vars<- train %>%   dplyr::select(starts_with('woe'))
woe_corr<- cor(as.matrix(woe_vars), method = 'spearman')
# Graphical inspection
library(corrplot)
corrplot(woe_corr, method = 'number')


# library(corrplot)
# data(mtcars)
# head(mtcars)
# M <- cor(mtcars)
# corrplot(M, order='hclust', addrect=3)
 
# woe_vars <- train[,iv_analysis$Char[!is.na(iv_analysis$IV) &  iv_analysis$IV >0.4 ] %>% as.character()]
# head(woe_vars)

# 4. Stepwise regression
# 4.1 Discard highly correlated variable

woe_vars$woe_max_arrears_bal_6m

woe_vars_clean<- woe_vars %>%
  dplyr::select( -woe_max_arrears_bal_6m)
#Support functions and databases
library(MASS)
attach(train)
# 4.2 Stepwise model fitting
logit_full<- glm(default_event~ woe_bureau_score+
                   woe_annual_income+woe_emp_length+woe_max_arrears_12m
                 +woe_months_since_recent_cc_delinq+woe_num_ccj+woe_cc_util,
                 family = binomial(link = 'logit'), data = train)
logit_stepwise<- stepAIC(logit_full, k=qchisq(0.05, 1,lower.tail=F), direction = 'both')
detach(train)
summary(logit_stepwise)





# scaled_score <- function(pd_selected) {
#   Odds <- 19  #評分卡模型的基礎odds是19:1
#   Score  <- 600
#   pdo <- 50 
#   Factor  <- pdo / log(2) # 每增加50點odds翻一倍  Factor = pdo/Ln(2)
#   Offset  <- Score  - Factor*log(Odds)  # Offset = Score — (Factor × ln(Odds))
#   
#   scores <- Offset  + Factor*log((1 - pd_selected) / pd_selected)
#   return(round(scores, 0))
#   
# }
# 
# pd_selected=0.6243
# Offset  + Factor*log((1 - pd_selected) / pd_selected)
# 
# # https://blog.csdn.net/textboy/article/details/46975985
# 
# # 比例因子和偏移量为：
# # 令好坏比为50，对应的评分为600；在些基础上评分值增加20分
# # factor = 20 / log(2)
# # Offset = 600 – factor * log(50)
# factor=28.85
# Offset=600-28.85* log(50)
# 600 = log(50) * factor + Offset



# EXAMPLE 2.3.2 FROM SCORE TO POINTS
# Let us consider the model developed in Example 2.3.1. Our aim is to define a new scale with
# anchor set at 660 points and log-odds doubling each 40 points. A 72:1 odds ratio is identified in
# line with credit bureau common practice. The following steps are performed:
# 1. Define a scaling function, and
# 2. Score the entire dataset.
# 1. Define a scaling function
scaled_score <- function(logit, odds, offset = 500, pdo = 20)
{
  b = pdo/log(2)
  a = offset - b*log(odds)
  round(a + b*log((1-logit)/logit))
}
# 2. Score the entire dataset
library(dplyr)
# 2.1 Use fitted model to score both test and train datasets
predict_logit_test <- predict(logit_stepwise, newdata = test, type = 'response')
predict_logit_train <- predict(logit_stepwise, newdata = train, type = 'response')

# 2.2 Merge predictions with train/test data
test$predict_logit <- predict(logit_stepwise, newdata = test, type = 'response')
train$predict_logit <- predict(logit_stepwise, newdata = train, type = 'response')
train$sample = 'train'
test$sample = 'test'
data_whole <- rbind(train, test)
data_score <- data_whole %>%
  dplyr::select(id, default_event, default_flag, woe_bureau_score,
                woe_annual_income, woe_max_arrears_12m,
                woe_months_since_recent_cc_delinq,
                woe_cc_util, sample, predict_logit)
# 2.3 Define scoring parameters in line with objectives
data_score$score<-  scaled_score(data_score$predict_logit, 72, 660, 40)
library(ggplot2)
ggplot(data_score, aes(x = score)) + geom_bar() 







# library(randomForest)
# set.seed(4543)
# data(mtcars)
# table(mtcars$vs)
# rm(mtcars.rf)
# mtcars.rf <- randomForest(mtcars$vs ~ ., data=mtcars)
# importance(mtcars.rf)
# varImpPlot(mtcars.rf)

# library(smbinning) # Load package and its data
# pop=smbsimdf1 # Set population
# train=subset(pop,rnd<=0.7) # Training sample
# # Binning application for a numeric variable
# result=smbinning(df=train,y='fgood',x='dep') # Run and save result
# # Generate a dataset with binned characteristic
# pop2=smbinning.gen(pop,result,'g1dep')



# EXAMPLE 2.3.3 PD CALIBRATION
# We aim to calibrate the scorecard investigated in Examples 2.3.1 and 2.3.2, by relying on
# Equation (2.6). For the sake of simplicity, the same data is used to perform the calibration,
# by means of the following steps:
# • Upload data, and
# • Fit logistic regression.
# 1. Upload data
attach(data_score)
head(data_score)
# 2. Fit logistic regression
pd_model<- glm(default_event~ score,
               family = binomial(link = 'logit'), data = data_score)
summary(pd_model)
# Estimate Std. Error z value Pr(>|z|)
# (Intercept) 7.1357807 0.1855217 38.46 <0.0000 ***
# score -0.0173218 0.0003519 -49.23 <0.0000 ***
# ---
# Signif. codes: 0 ‘***' 0.001 ‘**' 0.01 ‘*' 0.05 ‘.' 0.1 ‘ ' 1
# 2.1 Use model coefficients to obtain PDs
data_score$pd<- predict(pd_model, newdata = data_score,type = 'response')
head(data_score)


# EXAMPLE 2.3.4 MODEL DISCRIMINATORY POWER VALIDATION
# Let us consider the model developed in Example 2.3.1. Discriminatory power is inspected
# by means of the following:
# 1. Gini index, and
# 2. ROC curve.
# 1. Gini index
library(optiRum)
gini_train<- optiRum::giniCoef(train$predict_logit,train$default_event)
print(gini_train)
# 0.8335868
gini_test<- optiRum::giniCoef(test$predict_logit,test$default_event)
print(gini_test)


# Both train and test Gini indices (that is, 0.833 and 0.823) pinpoint a strong discriminatory
# power. (See Figure 2.8.)
# 2. ROC curve
library(pROC)
roc(train$default_event,train$predict_logit,direction='<')
plot(roc(train$default_event,train$predict_logit,direction='<'),col='blue', lwd=3, main='ROC Curve')

# EXAMPLE 2.3.5 COMPARISON OF ACTUAL VERSUS FITTED PDS (BY SCORE BAND)----
# Let us focus on the calibration process described in Example 2.3.3. The validation is
# performed by means of the following steps:
# 1. Create score bands, and
# 2. Compare actual against fitted PDs.

# 1. Create a validation database
# 1.1. Create score bands
library(smbinning)
data_score$score
data_score$default_flag
score_cust<- smbinning.custom(data_score, y = 'default_flag',x= 'score', cuts= c(517,576,605,632,667,716,746,773))
# 1.2. Group by bands
data_score<- smbinning.gen(data_score, score_cust,chrname = 'score_band')

# 2. Compare actual against fitted PDs
# 2.1. Compute mean values
data_score$pd
data_score$default_event
data_pd<- data_score %>%
  dplyr::select(score, score_band, pd, default_event) %>%
  dplyr::group_by(score_band) %>%
  dplyr::summarise(mean_dr = round(mean(default_event),4),mean_pd = round(mean(pd),4))
data_pd
# 2.2. Compute rmse
rmse<-sqrt(mean((data_pd$mean_dr - data_pd$mean_pd)^2))
rmse
# 0.002732317


# EXAMPLE 2.3.6 CROSS-VALIDATION
# Let us consider the model developed in Example 2.3.1. The following steps are performed:
# 1. Prepare cross-validation dataset, and
# 2. Perform cross-validation loop.


# 1. Prepare the cross-validation dataset
data_subset<- data_whole %>%
  dplyr::select(id, default_event, default_flag, woe_bureau_score,
                woe_annual_income, woe_max_arrears_12m,
                woe_months_since_recent_cc_delinq, woe_cc_util, sample)

# 2. Perform the cross-validation loop
# 2.1 Initialise loop arguments and vectors
j<-1 #initialise counter
m<- 20 #number of folds
n = floor(nrow(data_subset)/m) #size of each fold
auc_vector<- rep(NA,m)
gini_vector<- rep(NA, m)
ks_vector<- rep(NA, m)
# 2.2 Run the loop
attach(data_subset)

for (j in 1:m)
{
  s1 = ((j-1)*n+1) #start of the subset (fold)
  s2 = (j*n) # end of the subset (fold)
  data_cv_subset = s1:s2 #range of the subset (fold)
  train_set <- data_subset[-data_cv_subset, ]
  test_set <- data_subset[data_cv_subset, ]
  # Model Fitting
  model <- glm(default_event~ woe_bureau_score+
                 woe_annual_income+woe_max_arrears_12m+
                 woe_months_since_recent_cc_delinq+woe_cc_util,
               family=binomial(link = 'logit'), data = train_set)
  # Predict results
  predict_cv <- predict(model, newdata = test_set,
                        type = 'response')
  pred_obj<- ROCR::prediction(predict_cv, test_set[,2])
  perf_obj<- ROCR::performance(pred_obj, 'tpr', 'fpr')
  # Calculate performance metrics for each fold/run:
  test_auc<- ROCR::performance(pred_obj, 'auc')
  auc_vector[j] <- test_auc@y.values[[1]]
  gini_vector[j]<- optiRum::giniCoef(predict_cv,
                                     test_set[,2])
}













#Bureau score:
train$woe_bureau_score<- rep(NA, length(train$bureau_score))
train$woe_bureau_score[which(is.na(train$bureau_score))] <- -0.0910
train$woe_bureau_score[which(train$bureau_score <= 308)] <- -0.7994
train$woe_bureau_score[which(train$bureau_score > 308 & train$bureau_score <= 404)] <- -0.0545
train$woe_bureau_score[which(train$bureau_score > 404 & train$bureau_score <= 483)] <-  0.7722
train$woe_bureau_score[which(train$bureau_score > 483)] <-  1.0375

test$woe_bureau_score<- rep(NA, length(test$bureau_score))
test$woe_bureau_score[which(is.na(test$bureau_score))] <- -0.0910
test$woe_bureau_score[which(test$bureau_score <= 308)] <- -0.7994
test$woe_bureau_score[which(test$bureau_score > 308 & test$bureau_score <= 404)] <- -0.0545
test$woe_bureau_score[which(test$bureau_score > 404 & test$bureau_score <= 483)] <-  0.7722
test$woe_bureau_score[which(test$bureau_score > 483)] <-  1.0375

#CC utilization:

train$woe_cc_util<- rep(NA, length(train$cc_util))
train$woe_cc_util[which(is.na(train$cc_util))] <- 0
train$woe_cc_util[which(train$cc_util <= 0.55)] <- 1.8323
train$woe_cc_util[which(train$cc_util > 0.55 & train$cc_util <= 0.70)] <- -0.4867
train$woe_cc_util[which(train$cc_util > 0.70 & train$cc_util <= 0.85)] <- -1.1623
train$woe_cc_util[which(train$cc_util > 0.85)] <- -2.3562

test$woe_cc_util<- rep(NA, length(test$cc_util))
test$woe_cc_util[which(is.na(test$cc_util))] <- 0
test$woe_cc_util[which(test$cc_util <= 0.55)] <- 1.8323
test$woe_cc_util[which(test$cc_util > 0.55 & test$cc_util <= 0.70)] <- -0.4867
test$woe_cc_util[which(test$cc_util > 0.70 & test$cc_util <= 0.85)] <- -1.1623
test$woe_cc_util[which(test$cc_util > 0.85)] <- -2.3562

#Number of CCJ events:

train$woe_num_ccj<- rep(NA, length(train$num_ccj))
train$woe_num_ccj[which(is.na(train$num_ccj))] <- -0.0910
train$woe_num_ccj[which(train$num_ccj <= 0)] <- 0.1877
train$woe_num_ccj[which(train$num_ccj > 0 & train$num_ccj <= 1)] <- -0.9166
train$woe_num_ccj[which(train$num_ccj > 1)] <- -1.1322

test$woe_num_ccj<- rep(NA, length(test$num_ccj))
test$woe_num_ccj[which(is.na(test$num_ccj))] <- -0.0910
test$woe_num_ccj[which(test$num_ccj <= 0)] <- 0.1877
test$woe_num_ccj[which(test$num_ccj > 0 & test$num_ccj <= 1)] <- -0.9166
test$woe_num_ccj[which(test$num_ccj > 1)] <- -1.1322

#Maximum arrears in previous 12 months:

train$woe_max_arrears_12m<- rep(NA, length(train$max_arrears_12m))
train$woe_max_arrears_12m[which(is.na(train$max_arrears_12m))] <- 0
train$woe_max_arrears_12m[which(train$max_arrears_12m <= 0)] <- 0.7027
train$woe_max_arrears_12m[which(train$max_arrears_12m > 0 & train$max_arrears_12m <= 1)] <- -0.8291
train$woe_max_arrears_12m[which(train$max_arrears_12m > 1 & train$max_arrears_12m <= 1.4)] <- -1.1908
train$woe_max_arrears_12m[which(train$max_arrears_12m > 1.4)] <- -2.2223

test$woe_max_arrears_12m<- rep(NA, length(test$max_arrears_12m))
test$woe_max_arrears_12m[which(is.na(test$max_arrears_12m))] <- 0
test$woe_max_arrears_12m[which(test$max_arrears_12m <= 0)] <- 0.7027
test$woe_max_arrears_12m[which(test$max_arrears_12m > 0 & test$max_arrears_12m <= 1)] <- -0.8291
test$woe_max_arrears_12m[which(test$max_arrears_12m > 1 & test$max_arrears_12m <= 1.4)] <- -1.1908
test$woe_max_arrears_12m[which(test$max_arrears_12m > 1.4)] <- -2.2223

#Maximum arrears balance in previous 6 months:
train$woe_max_arrears_bal_6m<- rep(NA, length(train$max_arrears_bal_6m))
train$woe_max_arrears_bal_6m[which(is.na(train$max_arrears_bal_6m))] <- 0
train$woe_max_arrears_bal_6m[which(train$max_arrears_bal_6m <= 0)] <- 0.5771
train$woe_max_arrears_bal_6m[which(train$max_arrears_bal_6m > 0 & train$max_arrears_bal_6m <= 300)] <- -0.7818
train$woe_max_arrears_bal_6m[which(train$max_arrears_bal_6m > 300 & train$max_arrears_bal_6m <= 600)] <- -1.2958
train$woe_max_arrears_bal_6m[which(train$max_arrears_bal_6m > 600 & train$max_arrears_bal_6m <= 900)] <- -1.5753
train$woe_max_arrears_bal_6m[which(train$max_arrears_bal_6m > 900)] <- -2.2110

test$woe_max_arrears_bal_6m<- rep(NA, length(test$max_arrears_bal_6m))
test$woe_max_arrears_bal_6m[which(is.na(test$max_arrears_bal_6m))] <- 0
test$woe_max_arrears_bal_6m[which(test$max_arrears_bal_6m <= 0)] <- 0.5771
test$woe_max_arrears_bal_6m[which(test$max_arrears_bal_6m > 0 & test$max_arrears_bal_6m <= 300)] <- -0.7818
test$woe_max_arrears_bal_6m[which(test$max_arrears_bal_6m > 300 & test$max_arrears_bal_6m <= 600)] <- -1.2958
test$woe_max_arrears_bal_6m[which(test$max_arrears_bal_6m > 600 & test$max_arrears_bal_6m <= 900)] <- -1.5753
test$woe_max_arrears_bal_6m[which(test$max_arrears_bal_6m > 900)] <- -2.2110

#Employment length (years):

train$woe_emp_length<- rep(NA, length(train$emp_length))
train$woe_emp_length[which(is.na(train$emp_length))] <- 0
train$woe_emp_length[which(train$emp_length <= 2)] <- -0.7514
train$woe_emp_length[which(train$emp_length > 2 & train$emp_length <= 4)] <- -0.3695
train$woe_emp_length[which(train$emp_length > 4 & train$emp_length <= 7)] <-  0.1783
train$woe_emp_length[which(train$emp_length > 7)] <- 0.5827

test$woe_emp_length<- rep(NA, length(test$emp_length))
test$woe_emp_length[which(is.na(test$emp_length))] <- 0
test$woe_emp_length[which(test$emp_length <= 2)] <- -0.7514
test$woe_emp_length[which(test$emp_length > 2 & test$emp_length <= 4)] <- -0.3695
test$woe_emp_length[which(test$emp_length > 4 & test$emp_length <= 7)] <-  0.1783
test$woe_emp_length[which(test$emp_length > 7)] <- 0.5827

#Months since recent CC delinquency:
train$woe_months_since_recent_cc_delinq<- rep(NA, length(train$months_since_recent_cc_delinq))
train$woe_months_since_recent_cc_delinq[which(is.na(train$months_since_recent_cc_delinq))] <- 0
train$woe_months_since_recent_cc_delinq[which(train$months_since_recent_cc_delinq <= 6)] <- -0.4176
train$woe_months_since_recent_cc_delinq[which(train$months_since_recent_cc_delinq > 6 & train$months_since_recent_cc_delinq <= 11)] <- -0.1942
train$woe_months_since_recent_cc_delinq[which(train$months_since_recent_cc_delinq > 11)] <-  1.3166

test$woe_months_since_recent_cc_delinq<- rep(NA, length(test$months_since_recent_cc_delinq))
test$woe_months_since_recent_cc_delinq[which(is.na(test$months_since_recent_cc_delinq))] <- 0
test$woe_months_since_recent_cc_delinq[which(test$months_since_recent_cc_delinq <= 6)] <- -0.4176
test$woe_months_since_recent_cc_delinq[which(test$months_since_recent_cc_delinq > 6 & test$months_since_recent_cc_delinq <= 11)] <- -0.1942
test$woe_months_since_recent_cc_delinq[which(test$months_since_recent_cc_delinq > 11)] <-  1.3166

#Annual income:

train$woe_annual_income<- rep(NA, length(train$annual_income))
train$woe_annual_income[which(is.na(train$annual_income))] <- 0
train$woe_annual_income[which(train$annual_income <= 35064)] <- -1.8243
train$woe_annual_income[which(train$annual_income > 35064 & train$annual_income <= 41999)] <- -0.8272
train$woe_annual_income[which(train$annual_income > 41999 & train$annual_income <= 50111)] <- -0.3294
train$woe_annual_income[which(train$annual_income > 50111 & train$annual_income <= 65050)] <-  0.2379
train$woe_annual_income[which(train$annual_income > 65050)] <-  0.6234

test$woe_annual_income<- rep(NA, length(test$annual_income))
test$woe_annual_income[which(is.na(test$annual_income))] <- 0
test$woe_annual_income[which(test$annual_income <= 35064)] <- -1.8243
test$woe_annual_income[which(test$annual_income > 35064 & test$annual_income <= 41999)] <- -0.8272
test$woe_annual_income[which(test$annual_income > 41999 & test$annual_income <= 50111)] <- -0.3294
test$woe_annual_income[which(test$annual_income > 50111 & test$annual_income <= 65050)] <-  0.2379
test$woe_annual_income[which(test$annual_income > 65050)] <-  0.6234



# 2.3 Plot performance metrics
par(mfrow=c(2,1))
hist(unlist(auc_vector), xlab = 'AUC', ylab = 'Frequency', main = 'AUC Distribution: 20 Folds', col = 'royal blue')
hist(unlist(gini_vector), xlab = 'Gini', ylab = 'Frequency', main = 'Gini Distribution: 20 Folds', col = 'dark magenta')
hist(unlist(ks_vector), xlab = 'KS', ylab = 'Frequency', main = 'KS Distribution: 20 Folds', col = 'light green')
detach(data_subset)


# Example 2.4.1
# EXAMPLE 2.4.1 DECISION TREES FOR DEFAULT ANALYSIS
# Let us consider the portfolio studied in Example 2.8.1, where 20 accounts are considered over
# a one-year horizon: 5 accounts defaulted (that is, default flag 1), whereas the remaining 15 did
# not (that is, default flag 0). We consider both regression and classification trees by focusing on
# the following steps:
# 1. Upload data and create categories “No”, “Yes”,
# 2. Fit classification tree, and
# 3. Fit regression tree

# 1. Upload data and create categories 'No', 'Yes'
def <- read.csv('C:\\Users\\bigdata\\Documents\\chap2ptfregression.csv', stringsAsFactors=FALSE)
head(def)

source('http://bioconductor.org/biocLite.R')
biocLite('tree')

library(tree)
defflag_char=ifelse(def$DEF==0,'No','Yes')
def_new=data.frame(def,defflag_char)
# 2. Fit classification tree
def_cla_tree=tree(defflag_char~. -DEF,data=def_new)
summary(def_cla_tree)
plot(def_cla_tree)
text(def_cla_tree,pretty=0)
# 3. Fit regression tree
def_reg_tree=tree(DEF~., data=def)
summary(def_reg_tree)
plot(def_reg_tree)
text(def_reg_tree,pretty=0)


# Example 2.4.2
# EXAMPLE 2.4.2 RANDOM FOREST AND BOOSTING
# Let us consider the portfolio studied in Example 2.3.1. The following steps are performed:
# 1. Upload and prepare data,
# 2. Perform random forest analysis, and
# 3. Perform boosting analysis.


# 1. Upload and prepare data
# 1.1. Upload data
library(dplyr)
oneypd_tree <- read.csv('C:\\Users\\bigdata\\Documents\\chap2oneypd.csv', stringsAsFactors=FALSE)
dim(oneypd_tree)
dplyr::glimpse(oneypd_tree)
# Create default flag as per Example 2.3.1
# From 'default_event' derive 'default_indicator' as 'Yes' 'No'
data_whole$default_event

data_whole$default_indicator=ifelse(data_whole$default_event==0,"No","Yes")
dim(data_whole)
# as per Example 2.4.1
# 1.2 Select a subset of variables

oneypd_tree_sel_orig <- data_whole %>%
  dplyr::select('default_indicator', 'default_event','bureau_score',
                'time_since_bankrupt', 'num_ccj', 'time_since_ccj', 'ccj_amount',
                'ltv', 'mob', 'max_arrears_12m', 'max_arrears_bal_6m',
                'avg_bal_6m', 'annual_income', 'loan_balance', 'loan_term','cc_util', 'emp_length', 'months_since_recent_cc_delinq')
# 1.3 Filter out NAs
oneypd_tree_sel <- oneypd_tree_sel_orig %>%
  na.omit(oneypd_tree_sel_orig)
# 1.4 Create stratified samples: 70% train and 30% test
library(caret)
set.seed(123)
train_index <- caret::createDataPartition(oneypd_tree_sel$default_event,p = .7, list = FALSE)
train <- oneypd_tree_sel[ train_index,]
train2 <- train[,2:18]
head(train2)
test <- oneypd_tree_sel[-train_index,]
str(train)

train$default_indicator

# 2. Perform random forest analysis
# 2.1 Fit random forest
library(randomForest)
set.seed(123)
rf_oneypd <- randomForest(default_indicator~.-default_event,
                          data=oneypd_tree_sel[train_index,], mtry=4, ntree=100,
                          importance=TRUE, na.action=na.omit)
# Call:
# Type of random forest: classification
# Number of trees: 100
# No. of variables tried at each split: 4
# OOB estimate of error rate: 4.05%
# 2.2 Variable importance analysis
importance(rf_oneypd)
varImpPlot(rf_oneypd)




# Call:
# Type of random forest: classification
# Number of trees: 100
# No. of variables tried at each split: 4
# OOB estimate of error rate: 4.05%
# 2.2 Variable importance analysis
importance(rf_oneypd)
varImpPlot(rf_oneypd)


# EXAMPLE 2.4.4 ML DISCRIMINATORY POWER ASSESSMENT
# Let us consider random forest Example 2.4.2.What follows demonstrates how validation
# process is performed, based on the following steps:
# 1. ROC analysis (AUC assessment) (see Figure 2.17),
# 2. Kolmogorov–Smirnov (KS) analysis, and
# 3. Gini index.

# 1. ROC Analysis
library(ROCR)
predict_test_orig <- as.matrix(
  predict(rf_oneypd,newdata=oneypd_tree_sel[-train_index,],type="prob"))
predict_test <- as.matrix(predict_test_orig[,2])
oneypd_test <- oneypd_tree_sel[-train_index,"default_indicator"]
actual_test <- as.matrix(ifelse(oneypd_test=="Yes",1,0))
pred_test<- ROCR::prediction(predict_test,actual_test)
perf_test<- ROCR::performance(pred_test, 'tpr', 'fpr')
# 1.1 Plot graphs
plot(perf_test, main='ROC curve test', colorize=T)
abline(0,1, lty =8, col = 'black')
# 1.2 Calculate AUC
auc_test<- ROCR::performance(pred_test, 'auc')
# 0.9410772
# 2. KS Analysis
ks_test<- max(attr(perf_test,'y.values')[[1]]-
                attr(perf_test,'x.values')[[1]])
print(ks_test)
# 0.732483
# 3. Gini index
library(optiRum)
gini_test<- optiRum::giniCoef(predict_test,actual_test)
# 0.8821544



# EXAMPLE 2.4.5 CALIBRATED PD VALIDATION
# Let us consider Example 2.4.3. A validation process for PD calibration is performed,
# based on the following steps:
# 1. Group accounts in bands, and
# 2. Compare average actual and fitted PDs.


# 1. Group accounts in bands
# 1.1. Predict base on model parameters
rf_db_cal$pd<- predict(pd_model, newdata = rf_db_cal,
                       type = 'response')
library(smbinning)
# 1.2. Create bands
score_cust<- (rf_db_cal, y = 'def', x= 'pred',
                              cuts= c(0.2,0.4,0.6,0.8))
rf_db_cal<- smbinning.gen(rf_db_cal, score_cust, chrname ="band")
# 2. Compare average actual and fitted PDs
# 2.1 Calculate mean values


#-----



# Create default flag as per Example 2.3.1
# From 'default_event' derive 'default_indicator' as 'Yes'  'No' as per Example 2.4.1 

# 1.1.1. Glimpse
dplyr::glimpse(oneypd_tree)

# 1.1.2. Date format
library(vars)
oneypd_tree <- dplyr::mutate_at(oneypd_tree, vars(contains('date')),
                                funs(as.Date))
class(oneypd_tree$origination_date)

# 1.1.3. Round arrears count fields
oneypd_tree$max_arrears_12m<- round(oneypd_tree$max_arrears_12m,4)
oneypd_tree$arrears_months<- round(oneypd_tree$arrears_months,4)

# 1.1.4. Default flag definition
oneypd_tree<- dplyr::mutate(oneypd_tree,
                            default_event = if_else(oneypd_tree$arrears_event == 1 |
                                                      oneypd_tree$term_expiry_event == 1 |
                                                      oneypd_tree$bankrupt_event == 1, 1,0))

# 1.1.5. Database split in train and test samples
# Recode default event variables for more convenient use
# 0-default, 1-non-default
oneypd_tree$default_flag<-
  dplyr::if_else(oneypd_tree$default_event == 1,0,1)

default_indicator=ifelse(oneypd_tree$default_event==0,'No','Yes')
oneypd_tree=data.frame(oneypd_tree,default_indicator)







