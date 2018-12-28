# http://codewithzhangyi.com/2018/04/16/%E5%9F%BA%E4%BA%8ER%E7%9A%84%E4%BF%A1%E7%94%A8%E8%AF%84%E7%BA%A7-%E8%AF%84%E5%88%86%E5%8D%A1%E6%A8%A1%E5%9E%8B%E5%88%B6%E4%BD%9C%E6%95%99%E7%A8%8B/
# https://github.com/YZHANG1270/Markdown_pic/blob/master/scoreCard/data0_LR.RData


packages<-c("ggplot2","dplyr","smbinning","data.table","woe","gmodels","ROCR","knitr","reshape2","Information","corrgram","corrplot","varhandle","ROCR","stringr","DT","partykit","tcltk","Daim","vcd","caret")
UsePackages<-function(p){
  if (!is.element(p,installed.packages()[,1])){
    install.packages(p)}
  require(p,character.only = TRUE)}
for(p in packages){
  UsePackages(p)
}


library(data.table)
library(dplyr)
library(ggplot2)
library(reshape2)
library(corrgram)
library(corrplot)
library(stats)
library(smbinning)
library(woe)
library(gmodels)
library(Information)
library(knitr)
library(varhandle)

class(iris$Species)
species <- unfactor(iris$Species)
class(species)

library(ROCR)
library(stringr)
library(DT)
library(partykit)
library(tcltk)
library(Daim)
library(vcd)
library(caret)
options(warn=-1)
  
  # 源数据 data0 在data目录下
  load("C:/Users/0660/Desktop/DL/data0_LR.RData") 
  # 根据历史逾期天数overduedays 增加y变量bad ：逾期超过30天为坏客户，否则好客户
  # data0$bad = ifelse(data0$overduedays>30, 1, 0)
  
  
  summary(data0)
  
  
  
  # create_infotables {Information}
  # 计算dataframe里所有特征的IV值
  IV <- create_infotables(data=data0, y="bad",bins = 10, ncore = NULL,parallel=FALSE)
  # 显示IV计算结果
  (Summary<-IV$Summary)
  str(Summary)
  
  # 绘制每个变量的WOE分箱柱状图
  
  # 筛选变量：留下IV>0.1的变量
  Summary=Summary%>%
    filter(Summary$IV>0.1)%>%
    as.data.frame()
  (selected_names<-Summary$Variable) # 显示筛选后的变量名
  # [1] "ZM_SCORE"                       "IDENTIFICATION_RESULT_Value"   
  # [3] "extration_amount"               "CONTACTS_RELATIVES_COUNT_Value"
  # [5] "POSITION_Value"                 "ZM_SCORE_EXIST"     
  
  num<-length(selected_names) # 筛选后的变量个数
  
  # 绘制每个变量的WOE分箱柱状图
  # plot_infotables {Information}
  names <- selected_names # LOOP for ALL: names<-names(IV$Tables)
  plots <- list()
  IVtable<- IV$Tables
  class(IV)
  for (i in 1:length(selected_names)){
    
    plots[[i]] <- plot_infotables(IV, names[i],same_scales=FALSE,show_values = TRUE)
    IVtable[i]<-IV$Tables$names[i]
  }
  
  
  # Showing the variables whose iv >0.1
  plots[1:length(selected_names)]
  # MultiPlot(IV, IV$Summary$Variable[1:num]) # 绘制综合图code
  IVtable[selected_names]
  
  # 只有extration_amount的WOE分布呈现波浪不规则型，需要整改。
  
  # 相关性分析(CORRplot):这里只先示范协方差矩阵图
  col1 <- colorRampPalette(c("#7F0000","red","#FF7F00","yellow","white",
                             "cyan", "#007FFF", "blue","#00007F"))
  col2 <- colorRampPalette(c("#67001F", "#B2182B", "#D6604D", "#F4A582", "#FDDBC7",
                             "#FFFFFF", "#D1E5F0", "#92C5DE", "#4393C3", "#2166AC", "#053061"))
  col3 <- colorRampPalette(c("red", "white", "blue"))
  col4 <- colorRampPalette(c("#7F0000","red","#FF7F00","yellow","#7FFF7F",
                             "cyan", "#007FFF", "blue","#00007F"))
  wb <- c("white","black")
  par(ask = TRUE)
  
  data0= data0%>%
    select(selected_names,bad)%>%
    as.data.frame()
  summary(data0)
  
  M=data0[complete.cases(data0),]
  M<-cor(M)
  corrplot(M, method="color", col=col3(20), cl.length=21,order = "AOE",tl.cex = 0.6,addCoef.col="grey")
  corrplot(M, tl.cex = 0.5)
  
  # 删去CONTACTS_RELATIVES_COUNT_Value
  data0$CONTACTS_RELATIVES_COUNT_Value <- NULL
  # 循环分箱步骤（分箱调整）-----
  # (1)extration_amount----
  head(data0)
  data0$extration_amount=as.numeric(data0$extration_amount)
  data_tmp=data0%>%
    select(c(extration_amount,bad))%>%
    apply(2,as.numeric)%>%
    data.frame()
  
  
  # library(compare)
  # comparison <- compare(data_tmp,data_tmp2,allowAll=TRUE)
  
  
  IV <- create_infotables(data_tmp, y='bad', ncore=2,bins=5) # bins的数值随意定，一般2~10
  data0$extration_amount=cut(data0$extration_amount,breaks=c(-Inf,475,671,771,971,Inf),labels = IV$Tables$extration_amount$WOE[1:length(IV$Tables$extration_amount$WOE)])
  ggplot(IV$Tables$extration_amount,aes(x=extration_amount,y=WOE))+
    geom_bar(stat='identity',fill='lightblue')
  
  # WOE计算结果保留，在步骤4-Scaling会再次用到
  IV$Tables$extration_amount$WOE
  # [1] -0.4414730 -0.2142640  0.4596698  0.4386113 -0.3501923
  IV$Summary
  # 0.1327355 #IV值
  head(data0)

# (2)POSITION_Value-----
data0$POSITION_Value=as.numeric(data0$POSITION_Value)
data_tmp=data0%>%
  select(c(POSITION_Value,bad))%>%
  apply(2,as.numeric)%>%
  data.frame()
IV <- create_infotables(data_tmp, y='bad', ncore=2,bins=6)
ggplot(IV$Tables$POSITION_Value,aes(x=POSITION_Value,y=WOE))+
  geom_bar(stat='identity',fill='lightblue')

data0$POSITION_Value=cut(data0$POSITION_Value,breaks=c(-Inf,0,1,5),labels = IV$Tables$POSITION_Value$WOE[1:length(IV$Tables$POSITION_Value$WOE)])

# WOE计算结果保留，在步骤4-Scaling会再次用到
IV$Tables$POSITION_Value$WOE
# [1]  0.5817640 -0.2522849 -0.2905931
IV$Summary
# 0.1528563
head(data0)

# (3)ZM_SCORE------
data0$ZM_SCORE=as.numeric(data0$ZM_SCORE)
data_tmp=data0%>%
  select(c(ZM_SCORE,bad))%>%
  apply(2,as.numeric)%>%
  data.frame()
IV <- create_infotables(data_tmp, y='bad', ncore=2,bins=10)

ggplot(IV$Tables$ZM_SCORE,aes(x=ZM_SCORE,y=WOE))+
  geom_bar(stat='identity',fill='lightblue')

data0$ZM_SCORE=cut(data0$ZM_SCORE,breaks=c(-Inf,549,569,592,609,635,Inf),labels = IV$Tables$ZM_SCORE$WOE[1:length(IV$Tables$ZM_SCORE$WOE)])

# WOE计算结果保留，在步骤4-Scaling会再次用到
IV$Tables$ZM_SCORE$WOE
# [1]  0.40926664  0.30817452 -0.01635135
# [4] -0.38743811 -0.74663108 -1.52210534
IV$Summary
# 0.2749328

table(data0$ZM_SCORE_EXIST)
# (4)ZM_SCORE_EXIST-----
data0$ZM_SCORE_EXIST=as.numeric(data0$ZM_SCORE_EXIST)
data_tmp=data0%>%
  select(c(ZM_SCORE_EXIST,bad))%>%
  apply(2,as.numeric)%>%
  data.frame()
IV <- create_infotables(data_tmp, y='bad', ncore=2,bins=2)

ggplot(IV$Tables$ZM_SCORE_EXIST,aes(x=ZM_SCORE_EXIST,y=WOE))+
  geom_bar(stat='identity',fill='lightblue')

# WOE计算结果保留，在步骤4-Scaling会再次用到
IV$Tables$ZM_SCORE_EXIST$WOE
# [1]  0.2976555 -0.3847163
IV$Summary
# 0.1134327

data0$ZM_SCORE_EXIST=cut(data0$ZM_SCORE_EXIST,breaks=c(-Inf,0,1),labels = IV$Tables$ZM_SCORE_EXIST$WOE[1:length(IV$Tables$ZM_SCORE_EXIST$WOE)])

head(data0)

table(data0$IDENTIFICATION_RESULT_Value)
# (5)IDENTIFICATION_RESULT_Value----
data0$IDENTIFICATION_RESULT_Value=as.numeric(data0$IDENTIFICATION_RESULT_Value)
data_tmp=data0%>%
  select(c(IDENTIFICATION_RESULT_Value,bad))%>%
  apply(2,as.numeric)%>%
  data.frame()
IV <- create_infotables(data_tmp, y='bad', ncore=2,bins=5)

ggplot(IV$Tables$IDENTIFICATION_RESULT_Value,aes(x=IDENTIFICATION_RESULT_Value,y=WOE))+
  geom_bar(stat='identity',fill='lightblue')

data0$IDENTIFICATION_RESULT_Value=cut(data0$IDENTIFICATION_RESULT_Value,breaks=c(-Inf,2,3,4),labels = IV$Tables$IDENTIFICATION_RESULT_Value$WOE[1:length(IV$Tables$IDENTIFICATION_RESULT_Value$WOE)])

# WOE计算结果保留，在步骤4-Scaling会再次用到
IV$Tables$IDENTIFICATION_RESULT_Value$WOE
# [1]  0.6084594 -0.2568459 -0.4399638
IV$Summary
# 0.1897237

head(data0)

# 以上的5个变量的IV>0.1,且WOE分布呈Logical Trend，保存数据
data1 = data0 #备份数据，以下都对data1进行处理
head(data1)
str(data1)
data1[, c(1:length(data1))] <- unfactor(data1[, c(1:length(data1))])
cbind(apply(data1,2,function(x)length(unique(x))),sapply(data1,class))


#拆分训练集与测试集，建模
#----------------------------------------------------------
# train & test(80%-20%) select randomly
#----------------------------------------------------------
nrow(data1)
a = round(nrow(data1)*0.8)
b = sample(nrow(data1), a, replace = FALSE, prob = NULL)

data_train= data1[b,]
data_test = data1[-b,]

# 逻辑回归建模
m1=glm(bad~., data=data_train,binomial(link='logit'))
summary(m1)


# 通过检验
# 截距为 1.01769
# 各个系数为 0.99616, 0.61521, 0.69045, 0.24813, -0.41278
# 这些参数都十分重要，在Scaling中再次用到


anova(m1,test="Chisq") # ANOVA 检验通过

model=m1
# y值预测
yhat_train = fitted(model)
yhat_test  = predict(model,newdata=data_test,type='response')

data2 = data1 #数据备份

odds=sum(data2$bad==1)/sum(data2$bad==0)
log(odds,base=exp(1))

B=40/log(2,base=exp(1))
A=200-B*log(odds,base=exp(1))
score=yhat_test*B+A  #Score=258.152490+57.707802*yhat
summary(score)

# 好/坏客户分数分布-----
# 分数越高，客户的逾期风险越高，因此坏客户应该集中在分数偏高区域

index=which(data_test$bad==1)
m=seq(260,300,by=5) #260，300是根据分数值域定的，5为间隔数值
bad=cut(score[index],m)%>%table%>%data.frame
colnames(bad)=c('level','count')
ggplot(data = bad,aes(x =level,y=count)) + geom_bar(stat = 'identity')


# 好客户应该集中在风险分数低分区域。

index=which(data_test$bad==0)
good=cut(score[index],m)%>%table%>%data.frame
colnames(good)=c('level','count')
ggplot(data = good,aes(x =level,y=count)) + geom_bar(stat = 'identity')

# 5. 评估信用评分卡----
# KS检验：模型区分好坏客户的力度
# KS>0.3时，模型才能用。


ks.test(yhat_test[which(data1$bad==0)], yhat_test[which(data1$bad==1)])

# 6. 选择Cut-Off分数-----



