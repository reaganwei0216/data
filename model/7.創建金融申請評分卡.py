

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os
os.chdir('../data')

data=pd.read_csv('data_all_values.csv')



#####################################################
# Step 1: LightGBM重要性选择变量
#####################################################

data_copy=data.copy()

allFeatures = list(data.columns)   
allFeatures.remove('target')
for i in allFeatures:
    print('变量 {} 的不同水平值有 {} 个'.format(i,len(data[i].unique())))     
    
categorical_var=['贷款期限', '贷款等级', '工作年限', '房屋所有权', '收入是否由LC验证', '贷款目的', 
'过去6个月内被查询次数', '留置税数量']

continuous_var=['贷款金额', '利率', '每月还款金额', '年收入', '月负债比', '过去两年借款人逾期30天以上的数字', 
'摧毁公共记录的数量', '额度循环使用率', '总贷款笔数', '拖欠的逾期款项']



'''连续变量标准化'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()                                 
data[continuous_var] = sc.fit_transform(data[continuous_var])   



'''数值分类变量转整型'''      
string_var=list(data.select_dtypes(include=["object"]).columns) 
col=list(set(categorical_var)-set(string_var))
data[col]=data[col].astype(int)



'''字符分类变量按照坏样本率进行编码''' 
def Encoder(df, col, target):           
    encoder = {}
    for v in set(df[col]):
        if v == v:
            subDf = df[df[col] == v]
        else:
            xList = list(df[col])
            nanInd = [i for i in range(len(xList)) if xList[i] != xList[i]]
            subDf = df.loc[nanInd]
        encoder[v] = sum(subDf[target])*1.0/subDf.shape[0]
    newCol = [encoder[i] for i in df[col]]
    return newCol

string_var=list(data.select_dtypes(include=["object"]).columns) 
col=list(set(categorical_var)&set(string_var))

for i in col:
    data[i] = Encoder(data, i, 'target')



'''指定整型分类变量作为lightgbm的分类特征'''
col=list(set(categorical_var)-set(string_var))



'''保存变量和文件'''
import pickle
f =open('lgb_col.pkl','wb')
pickle.dump(col,f)
f.close()  



'''建模'''
allFeatures = list(data.columns)   
allFeatures.remove('target')
X = data[allFeatures]
y = data['target']
from sklearn.cross_validation import train_test_split as sp
X_train, X_test, y_train, y_test = sp(X, y, test_size=0.3, random_state=1)



'''加载分类变量'''
f =open('lgb_col.pkl','rb')
col = pickle.load(f)
f.close()



'''lightgbm建模'''
import lightgbm as LGB
params = {
'objective': 'binary', 
"boosting" : "gbdt",
'num_leaves': 4,    
'min_data_in_leaf': 20,
"subsample": 0.9,
"colsample_bytree": 0.8,
'learning_rate':0.09,
'tree_learner': 'voting',
'metric': 'auc'
        }
dtrain = LGB.Dataset(X_train, y_train, categorical_feature=col)
dtest = LGB.Dataset(X_test, y_test, reference=dtrain, categorical_feature=col)
lgb = LGB.train(params, dtrain, valid_sets=[dtrain, dtest], 
                num_boost_round=3000, early_stopping_rounds=100, verbose_eval=10)



'''lightgbm重要性选择变量'''
importace = list(lgb.feature_importance())
allFeatures=list(lgb.feature_name())
featureImportance = zip(allFeatures,importace)
featureImportanceSorted = sorted(featureImportance, key=lambda k: k[1],reverse=True)

plt.figure(figsize = (5, 10))                                                                                   
sns.barplot(x=[k[1] for k in featureImportanceSorted],y=[k[0] for k in featureImportanceSorted])
plt.xticks(rotation='vertical') 
plt.show()

feature_selection_lgb = [k[0] for k in featureImportanceSorted[:13]]    #选择前13个最重要的变量





#####################################################
# Step 2: 分类变量分箱
#####################################################

data=data_copy.copy()
data=data[feature_selection_lgb+['target']]
data_copy=data.copy()



'''分类变量按照ln(odds)进行编码''' 
def Ln_odds(df, col, target):
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()   
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    N = sum(regroup['total'])  
    B = sum(regroup['bad'])      
    regroup['good'] = regroup['total'] - regroup['bad']
    G = N - B
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x*1.0/B)
    regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
    regroup[col+'_WOE'] = regroup.apply(lambda x: np.log(x.good_pcnt*1.0/x.bad_pcnt),axis = 1)
    df=pd.merge(df, regroup[[col,col+'_WOE']], on=col, how='left')
    return df



'''分类变量分箱''' 
categorical_var=list(set(categorical_var)&set(feature_selection_lgb))
categorical_var=['贷款期限', '房屋所有权', '贷款等级', '过去6个月内被查询次数','工作年限']
for i in categorical_var:   
    data=Ln_odds(data, i, 'target')
    sns.pointplot(x=i, y=i+'_WOE', data=data)      
    plt.xticks(rotation=0)     
    plt.show()



'''逐个调试''' 
data=data_copy.copy()

i='贷款等级'
data[i]=data[i].apply(lambda x: 2 if 2<=x<5 else x)
data[i]=data[i].apply(lambda x: 3 if x>=5 else x)

data=Ln_odds(data, i, 'target')
sns.pointplot(x=i, y=i+'_WOE', data=data)     
plt.xticks(rotation=0)     
plt.show()



i='过去6个月内被查询次数'
data[i]=data[i].apply(lambda x: 4 if x>3 else x)

data=Ln_odds(data, i, 'target')
sns.pointplot(x=i, y=i+'_WOE', data=data)  
plt.xticks(rotation=0)     
plt.show()
    

i='工作年限'
data[i]=data[i].apply(lambda x: 1 if 1<=x<5 else x)
data[i]=data[i].apply(lambda x: 2 if x>=5 else x)

data=Ln_odds(data, i, 'target')
sns.pointplot(x=i, y=i+'_WOE', data=data)     
plt.xticks(rotation=0)     
plt.show()



'''保存文件''' 
col=[]
for i in list(data.columns):
    if i.find('_WOE')<0:
        col.append(i)    
data=data[col]
data.to_csv('分类变量分箱.csv', index=False, encoding='utf-8')





#####################################################
# Step 3: 连续变量分箱
#####################################################

data=pd.read_csv('分类变量分箱.csv')

continuous_var=list(set(continuous_var)&set(feature_selection_lgb))
continuous_var=['总贷款笔数', '每月还款金额', '过去两年借款人逾期30天以上的数字', 
                '贷款金额', '年收入', '利率', '月负债比', '额度循环使用率']

describe=data[continuous_var].describe().T[['max','min']]



'''连续变量分箱''' 
i='总贷款笔数'
sns.distplot(data[i][data['target'] == 0].dropna(),color='blue') 
sns.distplot(data[i][data['target'] == 1].dropna(),color='red')
plt.show() 

bins=[-1, 20, 30, 200]   #左开右闭，如果可能出现0，那么需要以-1开始
cats=pd.cut(list(data[i]), bins, precision=0)        #指定分组区间
cats.value_counts()  
data[i+'组别']=pd.Series(cats)
data=Ln_odds(data, i+'组别', 'target')
sns.pointplot(x=i+'组别', y=i+'组别_WOE', data=data.sort_values(i+'组别_WOE',ascending=False))      
plt.show()  



i='每月还款金额'
sns.distplot(data[i][data['target'] == 0].dropna(),color='blue') 
sns.distplot(data[i][data['target'] == 1].dropna(),color='red')
plt.show() 

bins=[-1, 300, 750, 2000]
cats=pd.cut(list(data[i]), bins, precision=0)        #指定分组区间
cats.value_counts()  
data[i+'组别']=pd.Series(cats)
data=Ln_odds(data, i+'组别', 'target')
sns.pointplot(x=i+'组别', y=i+'组别_WOE', data=data.sort_values(i+'组别_WOE',ascending=False))  
plt.show()  



i='过去两年借款人逾期30天以上的数字'
sns.distplot(data[i][data['target'] == 0].dropna(),color='blue') 
sns.distplot(data[i][data['target'] == 1].dropna(),color='red')
plt.show() 

bins=[-1, 1, 50]
cats=pd.cut(list(data[i]), bins, precision=0)        #指定分组区间
cats.value_counts()  
data[i+'组别']=pd.Series(cats)
data=Ln_odds(data, i+'组别', 'target')
sns.pointplot(x=i+'组别', y=i+'组别_WOE', data=data.sort_values(i+'组别_WOE',ascending=False))  
plt.show()  



i='贷款金额'
sns.distplot(data[i][data['target'] == 0].dropna(),color='blue') 
sns.distplot(data[i][data['target'] == 1].dropna(),color='red')
plt.show() 

bins=[-1, 10000, 20000, 50000]
cats=pd.cut(list(data[i]), bins, precision=0)        #指定分组区间
cats.value_counts()  
data[i+'组别']=pd.Series(cats)
data=Ln_odds(data, i+'组别', 'target')
sns.pointplot(x=i+'组别', y=i+'组别_WOE', data=data.sort_values(i+'组别_WOE',ascending=False))  
plt.show()  



i='年收入'
sns.distplot(data[i][data['target'] == 0].dropna(),color='blue') 
sns.distplot(data[i][data['target'] == 1].dropna(),color='red')
plt.xlim([0,1000000])
plt.show() 

bins=[-1, 150000, 300000, 10000000]
cats=pd.cut(list(data[i]), bins, precision=0)        #指定分组区间
cats.value_counts()  
data[i+'组别']=pd.Series(cats)
data=Ln_odds(data, i+'组别', 'target')
sns.pointplot(x=i+'组别', y=i+'组别_WOE', data=data.sort_values(i+'组别_WOE',ascending=False))  
plt.show()  



i='利率'
sns.distplot(data[i][data['target'] == 0].dropna(),color='blue') 
sns.distplot(data[i][data['target'] == 1].dropna(),color='red')
plt.show() 

bins=[0, 8, 13, 50]
cats=pd.cut(list(data[i]), bins, precision=0)        #指定分组区间
cats.value_counts()  
data[i+'组别']=pd.Series(cats)
data=Ln_odds(data, i+'组别', 'target')
sns.pointplot(x=i+'组别', y=i+'组别_WOE', data=data.sort_values(i+'组别_WOE',ascending=False))  
plt.show()  



i='月负债比'
sns.distplot(data[i][data['target'] == 0].dropna(),color='blue') 
sns.distplot(data[i][data['target'] == 1].dropna(),color='red')
plt.xlim([0,200])
plt.show() 

bins=[-1, 20, 1000]
cats=pd.cut(list(data[i]), bins, precision=0)        #指定分组区间
cats.value_counts()  
data[i+'组别']=pd.Series(cats)
data=Ln_odds(data, i+'组别', 'target')
sns.pointplot(x=i+'组别', y=i+'组别_WOE', data=data.sort_values(i+'组别_WOE',ascending=False))  
plt.show()  



i='额度循环使用率'
sns.distplot(data[i][data['target'] == 0].dropna(),color='blue') 
sns.distplot(data[i][data['target'] == 1].dropna(),color='red')
plt.show() 

bins=[-1, 50, 200]
cats=pd.cut(list(data[i]), bins, precision=0)        #指定分组区间
cats.value_counts()  
data[i+'组别']=pd.Series(cats)
data=Ln_odds(data, i+'组别', 'target')
sns.pointplot(x=i+'组别', y=i+'组别_WOE', data=data.sort_values(i+'组别_WOE',ascending=False))  
plt.show()  



'''保存文件''' 
col=[]
for i in list(data.columns):
    if i.find('_WOE')<0:
        col.append(i)    
data=data[col]
data.to_csv('分箱完成.csv', index=False, encoding='utf-8')





#####################################################
# Step 4: 计算WOE
#####################################################

def CalcWOE(df, col, target):
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()   
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    N = sum(regroup['total']) 
    B = sum(regroup['bad'])     
    regroup['good'] = regroup['total'] - regroup['bad']
    G = N - B
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x*1.0/B)
    regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
    regroup['WOE'] = regroup.apply(lambda x: np.log(x.good_pcnt*1.0/x.bad_pcnt),axis = 1)
    WOE_dict = regroup[[col,'WOE']].set_index(col).to_dict(orient='index')
    for k, v in WOE_dict.items():
        WOE_dict[k] = v['WOE']
    IV = regroup.apply(lambda x: (x.good_pcnt-x.bad_pcnt)*np.log(x.good_pcnt*1.0/x.bad_pcnt),axis = 1)
    IV = sum(IV)
    return {"WOE": WOE_dict, 'IV':IV}


all_var=[]
for i in list(data.columns):
    if i.find('组别')>0:
        all_var.append(i)      
all_var=all_var+categorical_var


WOE_dict = {}
IV_dict = {}
for var in all_var:
    woe_iv = CalcWOE(data, var, 'target')
    WOE_dict[var] = woe_iv['WOE']
    IV_dict[var] = woe_iv['IV']





#####################################################
# Step 5: 逻辑回归建模
#####################################################

'''选取IV>=0.02的变量'''
IV_dict_sorted = sorted(IV_dict.items(), key=lambda x: x[1], reverse=True)
IV_values = [i[1] for i in IV_dict_sorted]
IV_name = [i[0] for i in IV_dict_sorted]

high_IV = {k:v for k, v in IV_dict.items() if v >= 0.02}
high_IV_sorted = sorted(high_IV.items(),key=lambda x:x[1],reverse=True)
print ('总共',len(high_IV_sorted),'个变量IV >= 0.02')  



'''创建WOE变量'''
short_list = high_IV.keys()
short_list_2 = []
for var in short_list:
    newVar = var + '_WOE'
    data[newVar] = data[var].map(WOE_dict[var])
    short_list_2.append(newVar)



'''计算相关系数矩阵，删除与重要业务变量相关性较强的自变量''' 
dataWOE = data[short_list_2]
corr = round(dataWOE.corr(),2)

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True             
plt.figure(figsize = (5, 5))
cmap = sns.diverging_palette(220, 10, as_cmap=True) 
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot =True, cbar_kws={"shrink": .5})
plt.show()



"""选择方差共线性<10的变量"""
col = np.array(data[short_list_2])
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
for i in range(len(short_list_2)):                                              
    print ('{} VIF是{}'.format(short_list_2[i], vif(col, i))) 
        
        

"""判断显著性"""
X = data[short_list_2]
X['intercept'] = [1]*X.shape[0]
y = data['target']

import statsmodels.api as sm
lr_sm=sm.Logit(y, X).fit()
lr_sm.summary()



'''建模'''
X=data[short_list_2]
y=data['target'] 
from sklearn.cross_validation import train_test_split as sp
X_train, X_test, y_train, y_test = sp(X, y, test_size=0.3, random_state=1)


from sklearn.linear_model import LogisticRegression as LR  
lr=LR(random_state=1)
lr.fit(X_train, y_train) 

from sklearn import metrics
y_test_label = lr.predict(X_test)
y_test_value = lr.predict_proba(X_test)[:, 1] 
print("测试集准确率是: {:.2%}".format(metrics.accuracy_score(y_test, y_test_label)))  
print("测试集AUC是: {:.4}".format(metrics.roc_auc_score(y_test, y_test_value)))     


   


#####################################################
# Step 6: 创建评分卡
#####################################################    
    
b=lr.intercept_       #截距

coe=lr.coef_          #系数
a0 = coe[0][0]        #每月还款金额组别 系数          
a1 = coe[0][1]        #贷款金额组别 系数 
a2 = coe[0][2]        #利率组别 系数 
a3 = coe[0][3]        #月负债比组别 系数 
a4 = coe[0][4]        #贷款等级 系数 
a5 = coe[0][5]        #过去6个月内被查询次数 系数 

A = 500
PDO = 20              #每增加20分，odds(好坏比)增加1倍
B=PDO/np.log(2)



'''创建 每月还款金额 单变量得分'''
WOE_dict['每月还款金额组别']  #获取字典key，即变量水平值

woe1 = WOE_dict['每月还款金额组别']['(-1, 300]']
score1 = -(B * a0* woe1) + (A-B*b)/dataWOE.shape[0]

woe2 = WOE_dict['每月还款金额组别']['(300, 750]']
score2 = -(B * a0 * woe2) + (A-B*b)/dataWOE.shape[0]

woe3 = WOE_dict['每月还款金额组别']['(750, 2000]']
score3 = -(B * a0 * woe3) + (A-B*b)/dataWOE.shape[0]


case 
when 0 <= '每月还款金额' <= 300 then 7    
when 300 < '每月还款金额' <= 750 then -1  
when '每月还款金额' > 750 then -11  
else 0                                    #以业务逻辑/补缺规则来定                      



'''创建 贷款金额 单变量得分'''
WOE_dict['贷款金额组别']

woe1 = WOE_dict['贷款金额组别']['(-1, 10000]']
score1 = -(B * a1* woe1) + (A-B*b)/dataWOE.shape[0]

woe2 = WOE_dict['贷款金额组别']['(10000, 20000]']
score2 = -(B * a1 * woe2) + (A-B*b)/dataWOE.shape[0]

woe3 = WOE_dict['贷款金额组别']['(20000, 50000]']
score3 = -(B * a1 * woe3) + (A-B*b)/dataWOE.shape[0]


case 
when 0 <= '贷款金额' <= 10000 then -3    
when 10000 < '贷款金额' <= 20000 then 1  
when '贷款金额' > 20000 then 4  
else 0  



'''创建 利率 单变量得分'''
WOE_dict['利率组别']

woe1 = WOE_dict['利率组别']['(0, 8]']
score1 = -(B * a2* woe1) + (A-B*b)/dataWOE.shape[0]

woe2 = WOE_dict['利率组别']['(8, 13]']
score2 = -(B * a2 * woe2) + (A-B*b)/dataWOE.shape[0]

woe3 = WOE_dict['利率组别']['(13, 50]']
score3 = -(B * a2 * woe3) + (A-B*b)/dataWOE.shape[0]


case 
when 0 < '利率' <= 8 then 25    
when 8 < '利率' <= 13 then 8  
when '利率' > 13 then -8  
else 0  



'''创建 月负债比 单变量得分'''
WOE_dict['月负债比组别']

woe1 = WOE_dict['月负债比组别']['(-1, 20]']
score1 = -(B * a3* woe1) + (A-B*b)/dataWOE.shape[0]

woe2 = WOE_dict['月负债比组别']['(20, 1000]']
score2 = -(B * a3 * woe2) + (A-B*b)/dataWOE.shape[0]


case 
when 0 < '月负债比' <= 20 then 2    
when '月负债比'>20 then -3  
else 0  



'''创建 贷款等级 单变量得分'''
WOE_dict['贷款等级']  

woe1 = WOE_dict['贷款等级'][1]
score1 = -(B * a4* woe1) + (A-B*b)/dataWOE.shape[0]

woe2 = WOE_dict['贷款等级'][2]
score2 = -(B * a4 * woe2) + (A-B*b)/dataWOE.shape[0]

woe3 = WOE_dict['贷款等级'][3]
score3 = -(B * a4 * woe3) + (A-B*b)/dataWOE.shape[0]


case 
when '贷款等级' = 1 then 20    
when '贷款等级' = 2 then 0   
when '贷款等级' = 3 then -17  
else 0       



'''创建 过去6个月内被查询次数 单变量得分'''
WOE_dict['过去6个月内被查询次数']  

woe1 = WOE_dict['过去6个月内被查询次数'][0]
score1 = -(B * a5* woe1) + (A-B*b)/dataWOE.shape[0]

woe2 = WOE_dict['过去6个月内被查询次数'][1]
score2 = -(B * a5 * woe2) + (A-B*b)/dataWOE.shape[0]

woe3 = WOE_dict['过去6个月内被查询次数'][2]
score3 = -(B * a5 * woe3) + (A-B*b)/dataWOE.shape[0]

woe4 = WOE_dict['过去6个月内被查询次数'][3]
score4 = -(B * a5 * woe4) + (A-B*b)/dataWOE.shape[0]

woe5 = WOE_dict['过去6个月内被查询次数'][4]
score5 = -(B * a5 * woe5) + (A-B*b)/dataWOE.shape[0]

case 
when '过去6个月内被查询次数' = 1 then 3    
when '过去6个月内被查询次数' = 2 then -2   
when '过去6个月内被查询次数' = 3 then -5  
when '过去6个月内被查询次数' = 4 then -10 
when '过去6个月内被查询次数' = 5 then -12 
else 0       

