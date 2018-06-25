
# coding: utf-8

# In[1]:


import pandas as pd
import numpy  as np
import timeit
# https://stackoverflow.com/questions/842059/is-there-a-portable-way-to-get-the-current-username-in-python
import getpass
getpass.getuser()
print(getpass.getuser())
print(type(getpass.getuser()))


# In[94]:


def YYYMM():
    # http://blog.alarmchang.com/?p=230
    import datetime
#     print(datetime.datetime.now())
#     print(datetime.datetime.now().month)
#     print(datetime.datetime.now().year)
    if (x.month <=10):
        return str(datetime.datetime.now().year-1911)+"0"+str(datetime.datetime.now().month-1)
    else:
        return str(datetime.datetime.now().year-1911)+str(datetime.datetime.now().month-1)
YYYMM()
# YYYMM=YYYMM()
# print(YYYMM)
# print(type(YYYMM))


# In[2]:


start = timeit.default_timer()
print(start)


# In[3]:


GAPLF0 = pd.read_csv("GAPLF0.csv")
GAPLF0.head()


# In[4]:


# GAPLF0<- subset(GAPLF0, select =c(ban,company,overdueday))
GAPLF0 = GAPLF0[['ban','company','overdueday']]
GAPLF0.head()


# In[5]:


GCAS = pd.read_csv("GCAS.csv")
GCAS.head()


# In[6]:


# GCAS<- subset(GCAS, datacode %in% c('B') &  sys==31 ,select =c(ban,sys,datacode,grnt,grntret,loan,loanret))
GCAS = GCAS[['ban','sys','datacode','grnt','grntret','loan','loanret']]
GCAS.head()


# In[7]:


GCAS = GCAS[(GCAS['datacode']=='B') & (GCAS['sys'] == 31)].head()


# In[8]:


GCAS['grnt'][0:2]


# In[9]:


# https://stackoverflow.com/questions/32464280/converting-currency-with-to-numbers-in-python-pandas
GCAS['grnt'] = GCAS['grnt'].replace('[\$NT,]', '', regex=True).astype(float)
GCAS['grntret'] = GCAS['grntret'].replace('[\$NT,]', '', regex=True).astype(float)
GCAS['loan'] = GCAS['loan'].replace('[\$NT,]', '', regex=True).astype(float)
GCAS['loanret'] = GCAS['loanret'].replace('[\$NT,]', '', regex=True).astype(float)


# In[10]:


GCAS.head()


# In[11]:


GCAS.info()


# In[12]:


# https://stackoverflow.com/questions/43956335/convert-float64-column-to-int64-in-pandas
GCAS['grnt'] = GCAS['grnt'].astype(np.int64)
GCAS['grntret'] = GCAS['grntret'].astype(np.int64)
GCAS['loan'] = GCAS['loan'].astype(np.int64)
GCAS['loanret'] = GCAS['loanret'].astype(np.int64)


# In[13]:


GCAS.info()


# In[14]:


#逾期案件融資餘額----
#  GCAS_3<-sqldf("SELECT  ban,(loan-loanret) as Loan_sum7 FROM GCAS where datacode='B'",drv="SQLite")
#  head(GCAS_3)
#逾期案件保證餘額----
#  GCAS_4<-sqldf("SELECT ban, (grnt-grntret) as Grnt_sum FROM GCAS where datacode='B'",drv="SQLite")
#  head(GCAS_4)
# merge0= merge(x = GCAS_4, y = GCAS_3, by = "ban", all.x = TRUE)
# head(merge0)
# merge1= merge(x = merge0, y = GAPLF0, by = "ban", all.x = TRUE)
# head(merge1)
# table<-sqldf("SELECT Company,count(Ban) as count ,sum(Loan_sum7) as Loan_sum7 ,sum(Grnt_sum) as Grnt_sum,OverdueDay FROM merge1 group by Ban  order by Grnt_sum desc",drv="SQLite")
# tail(table)


# In[15]:


GCAS['loan_sum7']=GCAS['loan']-GCAS['loanret']
GCAS['grnt_sum']=GCAS['grnt']-GCAS['grntret']


# In[16]:


GCAS.head()


# In[17]:


GAPLF0.head()


# In[18]:


result = pd.merge(GCAS,GAPLF0, left_on='ban', right_on='ban', how='left')
result


# In[ ]:


result


# In[ ]:


# https://itw01.com/RMIE58I.html
# https://github.com/yhat/pandasql
# ! pip install pandasql


# In[34]:


# from pandasql import sqldf
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
# df=pysqldf(q)


# In[38]:


done = pysqldf("SELECT company,count(ban) as count ,sum(loan_sum7) as loan_sum7_123 ,sum(grnt_sum) as grnt_sum_123,overdueday FROM result group by ban  order by grnt_sum desc;")
done


# In[93]:


YYYMM()


# In[95]:


stop = timeit.default_timer()
print(stop)
print(stop - start)
done['yyymm']=YYYMM()
done['runtime']=stop - start
done['name']=getpass.getuser()
done


# In[ ]:


# count(Ban) as count ,sum(Loan_sum7) as Loan_sum7 ,sum(Grnt_sum) as Grnt_sum
result.groupby('ban').sum()


# In[ ]:


# https://medium.com/datainpoint/%E5%BE%9E-pandas-%E9%96%8B%E5%A7%8B-python-%E8%88%87%E8%B3%87%E6%96%99%E7%A7%91%E5%AD%B8%E4%B9%8B%E6%97%85-8dee36796d4a
# count(Ban) as count ,sum(Loan_sum7) as Loan_sum7 ,sum(Grnt_sum) as Grnt_sum
# table<-sqldf("SELECT Company,count(Ban) as count ,sum(Loan_sum7) as Loan_sum7 ,sum(Grnt_sum) as Grnt_sum,OverdueDay FROM merge1 group by Ban  order by Grnt_sum desc",drv="SQLite")
result.groupby(by = 'ban')['loan_sum7'].sum()


# In[ ]:


# table<-sqldf("SELECT Company,count(Ban) as count ,sum(Loan_sum7) as Loan_sum7 ,sum(Grnt_sum) as Grnt_sum,OverdueDay FROM merge1 group by Ban  order by Grnt_sum desc",drv="SQLite")
result.groupby(by = 'ban')['grnt_sum'].sum()


# In[ ]:


result.groupby(by = 'ban')['ban'].count()


# In[ ]:


# table<-sqldf("SELECT Company,count(Ban) as count ,sum(Loan_sum7) as Loan_sum7 ,sum(Grnt_sum) as Grnt_sum,OverdueDay FROM merge1 group by Ban  order by Grnt_sum desc",drv="SQLite")
result.groupby(by = 'ban')['ban'].count()
pd_count = pd.DataFrame(result.groupby(by = 'ban')['ban'].count())
print(type(pd_count))
print(pd_count.values)
result['count']=pd_count.values


# In[ ]:


# del result['aaa']
result


# In[ ]:


# table<-sqldf("SELECT Company,count(Ban) as count ,sum(Loan_sum7) as Loan_sum7 ,sum(Grnt_sum) as Grnt_sum,OverdueDay FROM merge1 group by Ban  order by Grnt_sum desc",drv="SQLite")
# result.groupby(by = 'ban')[['grnt_sum', 'loan_sum7']].sum()
# pd.DataFrame(result.groupby(by = 'ban')[['grnt_sum', 'loan_sum7']].sum())


# In[ ]:


result = result.sort_values('grnt_sum',ascending=False)
result


# In[ ]:


# result[['company','count','loan_sum7','grnt_sum','overdueDay']]


# In[ ]:


result


# In[ ]:


stop = timeit.default_timer()
print(stop)
print(stop - start)
result['runtime']=stop - start
result['name']=getpass.getuser()


# In[ ]:


result[['company','count','loan_sum7','grnt_sum','overdueday','runtime','name']]


# In[ ]:


# Python中pandas函数操作数据库
# https://blog.csdn.net/u011301133/article/details/52488690
# pandas 链接 PostgreSQL 数据库
# http://yuanjun.me/python/pandas-postgresql

