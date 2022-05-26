#!/usr/bin/env python
# coding: utf-8

# # 5.4 平行化(Paralleloze自助聚合法)

# In[1]:


# get_ipython().run_line_magic('pwd', '')


# In[2]:


# --- 第 1 部分 ---
# 載入函式庫與資料集
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np
from concurrent.futures import ProcessPoolExecutor
digits = load_digits()

np.random.seed(1)
train_size = 1500
train_x = digits.data[:train_size]
train_y = digits.target[:train_size]
test_x = digits.data[train_size:]
test_y = digits.target[train_size:]


# In[3]:


# --- 第 2 部分 ---
def create_learner(train_x, train_y):
    # 產生子樣本
    bootstrap_sample_indices = np.random.randint(0, train_size, size=train_size)
    bootstrap_x = train_x[bootstrap_sample_indices]
    bootstrap_y = train_y[bootstrap_sample_indices]
    # 訓練基學習器
    dtree = DecisionTreeClassifier()
    dtree.fit(bootstrap_x, bootstrap_y)
    return dtree

def predict(learner, test_x):
    return learner.predict(test_x)


# In[4]:


# --- 第 3 部分 ---
if __name__ == '__main__':

    ensemble_size = 1000
    base_learners = []

    # 利用平行運算建立基學習器
    with ProcessPoolExecutor() as executor:
        futures = []
        for _ in range(ensemble_size):
            future = executor.submit(create_learner, train_x, train_y)
            futures.append(future)

        for future in futures:
            base_learners.append(future.result())

    # 產生基學習器的預測值
    base_predictions = []
    base_accuracy = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for learner in base_learners:
            future = executor.submit(predict, learner, test_x)
            futures.append(future)

        for future in futures:
            predictions = future.result()
            base_predictions.append(predictions)
            acc = metrics.accuracy_score(test_y, predictions)
            base_accuracy.append(acc)


# In[ ]:


# --- 第 5 部分 ---
# 產生集成後預測並計算準確率
ensemble_predictions = []
# 找出每一筆資料得票最多的類別
for i in range(len(test_y)):
    # 計算每個類別的得票數
    counts = [0 for _ in range(10)]
    for learner_p in base_predictions:
        counts[learner_p[i]] = counts[learner_p[i]]+1

    # 找到得票最多的類別
    final = np.argmax(counts)
    # 將此類別加入最終預測中
    ensemble_predictions.append(final)

ensemble_acc = metrics.accuracy_score(test_y, 
                                      ensemble_predictions)


# In[ ]:


# --- 第 6 部分 ---
# 顯示準確率，從小到大依序印出來
print('Base Learners:')
print('-'*30)
for index, acc in enumerate(sorted(base_accuracy)):
    print(f'Learner {index+1}: %.2f' % acc)
print('-'*30)
print('Bagging: %.2f' % ensemble_acc)


# In[ ]:




