
# coding: utf-8

# In[1]:


import requests
import pandas as pd
from bs4 import BeautifulSoup
# from pandas import Series, DataFrame
from pandas import Series


# In[9]:


# encoding: utf-8

import urllib,urllib3
url = "https://www.cbc.gov.tw/sp.asp?xdurl=gopher/chi/busd/bkrate/interestrate.asp&ctNode=809"
request = urllib.Request(url) 
request.add_header("User-Agent","Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36")


# In[7]:


get_ipython().system(' pip install urllib3')


# In[10]:


import urllib,urllib3


# In[40]:


url = 'https://www.cbc.gov.tw/sp.asp?xdurl=gopher/chi/busd/bkrate/interestrate.asp&ctNode=809'
form_data = {"CompanyNo": "0530000"}
res_post = requests.post(url,data = form_data)
res_post.text
# soup=BeautifulSoup(res_post.text,'html.parser',from_encoding='utf-8')
# soup=BeautifulSoup(res_post.text,'html.parser',from_encoding='iso-8859-8')
# soup=BeautifulSoup(res_post.text,'html.parser',from_encoding='big5')
# soup=BeautifulSoup(res_post.text,'lxml',from_encoding='gb18030')
soup=BeautifulSoup(res_post.text,'lxml',exclude_encodings=['utf-8','iso-gb18030-8','big5'])
print(soup.prettify())


# In[34]:


# https://tnlin.wordpress.com/2017/03/02/%E5%A6%82%E4%BD%95%E7%88%AC%E5%8F%96%E6%B2%92%E6%9C%89%E6%8C%87%E5%AE%9Acharset%E7%9A%84%E7%B6%B2%E9%A0%81%EF%BC%9F/
# http://tech-marsw.logdown.com/blog/2016/01/10/02-post-crawler
# https://blog.gtwang.org/programming/python-beautiful-soup-module-scrape-web-pages-tutorial/
import requests
import chardet
url = "https://www.cbc.gov.tw/sp.asp?xdurl=gopher/chi/busd/bkrate/interestrate.asp&ctNode=809"
form_data = {"CompanyNo": "0050000"}
res_post = requests.post(url,data = form_data)
print(chardet.detect(res_post.content))






# In[37]:


soup = BeautifulSoup(res_post.content, "lxml", from_encoding="utf-8")
print(soup.title.text)


# In[ ]:


url = 'https://www.tgos.tw/TGOS_WEB_API/Sample_Codes/TGOSQueryAddr/QueryAddrGoogleMap.aspx'
    
payload = {
        "__VIEWSTATE":"/wEPDwULLTEwNDI1NTA0NjAPZBYCAgMPZBYCAgcPDxYCHgRUZXh0BY8Kew0KICAiSW5mbyI6IFsNCiAgICB7DQogICAgICAiSXNTdWNjZXNzIjogIlRydWUiLA0KICAgICAgIkluQWRkcmVzcyI6ICLoh7rljJfluILkuK3lsbHljYDmnb7msZ/ot680Njnlt7c06JmfIiwNCiAgICAgICJJblNSUyI6ICJFUFNHOjQzMjYiLA0KICAgICAgIkluRnV6enlUeXBlIjogIuacgOi/kemWgOeJjOiZn+apn+WItiIsDQogICAgICAiSW5GdXp6eUJ1ZmZlciI6ICIwIiwNCiAgICAgICJJbklzT25seUZ1bGxNYXRjaCI6ICJGYWxzZSIsDQogICAgICAiSW5Jc0xvY2tDb3VudHkiOiAiRmFsc2UiLA0KICAgICAgIkluSXNMb2NrVG93biI6ICJGYWxzZSIsDQogICAgICAiSW5Jc0xvY2tWaWxsYWdlIjogIkZhbHNlIiwNCiAgICAgICJJbklzTG9ja1JvYWRTZWN0aW9uIjogIkZhbHNlIiwNCiAgICAgICJJbklzTG9ja0xhbmUiOiAiRmFsc2UiLA0KICAgICAgIkluSXNMb2NrQWxsZXkiOiAiRmFsc2UiLA0KICAgICAgIkluSXNMb2NrQXJlYSI6ICJGYWxzZSIsDQogICAgICAiSW5Jc1NhbWVOdW1iZXJfU3ViTnVtYmVyIjogIkZhbHNlIiwNCiAgICAgICJJbkNhbklnbm9yZVZpbGxhZ2UiOiAiVHJ1ZSIsDQogICAgICAiSW5DYW5JZ25vcmVOZWlnaGJvcmhvb2QiOiAiVHJ1ZSIsDQogICAgICAiSW5SZXR1cm5NYXhDb3VudCI6ICIwIiwNCiAgICAgICJPdXRUb3RhbCI6ICIxIiwNCiAgICAgICJPdXRNYXRjaFR5cGUiOiAi5a6M5YWo5q+U5bCNIiwNCiAgICAgICJPdXRNYXRjaENvZGUiOiAiW+iHuuWMl+W4gl1cdEZVTEw6MSIsDQogICAgICAiT3V0VHJhY2VJbmZvIjogIlvoh7rljJfluIJdXHQgeyDlrozlhajmr5TlsI0gfSDmib7liLDnrKblkIjnmoTploDniYzlnLDlnYAiDQogICAgfQ0KICBdLA0KICAiQWRkcmVzc0xpc3QiOiBbDQogICAgew0KICAgICAgIkZVTExfQUREUiI6ICLoh7rljJfluILkuK3lsbHljYDooYzmlL/ph4wx6YSw5p2+5rGf6LevNDY55be3NOiZnyIsDQogICAgICAiQ09VTlRZIjogIuiHuuWMl+W4giIsDQogICAgICAiVE9XTiI6ICLkuK3lsbHljYAiLA0KICAgICAgIlZJTExBR0UiOiAi6KGM5pS/6YeMIiwNCiAgICAgICJORUlHSEJPUkhPT0QiOiAiMemEsCIsDQogICAgICAiUk9BRCI6ICLmnb7msZ/ot68iLA0KICAgICAgIlNFQ1RJT04iOiAiIiwNCiAgICAgICJMQU5FIjogIjQ2OeW3tyIsDQogICAgICAiQUxMRVkiOiAiIiwNCiAgICAgICJTVUJfQUxMRVkiOiAiIiwNCiAgICAgICJUT05HIjogIiIsDQogICAgICAiTlVNQkVSIjogIjTomZ8iLA0KICAgICAgIlgiOiAxMjEuNTM0MTk3LA0KICAgICAgIlkiOiAyNS4wNjYzMTINCiAgICB9DQogIF0NCn1kZGTEpsBwZvdnDWplcJoPxgbURl2tIA==",
        "__VIEWSTATEGENERATOR": "FE1D42A5",
    "__EVENTVALIDATION": "/wEdAAMVG1R6cPSiWRxdit4Gtrxvzh9mL6dLWI6NyCvVUGLeEOh2ZUG1jMhoL4US5tgdjipApcJcCZ0UDcODf/oT6s7iBb64Ow==",
    "TxtAddress": Address,
    "BtnQuery": "查詢"
}
        
    res_post = requests.post(url,data = payload)
    soup=BeautifulSoup(res_post.text,'html.parser',from_encoding='utf-8')
    items = soup.select('span') [3].text.strip().split('\r\n')
    result = {}
    for item in items:
#         if item[0] =='{' :
# #         if item[0] =='{' or item[0] =="Incorrect syntax near 'WHERE'.":            
#             continue
        parts = item.split(':')
        if len(parts) > 1:
            key = parts[0].strip()
            value = parts[1].strip()
            result[key] = value
#             print('key:', key)
#             print('value:', value)
#             print('---')
        df=pd.DataFrame([result])
        df['buss_addr'] = Series(Address, index=df.index)
#     print(df)
    return df
    sleep(30)

