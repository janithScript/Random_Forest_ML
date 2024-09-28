#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


data = pd.read_csv('kyphosis.csv')
data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


x = data.drop('Kyphosis', axis = 1)
x.head()


# In[7]:


y = data['Kyphosis']
y.head()


# In[8]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# In[9]:


x_train.shape


# In[10]:


x_test.shape


# In[12]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 50)
model.fit(x_train, y_train)


# In[13]:


pred = model.predict(x_test)
pred


# In[14]:


y_test


# In[15]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, pred)
accuracy


# In[16]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
cm


# In[17]:


data['Kyphosis'].value_counts()


# In[ ]:




