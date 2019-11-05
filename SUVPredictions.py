#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataset=pd.read_csv("SUV_Predictions.csv")


# In[3]:


dataset.head(10)


# In[4]:


X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values


# In[8]:



y


# In[10]:


from sklearn.model_selection import train_test_split


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=0)


# In[13]:


from sklearn.preprocessing import StandardScaler


# In[14]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[15]:


from sklearn.linear_model import LogisticRegression


# In[16]:


classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)


# In[17]:


y_pred = classifier.predict(X_test)


# In[18]:


from sklearn.metrics import accuracy_score


# In[19]:


accuracy_score(y_test,y_pred)*100


# In[ ]:




