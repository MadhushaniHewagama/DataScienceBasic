#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
get_ipython().run_line_magic('matplotlib', 'inline')

### collecting data

titanic_data = pd.read_csv('train.csv')
print(titanic_data)


# In[7]:


print ('number of passenger in original data: '+str(len(titanic_data.index)))


# In[8]:


### Analyzing Data
sns.countplot(x="Survived", data=titanic_data)


# In[9]:


sns.countplot(x="Survived",hue="Sex" , data=titanic_data)


# In[10]:


sns.countplot(x="Survived",hue="Pclass",data=titanic_data)


# In[11]:


titanic_data["Age"].plot.hist()


# In[14]:


titanic_data["Fare"].plot.hist(bins=20, figsize=(10,5))


# In[15]:


titanic_data.info()


# In[47]:


### Data Wrangling (Data cleaning)

titanic_data.isnull()


# In[18]:


titanic_data.isnull().sum()


# In[22]:


sns.heatmap(titanic_data.isnull(), yticklabels=False, cmap="viridis" )


# In[24]:


sns.boxplot(x="Pclass", y="Age",data=titanic_data)


# In[25]:


titanic_data.head(5)


# In[26]:


titanic_data.drop("Cabin", axis=1, inplace=True)


# In[27]:


titanic_data.head(5)


# In[29]:


titanic_data.dropna(inplace=True)


# In[30]:


sns.heatmap(titanic_data.isnull(), yticklabels=False, cbar=False)


# In[32]:


titanic_data.isnull().sum()


# In[33]:


titanic_data.head(2)


# In[37]:


sex=pd.get_dummies(titanic_data['Sex'],drop_first=True) 
#press shift+tab to more detailes
sex.head(5)


# In[38]:


embark=pd.get_dummies(titanic_data["Embarked"],drop_first=True)
embark.head(5)


# In[40]:


Pcl=pd.get_dummies(titanic_data["Pclass"],drop_first=True)
Pcl.head(5)


# In[41]:


titanic_data=pd.concat([titanic_data,sex,embark,Pcl], axis=1)


# In[42]:


titanic_data.head(5)


# In[43]:


titanic_data.drop(['Sex','Embarked','PassengerId','Name','Ticket'],axis=1,inplace=True)


# In[44]:


titanic_data.head()


# In[45]:


titanic_data.drop('Pclass',axis=1,inplace=True)


# In[46]:


titanic_data.head()


# ### Train Data

# In[52]:


X = titanic_data.drop("Survived",axis=1)
y = titanic_data["Survived"]


# In[53]:


#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split


# In[54]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)


# In[55]:


predictions = logmodel.predict(X_test)


# In[58]:


from sklearn.metrics import classification_report


# In[59]:


classification_report(y_test,predictions)


# In[60]:


from sklearn.metrics import confusion_matrix


# In[61]:


confusion_matrix(y_test,predictions)


# In[62]:


from sklearn.metrics import accuracy_score


# In[63]:


accuracy_score(y_test,predictions)


# In[ ]:




