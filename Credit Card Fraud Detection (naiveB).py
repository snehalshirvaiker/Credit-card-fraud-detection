#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


creditcard_df = pd.read_csv("creditcard.csv")


# In[3]:


creditcard_df.head(10)


# In[4]:


creditcard_df.tail()


# In[5]:


creditcard_df.describe()


# In[6]:


creditcard_df.info()


# In[7]:


non_fraud = creditcard_df[creditcard_df['Class']==0]


# In[8]:


fraud = creditcard_df[creditcard_df['Class']==1]


# In[9]:


fraud


# In[10]:


non_fraud


# In[11]:


print( 'fraud transactions percentage =', (len(fraud) / len(creditcard_df) )*100,"%")


# In[12]:


sns.countplot(creditcard_df['Class'], label = "Count") 
# Data is extremely unbalanced


# In[13]:


plt.figure(figsize=(30,10)) 
sns.heatmap(creditcard_df.corr(), annot=True) 
# Most of the dataset is uncorrelated, its probably because the data is a result of Principal Componenet Analysis (PCA)
# Features V1 to V28 are Principal Components resulted after propagating real features through PCA. 


# In[14]:


column_headers = creditcard_df.columns.values


# In[15]:


# kernel density estimation (KDE) is a non-parametric way to estimate the probability density function of a random variable.
i = 1

fig, ax = plt.subplots(8,4,figsize=(18,30))
for column_header in column_headers:    
    plt.subplot(8,4,i)
    sns.kdeplot(fraud[column_header], bw = 0.4, label = "Fraud", shade=True, color="r", linestyle="--")
    sns.kdeplot(non_fraud[column_header], bw = 0.4, label = "Non Fraud", shade=True, color= "y", linestyle=":")
    plt.title(column_header, fontsize=12)
    i = i + 1
plt.show();


# In[16]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
creditcard_df['Amount_Norm'] = sc.fit_transform(creditcard_df['Amount'].values.reshape(-1,1))


# In[17]:


creditcard_df


# In[18]:


creditcard_df = creditcard_df.drop(['Amount'], axis = 1)


# In[19]:


# Let's drop the target label coloumns
X = creditcard_df.drop(['Class'],axis=1)
y = creditcard_df['Class']


# In[20]:


X


# In[21]:


y


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[24]:


X_train.shape


# In[25]:


y_train.shape


# In[26]:


X_test.shape


# In[27]:


y_test.shape


# In[28]:


from sklearn.naive_bayes import GaussianNB 
NB_classifier = GaussianNB()
NB_classifier.fit(X_train, y_train)


# In[29]:


from sklearn.metrics import classification_report, confusion_matrix


# In[30]:


y_predict_train = NB_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)


# In[31]:


# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)


# In[32]:


print(classification_report(y_test, y_predict_test))


# In[33]:


X = creditcard_df.drop(['Time','V8','V13','V15','V20','V22','V23','V24','V25','V26','V27','V28','Class'], axis = 1)


# In[34]:


X


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
NB_classifier = GaussianNB()
NB_classifier.fit(X_train, y_train)
y_predict = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True)


# In[36]:


print(classification_report(y_test, y_predict))


# In[37]:


print("Number of fraud points in the testing dataset = ", sum(y_test))

