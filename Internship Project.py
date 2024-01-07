#!/usr/bin/env python
# coding: utf-8

# ### PROJECT: Bank Customer Churn

# 1. OBJECTIVE: The aim of this project to analyze the bank customer's demographics and financial information which inculdes               customer's age, gender. country, credit score, balance and many others to predict whether the customer will                 leave the bank or not

# In[1]:


# Import Library:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Import Data:

df=pd.read_csv(r"C:\Users\Dell\Downloads\Bank Churn Modelling.csv")
df.head()


# In[3]:


df.shape


# Describe Data: Data Preprocessing:

# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.duplicated().sum()


# In[8]:


df.duplicated(['CustomerId']).sum()


# In[10]:


df.dtypes


# In[9]:


df.describe()


# EDA Data Visualization

# In[12]:


# Pie Chart for Customer Churn

plt.figure(figsize=(10,6))
plt.pie(df['Churn'].value_counts(),labels=['No','Yes'],autopct='%1.2f%%')
plt.title('Churn Percentage')
plt.show()


# The pie chart clearly visulaizes the customer churn in the dataset. The majority of the customers in the dataset continue 
# to use the serivces of the bank .

# In[13]:


#gender and customer churn

sns.countplot(x = 'Gender', data = df, hue = 'Churn')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# According to the above plot majority of customers are male.
# we can see that females have more tendency to churn as compared to males

# In[21]:


fig, ax = plt.subplots(figsize=(15, 5))
sns.boxplot(x="Churn", y="CreditScore", data=df)


# In[22]:


# customer location:
sns.countplot(x = 'Geography', hue = 'Churn', data = df)
plt.title('Geography and Churn')
plt.xlabel('Geography')
plt.ylabel('Count')
plt.show()


# Encoding:

# In[23]:


df['Geography'].value_counts()


# In[24]:


df.replace({'Geography':{'France':2, 'Germany':1, 'Spain':0}},inplace=True)


# In[26]:


df['Gender'].value_counts()


# In[27]:


df.replace({'Gender':{'Male':0,'Female':1}},inplace=True)


# In[28]:


df.replace({'Num Of Products':{1:0, 2:1,3:1,4:1}},inplace=True)


# In[29]:


df['Has Credit Card'].value_counts()


# In[30]:


df.loc[(df['Balance']==0), 'Churn'].value_counts()


# In[31]:


df['Zero Balance']=np.where(df['Balance']>0,1,0)


# In[32]:


df['Zero Balance'].hist()


# In[33]:


df.groupby(['Churn','Geography']).count()


# Define Target Variable (y) and Feature Variables (X)

# In[34]:


df.columns


# In[39]:


x= df.drop(['CustomerId','Surname','Churn'], axis=1)
x


# In[40]:


y=df['Churn']
y


# In[37]:


x.shape , y.shape


# train test split

# In[38]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[41]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# #### Logistic Regression Model:

# In[42]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[43]:


X_train=scaler.fit_transform(x_train)
X_test=scaler.transform(x_test)


# In[44]:


from sklearn.linear_model import LogisticRegression
LGR=LogisticRegression()


# In[45]:


LGR.fit(X_train,y_train)


# In[46]:


y_pred=LGR.predict(X_test)


# In[47]:


y_pred


# In[48]:


y_test


# In[49]:


from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score,classification_report


# In[50]:


accuracy_score(y_pred,y_test)


# In[51]:


confusion_matrix(y_pred,y_test)


# In[52]:


precision_score(y_pred,y_test)


# In[53]:


f1_score(y_pred,y_test)


# In[55]:


print(classification_report(y_pred,y_test))


# #### Decision Tree Classifier Algorithm:

# In[58]:


from sklearn.tree import DecisionTreeClassifier


# In[59]:


DT=DecisionTreeClassifier()


# In[60]:


DT.fit(x_train,y_train)


# In[61]:


y_pred=DT.predict(x_test)
y_pred


# In[62]:


y_test


# In[63]:


from sklearn.metrics import accuracy_score


# In[64]:


accuracy_score(y_pred,y_test)


# In[ ]:





# Using GridSearchCV to find the best parameters for the model.

# In[66]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
#creating Decision Tree Classifer object
dtree = DecisionTreeClassifier()

#defining parameter range
param_grid = {
    'max_depth': [2,4,6,8,10,12,14,16,18,20],
    'min_samples_leaf': [1,2,3,4,5,6,7,8,9,10],
    'criterion': ['gini', 'entropy'],
    'random_state': [0,42]
    }

#Creating grid search object
grid_dtree = GridSearchCV(dtree, param_grid, cv = 5, scoring = 'roc_auc', n_jobs = -1, verbose = 1)

#Fitting the grid search object to the training data
grid_dtree.fit(X_train, y_train)

#Printing the best parameters
print('Best parameters found: ', grid_dtree.best_params_)


# In[67]:


#Adding the parameters to the model

dtree = DecisionTreeClassifier(criterion='gini', max_depth=6, random_state=42, min_samples_leaf=10)
dtree


# In[68]:


#training the model
dtree.fit(X_train,y_train)
#training accuracy
dtree.score(X_train,y_train)


# Both the models have nearly equal accuracy score.But, the Random Forest Classifier has a better accuracy and precision score than the logistic Regression.
