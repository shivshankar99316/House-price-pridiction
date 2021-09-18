#!/usr/bin/env python
# coding: utf-8

# # Description

# This is a notebook for visualization of various features which the sales price of houses. Then data is taken from the "Kaggle House Price Prediction" challenge.

# # 1. Load Data

# 
# First lets import all the libraries that will be used to load train and test datasets and data manipulation.

# In[1]:


# Import libraries

# Pandas 
import pandas as pd
from pandas import Series,DataFrame 

# Numpy and Matplotlib
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
#sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

# Machine Learning 
from sklearn import preprocessing


# Loading train and test data

# In[2]:


# Get Data in Dataframe 
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# Preview of train and test data

# In[3]:


# head() shows the first 5 rows of the data
train.head()


# In[4]:


test.head()


# There are 1460 entries in the train data set and 1459 entries in test data set. The data contains some NaN values too.

# In[5]:


train.info()


# In[6]:


test.info()


# In[7]:


train.isnull().sum()


# In[8]:


test.isnull().sum()


# In[9]:



train['SalePrice'].describe()


# Sales price is right skewed. So, we perform log transformation so that the skewness is nearly zero.

# In[10]:


# Determining the Skewness of data 
print ("Skew is:", train.SalePrice.skew())

plt.hist(train.SalePrice)
plt.show()

# After log transformation of the data it looks much more center aligned
train['Skewed_SP'] = np.log(train['SalePrice']+1)
print ("Skew is:", train['Skewed_SP'].skew())
plt.hist(train['Skewed_SP'], color='blue')
plt.show()


# In[11]:


sns.factorplot('MSSubClass', 'Skewed_SP', data=train,kind='bar',size=3,aspect=3)
fig, (axis1) = plt.subplots(1,1,figsize=(10,3))
sns.countplot('MSSubClass', data=train)
train['MSSubClass'].value_counts()


# 
# MSSubClass = 60 has highest SalePrice while the sales of houses with MSSubClass = 20 is the highest.

# In[12]:


sns.factorplot('MSZoning', 'Skewed_SP', data=train,kind='bar',size=3,aspect=3)
fig, (axis1) = plt.subplots(1,1,figsize=(10,3))
sns.countplot(x='MSZoning', data=train, ax=axis1)
train['MSZoning'].value_counts()
import warnings
warnings.filterwarnings('ignore')


# In[13]:


sns.factorplot(x='MSZoning', y='SalePrice', col='MSSubClass', data=train, kind='bar', col_wrap=4, aspect=0.8)


# In[14]:


numerical_features = train.select_dtypes(include=[np.number])
numerical_features.dtypes


# In[15]:


# Then we will try to find the corretation between the feature and target
corr = numerical_features.corr()
#print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
#print (corr['SalePrice'].sort_values(ascending=False)[-5:])
print (corr['SalePrice'].sort_values(ascending=False)[:], '\n')


# We will analyze the features in their descending of correlation with sales price

# In[16]:


train.OverallQual.unique()


# In[17]:


#Creating a pivot table 
quality_pivot = train.pivot_table(index='OverallQual',values='SalePrice', aggfunc=np.median)


# In[18]:


quality_pivot


# In[19]:


quality_pivot.plot(kind='bar',color='blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median')
plt.xticks(rotation=0)
plt.show()


# 
# SalePrice varies directly with the Overall quality

# In[20]:


sns.regplot(x='GrLivArea',y='Skewed_SP',data=train)


# 
# SalePrice increases as the GrLivArea increases. We will also get rid of the outliers which severely affect the prediction of the survival rate.

# In[21]:


#Removing outliers
train = train[train['GrLivArea'] < 4000]
sns.regplot(x='GrLivArea',y='Skewed_SP',data=train)


# In[22]:


sns.regplot(x='GarageArea',y='Skewed_SP',data=train)


# 
# GarageArea and SalePrice are directly proportional.
# 
# 
# 
# We will again get rid of the outliers.

# In[23]:


#Removing outliers
train = train[train['GarageArea'] < 1200]
sns.regplot(x='GarageArea',y='Skewed_SP',data=train)


# In[24]:


#Removing the null values
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
nulls


# In[25]:


# Pool null value refers to no pool area
print ("Unique values are:", train.MiscFeature.unique())


# In[26]:


#Analysing the non numeric data 
categoricals = train.select_dtypes(exclude=[np.number])
categoricals.describe(include='all')


# In[27]:


train['Neighborhood'].value_counts().plot(kind='bar')


# In[28]:


g = sns.factorplot(x='Neighborhood', y='Skewed_SP', data=train, kind='bar', aspect=3)
g.set_xticklabels(rotation=90)


# In[29]:


train['Condition1'].value_counts()


# In[30]:



train['Condition2'].value_counts()


# In[31]:


g = sns.factorplot(x='Condition1', y='Skewed_SP', col='Condition2', data=train, kind='bar', col_wrap=4, aspect=0.8)
g.set_xticklabels(rotation=90)


# In[32]:


train['SaleCondition'].value_counts()


# In[33]:


train['SaleType'].value_counts()


# In[34]:


g = sns.factorplot(x='SaleCondition', y='Skewed_SP', col='SaleType', data=train, kind='bar', col_wrap=4, aspect=0.8)
g.set_xticklabels(rotation=90)


# In[35]:


#Data Trasformation 
print ("Original: \n") 
print (train.Street.value_counts(), "\n")


# In[36]:


# Turn into one hot encoding 
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)


# In[37]:


# Encoded 
print ('Encoded: \n') 
print (train.enc_street.value_counts())


# In[38]:


# Feature Engineering
condition_pivot = train.pivot_table(index='SaleCondition',
                                    values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='yellow')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()


# In[39]:


def encode(x): return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)


# In[40]:


condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()


# In[41]:



#Interpolation of data 
data = train.select_dtypes(include=[np.number]).interpolate().dropna()


# In[42]:


sum(data.isnull().sum() != 0)


# In[83]:


# Linear Model for the  train and test
y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)


# In[84]:


print("shape of X ", X.shape)
print("shape of y ", y.shape)


# In[44]:


train.shape


# In[57]:


test.shape


# In[48]:


# Normalize features using min-max scaling.
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()


# In[92]:


from sklearn.model_selection import train_test_split


# In[96]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[ ]:





# In[91]:


X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.25
                                                  )
print("Shape of X_train =", X_train.shape)
print("Shape of y_train =", y_train.shape)
print("Shape of X_test =", X_test.shape)
print("Shape of y_test =", y_test.shape)


# In[ ]:





# In[ ]:





# In[65]:


from sklearn.linear_model import LinearRegression
clf = LinearRegression()


# In[67]:


train.info()


# In[71]:


from sklearn import linear_model
from sklearn import ensemble

#lr =  ensemble.RandomForestRegressor(n_estimators = 100, oob_score = True, n_jobs = -1,random_state =50,max_features = "sqrt", min_samples_leaf = 50)
#lr = linear_model.LinearRegression()
lr = ensemble.GradientBoostingRegressor()
#lr = linear_model.TheilSenRegressor()
#lr = linear_model.RANSACRegressor(random_state=50)


# In[82]:


model = lr.fit(X_train, y_train)


# In[ ]:




