#!/usr/bin/env python
# coding: utf-8

# Modify the notebook to answer the following:
# 
# Take the Boston Housing Data. 
# We want to predict median house price. Select the most important features.
# Divide the data into training and testing. Fit a linear model.
# Which variable have a strong relation to median price.
# Plot the predicted values of the test data. 
# 
# Select the most important features
# 
# Which features have the strong relation to median price. 
# 
# Fit a linear model 
# 
# Divide the data into training and testing
# 
# Plot the predicted values of the test data.
# 
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import datasets
import seaborn as sns


# In[2]:


boston = datasets.load_boston() 
boston.keys()


# In[3]:


#from sklearn.datasets import load_boston
boston = datasets.load_boston() 
df_x = pd.DataFrame(boston.data, columns = boston.feature_names)
df_y = pd.DataFrame(boston.target, columns = ['price'])


# In[4]:


df_x.head(8)


# In[5]:


df_y.head()


# ### Features 
# 
# CRIM: Per capita crime rate by town
# 
# ZN: Proportion of residential land zoned for lots over 25,000 sq. ft
# 
# INDUS: Proportion of non-retail business acres per town
# 
# CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# 
# NOX: Nitric oxide concentration (parts per 10 million)
# 
# RM: Average number of rooms per dwelling
# 
# AGE: Proportion of owner-occupied units built prior to 1940
# 
# DIS: Weighted distances to five Boston employment centers
# 
# RAD: Index of accessibility to radial highways
# 
# TAX: Full-value property tax rate per $10,000
# 
# PTRATIO: Pupil-teacher ratio by town
# 
# B: 1000(Bk — 0.63)², where Bk is the proportion of [people of African American descent] by town
# 
# LSTAT: Percentage of lower status of the population
# 
# MEDV: Median value of owner-occupied homes in $1000s

# We see that we have 506 observations and 13 attributes. The goal is to predict the price of the house using the features given. 

# Feature name

# In[6]:


print(df_x.columns)


# convert df_bos.data into pandas data frame

# In[7]:


df_x.describe()


# In[8]:


def plotFeatures(col_list,title):
    plt.figure(figsize=(10, 14))
    i = 0
    print(len(col_list))
    for col in col_list:
        i+=1
        plt.subplot(7,2,i)
        plt.plot(df_x[col],df_y,marker='.',linestyle='none')
        plt.title(title % (col))   
        plt.tight_layout()
colnames = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
plotFeatures(colnames,"Relationship bw %s and price")


# These graphs gives a clear intuition of relationship of target variable with features. We can see that RM(Avg. rooms per dwelling) has very linear relationship with price.
# 
# Another option to see the relationship is to plot the correlation of features and target variable with each other using heatmap of seaborn.This is much more descriptive also.

# In[9]:


import seaborn as sns
fig = plt.subplots (figsize = (10,10))
sns.set (font_scale = 1.5)
sns.heatmap (df_x.corr (), square = True, cbar = True, annot = True, annot_kws = {'size': 10})
plt.show ()


# In[10]:


df = pd.concat([df_x, df_y],axis = 1,sort = True)
fig = plt.subplots (figsize = (10,10))
sns.set (font_scale = 1.5)
sns.heatmap (df.corr (), square = True, cbar = True, annot = True, annot_kws = {'size': 10})
plt.show ()


# In[11]:


df.corr()


# In[12]:


df[df.columns[:]].corr()['price'][:].sort_values(ascending=False)


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=42)


# In[14]:


## Finding p-values using statmodels

Xs = X_train.values
y = y_train.values


# In[15]:


import statsmodels.api as sm 
X2 = sm.add_constant(Xs)
est = sm.OLS(y,X2)
est2 = est.fit()
print(est2.summary())


# # Which features have a strong relationship to median price?
# 
# The feature RM(0.695360) has the strongest relationship to price as evident from the .corr values above and the graph plot
# RM has the strongest positive correlationship with the price
# 
# The other features that have a strong relationship to price (.corr()>0.4) are:
# 
# LSTAT (-0.737663) -> (LSTAT has the strongest negative correlation with price),
# TAX       -0.468536,
# INDUS     -0.483725,
# PTRATIO   -0.507787

# In[16]:


sns.set(rc={'figure.figsize':(16,9)})
sns.distplot(df['price'], bins=30)
plt.show()


# In[17]:


sns.set(rc={'figure.figsize':(16,9)})
sns.distplot(np.log(df['price']), bins=30)
plt.show()


# In[18]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=42)


# In[19]:


from sklearn.linear_model import LinearRegression  
linear_model = LinearRegression()  
linear_model.fit(X_train, y_train)


# In[20]:


# Intercept
print(linear_model.intercept_)


# In[21]:


X_train.head()


# In[22]:


# The coefficients
print('Coefficients: \n', linear_model.coef_)


# In[23]:


y_pred = linear_model.predict(X_test)


# In[24]:


# Explained variance score: 1 is perfect  R^2  
print('Variance Score: %0.2f' % linear_model.score(X_train, y_train))


# In[25]:


# Adjusted R^2 
adjusted_r_squared = 1 - (1-linear_model.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
adjusted_r_squared


# In[26]:


# Explained variance score: 1 is perfect 
print('Variance Score: %0.2f' % linear_model.score(X_test, y_test))


# In[27]:


from sklearn import metrics  
print('Mean Absolute Error:',  metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error: ',  metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:',  np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


##pd.DataFrame(zip(X.columns, linear_model.coef_), columns = ['features', 'estimatedCoefficients'])


# In[28]:


# Compute R^2 and adjusted R^2 with formulas 
yhat = linear_model.predict(X_train)
SS_Residual = sum((np.asarray(y_train)-yhat)**2)
SS_Total = sum(((np.asarray(y_train))-np.mean(np.asarray(y_train)))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
print(r_squared, adjusted_r_squared)


# In[29]:


#Residual plot
plt.scatter(linear_model.predict(X_train), linear_model.predict(X_train)-y_train, c= 'b', alpha = 0.5)
plt.scatter(linear_model.predict(X_test), linear_model.predict(X_test)-y_test, c= 'g')
plt.hlines(y = 0, xmin = 0, xmax = 60)
plt.title('Residual plot using training (blue) and test(green) data')
plt.ylabel('Residuals')
plt.xlabel('Fitted values')


# In[30]:


# plot of predicted and test 

plt.scatter(y_pred, y_test)


# In[31]:


#predicted values of test data
y_pred
plt.plot(y_pred)


# # Most important feature
# 
# - Based on the above coefficients, RM has the highest importance
# - The other important features based on the coefficients are: CHAS(2.78443820e+00), RAD(2.62429736e-01)

# # The training with the dropped variables

# # Eliminating variables:
# 
# - the features INDUS , #ZN, #TAX and AGE have p vakues higher than 0.05
# - RAD or TAX have a high correlation from the seaborn map and one of it can be dropped. RAD is removed
# - The features dropped based on weak correlation of 0.4 are RAD, AGE, ZN, B, DIS, CHAS

# In[32]:


df_sub= df.drop(['RAD','AGE','ZN','B', 'DIS','CHAS','INDUS'], axis =1)


# In[33]:


df_sub.shape


# In[34]:


X_sub = df_sub.drop('price', axis =1)
X_sub


# In[35]:


y_sub = df_sub['price']
y_sub


# In[36]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)


# In[37]:


from sklearn.linear_model import LinearRegression  
linear_model1 = LinearRegression()  
linear_model1.fit(X_train, y_train)


# In[38]:


# Intercept
print(linear_model1.intercept_)


# In[39]:


X_train.head()


# In[40]:


y_pred = linear_model1.predict(X_test)


# In[41]:


# Explained variance score: 1 is perfect  R^2  
print('Variance Score: %0.2f' % linear_model1.score(X_train, y_train))


# In[42]:


# Adjusted R^2 
adjusted_r_squared = 1 - (1-linear_model1.score(X_train, y_train))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
adjusted_r_squared


# In[43]:


# Explained variance score: 1 is perfect 
print('Variance Score: %0.2f' % linear_model1.score(X_test, y_test))


# In[44]:


from sklearn import metrics  
print('Mean Absolute Error:',  metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error: ',  metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:',  np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


##pd.DataFrame(zip(X.columns, linear_model.coef_), columns = ['features', 'estimatedCoefficients'])


# In[45]:


# Compute R^2 and adjusted R^2 with formulas 
yhat = linear_model1.predict(X_train)
SS_Residual = sum((np.asarray(y_train)-yhat)**2)
SS_Total = sum(((np.asarray(y_train))-np.mean(np.asarray(y_train)))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
print(r_squared, adjusted_r_squared)


# In[46]:


#predicted values of test data
y_pred
plt.plot(y_pred)

