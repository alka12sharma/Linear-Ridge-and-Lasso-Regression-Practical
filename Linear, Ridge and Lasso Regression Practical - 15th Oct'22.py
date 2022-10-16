#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_boston


# In[5]:


boston = load_boston()


# In[6]:


boston


# In[7]:


boston.keys()


# In[9]:


print(boston.DESCR)


# In[10]:


print(boston.data)


# In[11]:


print(boston.target)


# In[12]:


print(boston.feature_names)


# # Let's create the dataframe

# In[13]:


dataset = pd.DataFrame(boston.data, columns = boston.feature_names)


# In[14]:


dataset


# In[15]:


dataset.head()


# In[16]:


dataset['Price'] = boston.target // Add price column to dataset


# In[17]:


dataset.head()


# In[18]:


dataset.info()


# In[19]:


dataset.describe()


# # Check the missing values

# In[20]:


dataset.isnull().sum()


# In[21]:


dataset.corr() // checking correlation between columns 


# In[22]:


sns.pairplot(dataset)


# In[31]:


sns.set(rc={'figure.figsize':(8,6)})


# In[32]:


sns.heatmap(dataset.corr(), annot= True)


# In[33]:


plt.scatter(dataset['CRIM'], dataset['Price'])
plt.xlabel('CRIM Rate')
plt.ylabel('Price Rate')


# In[34]:


plt.scatter(dataset['RM'], dataset['Price'])
plt.xlabel('RM')
plt.ylabel('Price')


# # Creating Regression Plots

# In[35]:


sns.regplot(x='RM', y='Price', data=dataset)


# In[36]:


sns.regplot(x='LSTAT', y='Price', data=dataset)


# In[37]:


sns.regplot(x='CRIM', y='Price', data=dataset)


# # Creating Boxplot

# In[38]:


sns.boxplot(dataset['CRIM'])


# In[39]:


sns.boxplot(dataset['Price'])


# In[40]:


dataset.head()


# Above dataset we can see 13 Independent Features and 1 Dependent Feature (i.e Price )

# # Independent And Dependent Features

# In[42]:


x = dataset.iloc[:,:-1] // we have removed Price column from the dataset


# In[43]:


x


# In[44]:


y = dataset.iloc[:,-1] // showing only Price column


# In[45]:


y


# # Sklearn train_test_split

# In[46]:


from sklearn.model_selection import train_test_split


# In[47]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 10)


# In[49]:


x_train.head()


# In[50]:


x_train.shape


# 339 means number of rows and 13 number of columns (features)

# In[51]:


y_train.head()


# In[52]:


y_train.shape


# 339 means number of rows and only 1 columns i.e Price

# In[53]:


x_test.head()


# In[55]:


x_test.shape


# In[54]:


y_test.head()


# In[56]:


y_test.shape


# # Standardize or feature scaling the datasets

# In[57]:


from sklearn.preprocessing import StandardScaler


# In[58]:


scaler = StandardScaler()


# In[59]:


scaler


# In[60]:


x_train = scaler.fit_transform(x_train)


# In[61]:


x_test = scaler.transform(x_test)


# not using transform in x_test to avoid data leakage

# In[62]:


x_train


# In[63]:


x_test


# # Model Training

# In[64]:


from sklearn.linear_model import LinearRegression


# In[65]:


regression = LinearRegression()


# In[66]:


regression


# In[67]:


regression.fit(x_train, y_train)


# In[69]:


## print the coefficients (Independent (13)) and the intercept (dependent (1))


# In[70]:


print(regression.coef_) // for 13 features


# In[71]:


print(regression.intercept_) // for 1 features


# # Prediction for the test data

# In[72]:


reg_pred = regression.predict(x_test)


# In[73]:


reg_pred


# # Assumptions Of Linear Regression

# In[76]:


plt.scatter(y_test, reg_pred)
plt.xlabel('Test Truth Data')
plt.ylabel('Test Predicate Data')


# # Residuals (It means ERROR)

# In[77]:


residuals = y_test - reg_pred


# In[78]:


residuals


# In[79]:


sns.displot(residuals, kind = 'kde')


# # Scatter plot with predictions and residual
# Uniform distribution

# In[80]:


plt.scatter(reg_pred, residuals)


# # Performance Metrics

# In[81]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# In[82]:


print(mean_squared_error(y_test, reg_pred))


# In[83]:


print(mean_absolute_error(y_test, reg_pred))


# In[84]:


print(np.sqrt(mean_squared_error(y_test, reg_pred)))


# # R square and adjusted R square

# R square

# In[85]:


from sklearn.metrics import r2_score


# In[88]:


score = r2_score(y_test, reg_pred)
print(score)


# Adjusted R square

# In[91]:


1 - (1 - score) * (len(y_test)-1)/(len(y_test) - x_test.shape[1]-1)


# # Ridge Regression

# In[92]:


from sklearn.linear_model import Ridge


# In[93]:


ridge = Ridge()


# In[94]:


ridge


# In[96]:


ridge.fit(x_train, y_train)


# coefficients and the intercept

# In[114]:


print(ridge.coef_)


# In[115]:


print(ridge.intercept_)


# In[101]:


ridge_pred = ridge.predict(x_test)


# In[102]:


ridge_pred 


# In[106]:


plt.scatter(y_test,ridge_pred)
plt.xlabel("Test Truth Data")
plt.ylabel("Test Predicted Data")


# In[116]:


# Residuals


# In[110]:


residual_ridge = y_test - ridge_pred


# In[111]:


residual_ridge


# In[112]:


sns.displot(residual_ridge,kind="kde")


# In[113]:


plt.scatter(ridge_pred, residual_ridge) //uniform distribution


# In[117]:


## Performance Metrics


# In[118]:


print(mean_squared_error(y_test, ridge_pred ))


# In[119]:


print(mean_absolute_error(y_test, ridge_pred ))


# In[120]:


print(np.sqrt(mean_squared_error(y_test, ridge_pred )))


# In[122]:


## R square and adjusted R square


# In[123]:


ridge_score = r2_score(y_test,ridge_pred)


# In[124]:


ridge_score


# In[125]:


## Adjusted R squar


# In[127]:


1 - (1-ridge_score)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)


# # Lasso regression

# In[128]:


from sklearn.linear_model import Lasso


# In[129]:


lasso = Lasso()


# In[130]:


lasso


# In[131]:


lasso.fit(x_train, y_train)


# In[132]:


## print the coefficients and the intercept


# In[133]:


print(lasso.coef_)


# In[134]:


print(lasso.intercept_)


# In[135]:


## Prediction for the test data


# In[136]:


lasso_pred = lasso.predict(x_test)


# In[137]:


lasso_pred


# In[138]:


plt.scatter(y_test,lasso_pred)
plt.xlabel("Test Truth Data")
plt.ylabel("Test Predicted Data")


# In[139]:


## residuals


# In[140]:


residual_lasso =y_test - lasso_pred


# In[141]:


residual_lasso


# In[142]:


sns.displot(residual_lasso, kind = 'kde')


# In[144]:


## Scatter plot with predictions and residual


# In[149]:


plt.scatter(lasso_pred,residual_lasso) //uniform distribution


# In[150]:


## Performance Metrics


# In[152]:


print(mean_squared_error(y_test, lasso_pred))


# In[153]:


print(mean_absolute_error(y_test, lasso_pred))


# In[154]:


print(np.sqrt(mean_squared_error(y_test, lasso_pred)))


# In[155]:


#R square


# In[157]:


lasso_score = r2_score(y_test, lasso_pred)
print(lasso_score)


# In[158]:


## Adjusted R square


# In[160]:


1 - (1-lasso_score)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)

