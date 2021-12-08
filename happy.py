#!/usr/bin/env python
# coding: utf-8

# # Happiness project.
# 
# I want to understand the relationship between some country specific indexes like its "Healthy life expectancy" and the people happiness in that country.

# In[1]:


import pandas as pd


# In[2]:


happy = pd.read_csv('2019.csv')


# In[3]:


# Columns and the main countries
happy.head()


# In[4]:


# Data information per column
happy.info()


# **There are 156 countries in this dataset**

# In[5]:


happy.describe()


# In[6]:


# Which is the country with the highest -Healthy life expectancy-?
happy[happy['Healthy life expectancy']==happy['Healthy life expectancy'].max()]['Country or region']


# In[7]:


# Singapore's Healthy life expectancy value
happy['Healthy life expectancy'].max()


# In[8]:


# Which is the country with the highest -Freedom to make life choices-?
happy[happy['Freedom to make life choices']==happy['Freedom to make life choices'].max()]['Country or region']


# In[9]:


# Uzbekistan's Freedom to make life choices value
happy['Freedom to make life choices'].max()


# In[10]:


# Which country has the biggest -GDP per capita-?
happy[happy['GDP per capita']==happy['GDP per capita'].max()]['Country or region']


# In[11]:


# Which country has the smaller GDP per capita
happy[happy['GDP per capita']==happy['GDP per capita'].min()]['Country or region']


# In[12]:


# Which country is the most generous?
happy[happy['Generosity']==happy['Generosity'].max()]['Country or region']


# In[13]:


# Which country is the less generous?
happy[happy['Generosity']==happy['Generosity'].min()]['Country or region']


# In[14]:


# Which country is the most corrupt according to its people perception?
happy[happy['Perceptions of corruption']==happy['Perceptions of corruption'].max()]['Country or region']


# In[15]:


# Which country is the less corrupt according to its people perception?
happy[happy['Perceptions of corruption']==happy['Perceptions of corruption'].min()]['Country or region']


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Important plots to understand the correlation between columns

# In[17]:


# Relation between GDP and Social support
sns.jointplot(x='GDP per capita', y='Social support', kind='reg', data=happy)


# As we can see clearly when a country has a bigger GDP the social support is bigger too.

# In[18]:


# Relation between GDP and Generosity
sns.jointplot(x='GDP per capita', y='Generosity', kind='reg', data=happy)


# There is no relationship between GDP and Generosity. Actually, the data has a plateau that goes down as the GDP increase.

# In[19]:


# Relation between GDP and Perception of corruption
sns.jointplot(x='GDP per capita', y='Perceptions of corruption', kind='reg', data=happy)


# There is a clear increase in the perception of corruption when the GDP increase. The use of the line is just for a rapid view, but the points has a fast increase after GDP=1.25 (as an exponential function).

# In[20]:


# Relation between GDP and Perception of corruption
sns.jointplot(x='GDP per capita', y='Healthy life expectancy', kind='reg', data=happy)


# In a similar manner as the social support plot, there is a clear correlation between GDP per capita and Healthy life expectancy. When the GDP increases the Healthy life expectancy increases.

# In[21]:


# Relation between GDP and Social support (In blue Argentina)
g = sns.jointplot(x='Social support', y='Healthy life expectancy', data=happy, color='grey')
df = happy.loc[happy['Country or region'] == 'Argentina']
g.ax_joint.scatter(x = df['Social support'], y=df['Healthy life expectancy'], color = 'blue', s=100)


# As it was expected, when there is more social support more is the Healthy life expectancy. Argentina is one of the best countries in this subject.

# In[22]:


# A fast view of some special data.
ppf = happy[['GDP per capita', 'Generosity', 'Healthy life expectancy', 'Perceptions of corruption']]
sns.pairplot(ppf, kind='reg')


# In[23]:


sns.heatmap(happy.corr(), annot=True, cmap="YlGnBu")


# In[24]:


sns.distplot(happy['Score'], bins=20)


# In[25]:


happy.columns


# In[26]:


y = happy['Score']


# In[27]:


X = happy[['GDP per capita',
       'Social support', 'Healthy life expectancy',
       'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption']]


# In[28]:


# Create a test data
from sklearn.model_selection import train_test_split


# In[29]:


# Use random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)


# In[30]:


from sklearn.linear_model import LinearRegression


# In[31]:


# Linear regression object
lm = LinearRegression()


# In[32]:


# Fit the training data
lm.fit(X_train,y_train)


# In[33]:


# The coefficients
print('Coefficients: \n', lm.intercept_)


# In[34]:


# The coefficients
print('Coefficients: \n', lm.coef_)


# In[35]:


cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coeff'])
cdf.head()


# ## Predictions

# In[36]:


# We give to the model data that never saw (in our case X_text)
# We train our model with X_train
predictions = lm.predict( X_test)


# In[37]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[38]:


# calculate these metrics by hand!
from sklearn import metrics
import numpy as np

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[39]:


sns.distplot((y_test-predictions),bins=30);


# In[40]:


happy.head()


# In[41]:


# 'GDP','Social', 'Health', Freedom ', 'Generosity',   'Perceptions of corruption']
# Prediction of new data using the model
lm.predict([[1.340, 1.59, 0.99, 0.60, 0.15, 0.39]])


# In[ ]:




