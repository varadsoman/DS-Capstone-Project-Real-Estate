#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Importing module
import warnings
# Warnings filter.
warnings.filterwarnings('ignore')
# Import the necessary libraries
import plotly.offline as pyo
import plotly.graph_objs as go
# Set notebook mode to work in offline
pyo.init_notebook_mode()


# In[2]:


train=pd.read_csv("C:/Users/91987/OneDrive/Desktop/Varad DS Capstone Project/Project 1/train.csv")
test=pd.read_csv("C:/Users/91987/OneDrive/Desktop/Varad DS Capstone Project/Project 1/test.csv")


# # Descriptive Analysis

# In[3]:


train.head()


# In[4]:


test.head()


# In[5]:


train.describe()


# In[6]:


test.describe()


# In[7]:


train.columns


# In[8]:


test.columns


# In[9]:


# UID is unique userID value in the train and test dataset. So an index can be created from the UID feature
train.set_index(keys=['UID'],inplace=True)#Set the DataFrame index using existing columns.
test.set_index(keys=['UID'],inplace=True)


# In[10]:


# Handling Missing value
train.isnull().sum()/len(train)*100


# In[11]:


test.isnull().sum()/len(test)*100


# In[12]:


## Remove columns from train and test which are blank or have one value
train=train.drop(['BLOCKID','SUMLEVEL'],axis=1)
test=test.drop(['BLOCKID','SUMLEVEL'],axis=1)


# In[13]:


## Imputing  missing values with mean
# search columns which have a missing values
missing_train_cols=[]
for col in train.columns:
    if train[col].isna().sum() !=0:
         missing_train_cols.append(col)
print(missing_train_cols)


# In[14]:


missing_test_cols=[]
for col in test.columns:
    if test[col].isna().sum() !=0:
         missing_test_cols.append(col)
print(missing_test_cols)


# In[15]:


## Missing columns are numeric, replace the nulls with mean
# for train
for col in train.columns:
    if col in (missing_train_cols):
        train[col].replace(np.nan,train[col].mean(),inplace=True)

# for test
for col in test.columns:
    if col in (missing_test_cols):
        test[col].replace(np.nan,test[col].mean(),inplace=True)


# In[16]:


train.isna().sum().sum()
test.isna().sum().sum()


# In[17]:


## write cleansed file for Tableau dashboard
train.to_csv("C:/Users/91987/OneDrive/Desktop/Varad DS Capstone Project/Project 1/train_Cleansed.csv")


# # Week 1 : Exploratory Data Analysis (EDA):

# In[18]:


df = train[train['pct_own']>0.1]
df.shape


# In[19]:


df = df.sort_values(by='second_mortgage',ascending=False)
pd.set_option('display.max_columns', None)
df.head()


# In[20]:


top_2500_second_mortgage_pctown_10 = df.head(2500)
top_2500_second_mortgage_pctown_10


# In[21]:


import plotly.express as px
import plotly.graph_objects as go


# In[22]:


# Visualization 1 (Geo-Map):
fig = go.Figure(data=go.Scattergeo(
    lat = top_2500_second_mortgage_pctown_10['lat'],
    lon = top_2500_second_mortgage_pctown_10['lng']),
    )
fig.update_layout(
    geo=dict(
        scope = 'north america',
        showland = True,
        landcolor = "rgb(212, 212, 212)",
        subunitcolor = "rgb(255, 255, 255)",
        countrycolor = "rgb(255, 255, 255)",
        showlakes = True,
        lakecolor = "rgb(255, 255, 255)",
        showsubunits = True,
        showcountries = True,
        resolution = 50,
        projection = dict(
            type = 'conic conformal',
            rotation_lon = -100
        ),
        lonaxis = dict(
            showgrid = True,
            gridwidth = 0.5,
            range= [ -140.0, -55.0 ],
            dtick = 5
        ),
        lataxis = dict (
            showgrid = True,
            gridwidth = 0.5,
            range= [ 20.0, 60.0 ],
            dtick = 5
        )
    ),
    title='Top 2,500 locations with second mortgage is the highest and percent ownership is above 10 percent')
fig.show()


# In[23]:


# Use the following bad debt equation:
#0ad Debt = P (Second Mortgage ∩ Home Equity Loan)
#Bad Debt = second_mortgage + home_equity - home_equity_second_mortgage
train['bad_debt']=train['second_mortgage']+train['home_equity']-train['home_equity_second_mortgage']


# In[24]:


#Create pie charts  to show bad debt
train['bins_bad_debt'] = pd.cut(train['bad_debt'],bins=[0,0.1,.5,1], labels=["less than 10%","10-50%","50-100%"])
train.groupby(['bins_bad_debt']).size().plot(kind='pie',subplots=True,startangle=90, autopct='%1.1f%%')
plt.title('Bad Debt pct')
plt.ylabel("")

plt.show()


# In[25]:


#Create pie charts  to show overall debt
train['bins_debt'] = pd.cut(train['debt'],bins=[0,0.1,.5,1], labels=["less than 10%","10-50%","50-100%"])
train.groupby(['bins_debt']).size().plot(kind='pie',subplots=True,startangle=90, autopct='%1.1f%%')
plt.title('Debt pct')
plt.ylabel("")

plt.show()


# In[26]:


cols=['second_mortgage','home_equity','debt','bad_debt']
df_box_hamilton=train.loc[train['city'] == 'Hamilton']
df_box_manhattan=train.loc[train['city'] == 'Manhattan']
df_box_city=pd.concat([df_box_hamilton,df_box_manhattan])
df_box_city.head(4)


# In[27]:


# Visualization 4:
plt.figure(figsize=(10,5))
sns.boxplot(data=df_box_city,x='second_mortgage', y='city',width=0.5,palette="Set3")
plt.show()


# In[28]:


# Visualization 5:
plt.figure(figsize=(10,5))
sns.boxplot(data=df_box_city,x='home_equity', y='city',width=0.5,palette="Set3")
plt.show()


# In[29]:


# Visualization 6:
plt.figure(figsize=(10,5))
sns.boxplot(data=df_box_city,x='debt', y='city',width=0.5,palette="Set3")
plt.show()


# In[30]:


# Visualization 7:
plt.figure(figsize=(10,5))
sns.boxplot(data=df_box_city,x='bad_debt', y='city',width=0.5,palette="Set3")
plt.show()


# In[31]:


# Visualization 8:
sns.distplot(train['hi_mean'])
plt.title('Household income distribution chart')
plt.show()


# In[32]:


# Visualization 9:
sns.distplot(train['family_mean'])
plt.title('Family income distribution chart')
plt.show()


# In[33]:


# Visualization 10:
sns.distplot(train['family_mean']-train['hi_mean'])
plt.title('Remaining income distribution chart')
plt.show()


# In[ ]:


#####


# # Week 1 EDA 

# In[34]:


# Visualization 11:
sns.histplot(train['pop'])
plt.title('Population distribution chart')
plt.show()


# In[35]:


# Visualization 12:
sns.histplot(train['male_pop'])
plt.title('Male population distribution chart')
plt.show()


# In[36]:


# Visualization 13:
sns.histplot(train['female_pop'])
plt.title('Female population distribution chart')
plt.show()


# In[37]:


# Visualization 14:
sns.histplot(train['male_age_median'])
plt.title('Male age distribution chart')
plt.show()


# In[38]:


# Visualization 15:
sns.histplot(train['female_age_median'])
plt.title('Female age distribution chart')
plt.show()


# In[39]:


train["pop_density"]=train["pop"]/train["ALand"]
test["pop_density"]=test["pop"]/test["ALand"]


# In[40]:


# Visualization 16:
sns.distplot(train['pop_density'])
plt.title('Population density distribution chart')
plt.show()


# In[41]:


# Visualization 17:
sns.boxplot(train['pop_density'])
plt.title('Population density distribution chart')
plt.show()


# In[42]:


train["median_age"]=(train["male_age_median"]+train["female_age_median"])/2
test["median_age"]=(test["male_age_median"]+test["female_age_median"])/2
train[['male_age_median','female_age_median','male_pop','female_pop','median_age']].head()


# In[43]:


# Visualization 18:
sns.histplot(train['median_age'])
plt.title('Age median distribution chart')
plt.show()


# In[44]:


train["pop"].describe()


# In[45]:


train['pop_bins']=pd.cut(train['pop'],bins=5,labels=['very low','low','medium','high','very high'])
train[['pop','pop_bins']]


# In[46]:


train['pop_bins'].value_counts()


# In[47]:


train.groupby(by='pop_bins')[['married','separated','divorced']].count()


# In[48]:


train.groupby(by='pop_bins')[['married','separated','divorced']].agg(["mean", "median"])


# In[49]:


# Visualization 19:
pop_bin_married=train.groupby(by='pop_bins')[['married','separated','divorced']].agg(["mean"])
sns.lineplot(data=pop_bin_married)
plt.show()


# In[50]:


rent_state_mean=train.groupby(by='state')['rent_mean'].agg(["mean"])
rent_state_mean.head()


# In[51]:


income_state_mean=train.groupby(by='state')['family_mean'].agg(["mean"])
income_state_mean.head()


# In[52]:


rent_perc_of_income=rent_state_mean['mean']/income_state_mean['mean']
rent_perc_of_income.head(10)


# In[53]:


#overall level rent as a percentage of income
sum(train['rent_mean'])/sum(train['family_mean'])


# In[54]:


#Correlation analysis and heatmap
train[["COUNTYID","STATEID","zip_code", "type","pop","family_mean",'second_mortgage', 'home_equity', 'debt','hs_degree','median_age','pct_own', 'married','separated', 'divorced']].corr()


# In[55]:


# Visualization 20:
sns.heatmap(train[["COUNTYID","STATEID","zip_code", "type","pop","family_mean",'second_mortgage', 'home_equity', 'debt','hs_degree','median_age','pct_own', 'married','separated', 'divorced']].corr())


# In[118]:


###


# # Week 2 : Data Pre-processing:
# 
# The economic multivariate data has a significant number of measured variables. The goal is to find where the measured variables depend on a number of smaller unobserved common factors or latent variables. 
# 
# Each variable is assumed to be dependent upon a linear combination of the common factors, and the coefficients are known as loadings. Each measured variable also includes a component due to independent random variability, known as “specific variance” because it is specific to one variable. Obtain the common factors and then plot the loadings. Use factor analysis to find latent variables in our dataset and gain insight into the linear relationships in the data. 
# 
#        Following are the list of latent variables:
# 
# Highschool graduation rates
# 
# Median population age
# 
# Second mortgage statistics
# 
# Percent own
# 
# Bad debt expense

# In[56]:


from sklearn.decomposition import FactorAnalysis


# In[57]:


fa = FactorAnalysis(n_components=5,random_state=11)


# In[58]:


train_transformed = fa.fit_transform(train.select_dtypes(exclude=('object','category')))


# In[59]:


train_transformed.shape


# In[60]:


train_transformed


# # Data Modeling :
# 
# Build a linear Regression model to predict the total monthly expenditure for home mortgages loan. 
# 
#        Please refer deplotment_RE.xlsx. Column hc_mortgage_mean is predicted variable. This is the mean monthly mortgage and owner costs of specified geographical location.
# 
#        Note: Exclude loans from prediction model which have NaN (Not a Number) values for hc_mortgage_mean. 
# 
#        a) Run a model at a Nation level. If the accuracy levels and R square are not satisfactory proceed to below step.
# 
#        b) Run another model at State level. There are 52 states in USA.
# 
#        c) Keep below considerations while building a linear regression model:
# 
# Variables should have significant impact on predicting Monthly mortgage and owner costs
# 
# Utilize all predictor variable to start with initial hypothesis
# 
# R square of 60 percent and above should be achieved
# 
# Ensure Multi-collinearity does not exist in dependent variables
# 
# Test if predicted variable is normally distributed

# In[61]:


x_train=pd.read_csv("C:/Users/91987/OneDrive/Desktop/Varad DS Capstone Project/Project 1/train.csv")
x_test=pd.read_csv("C:/Users/91987/OneDrive/Desktop/Varad DS Capstone Project/Project 1/test.csv")


# In[62]:


x_train.drop(['BLOCKID','SUMLEVEL'],axis=1,inplace=True)


# In[63]:


x_train.dropna(axis=0,inplace=True)
x_train.head()


# In[64]:


x_train.drop_duplicates(inplace=True)
x_train.shape


# In[65]:


x_test.head()


# In[66]:


x_test.drop(['BLOCKID','SUMLEVEL'],axis=1,inplace=True)


# In[67]:


x_test.isna().sum()


# In[68]:


x_test.dropna(axis=0,inplace=True)


# In[69]:


x_test.drop_duplicates(inplace=True)
x_test.shape


# In[70]:


imp_feature = x_train.select_dtypes(exclude=('object','category'))
imp_feature.head()


# In[71]:


to_drop = ['UID','COUNTYID', 'STATEID', 'zip_code', 'area_code', 'lat', 'lng']
for col in imp_feature.columns:
    if col in to_drop:
        imp_feature.drop(col,axis=1,inplace=True)

imp_feature.head()


# In[72]:


x_train_features = imp_feature[['pop','rent_median','hi_median','family_median','hc_mean','second_mortgage','home_equity','debt','hs_degree','pct_own','married','separated','divorced']]


# In[73]:


x_train_features.shape


# In[74]:


y_train = imp_feature['hc_mortgage_mean']


# In[75]:


x_test_feature = x_test[['pop','rent_median','hi_median','family_median','hc_mean','second_mortgage','home_equity','debt','hs_degree','pct_own','married','separated','divorced']]


# In[76]:


from sklearn.linear_model import LinearRegression
le = LinearRegression()


# In[77]:


le.fit(x_train_features,y_train)


# In[78]:


y_pred = le.predict(x_test_feature)


# In[79]:


y_test = x_test['hc_mortgage_mean']


# In[80]:


from sklearn.metrics import r2_score,mean_squared_error


# In[81]:


r2_score(y_test,y_pred)


# In[82]:


np.sqrt(mean_squared_error(y_test,y_pred))


# In[83]:


# Visualization 21:
sns.distplot(y_pred)
plt.show()


# In[ ]:


## Result : Data looks predicted variables are not normally Distributed but right positive skewed

