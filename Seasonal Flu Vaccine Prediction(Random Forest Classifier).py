#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import io
get_ipython().run_line_magic('cd', '"C:\\Users\\tsgee\\OneDrive\\Desktop\\New folder\\sesonal flu vaccine"')


# In[3]:


flutrain=pd.read_csv("training_set_features.csv")


# In[4]:


flutest=pd.read_csv("test_set_features.csv")


# In[5]:


flulabels=pd.read_csv("training_set_labels.csv")


# In[6]:


print(flutrain.shape)
print(flutest.shape)
print(flulabels.shape)


# In[7]:


# Algorithm based Missing Value Imputation - Considers the impact of
# other variables on the missing value and impute accordingly.
# MICE - Multivariate Imputation using Chained Equations is the most
# popular algorithm
# Imputes both numeric and non numeric object/categrical missing values.
# Intitially imputed with mean, median or mode and chained equations are
# built
# Chained Equations wherein the missing value column/variable is treated
# as dependent variable and relevant classification/regression model is
# built and prediction done.
# Missing value will be replaced with predicted value.


# In[8]:


flutrain.info()


# In[9]:


flutest.info()


# In[10]:


# Combine both dataframes for preprocessing
combinedf=pd.concat([flutrain,flutest],axis=0)


# In[11]:


combinedf.info()


# In[12]:


combinedf=combinedf.drop('respondent_id',axis=1)


# In[13]:


# For using Iterative Imputer in sklearn which is experimental as of now
# 1) remove variables or columns not needed
# 2) Labelencode all object and categorical data but retain the Missing
# value as it is.


# In[14]:


combinedf.head()


# In[15]:


from sklearn.preprocessing import LabelEncoder


# In[16]:


original=combinedf


# In[17]:


mask=combinedf.isnull()


# In[18]:


combinedf=combinedf.astype(str).apply(LabelEncoder().fit_transform)


# In[19]:


combinedf=combinedf.where(~mask,original)


# In[20]:


combinedf.head()


# In[21]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeClassifier


# In[22]:


imputer=IterativeImputer(estimator=DecisionTreeClassifier(),
                        initial_strategy="most_frequent",
                        max_iter=20)


# In[23]:


combinedf_impute=imputer.fit_transform(combinedf)


# In[24]:


combinedf_impute=pd.DataFrame(combinedf_impute,columns=combinedf.columns)


# In[25]:


combinedf.employment_industry.value_counts(dropna=False).plot(kind='bar')


# In[26]:


combinedf_impute.employment_occupation.value_counts(
    dropna=False).plot(kind="bar")


# In[27]:


combinedf_impute.info()


# In[28]:


# Split Data back to train & test
flu_train=combinedf_impute.loc[0:26706]


# In[29]:


flu_test=combinedf_impute.loc[26707:53414]


# In[30]:


print(flu_train.shape)
print(flutrain.shape)
print(flu_test.shape)
print(flutest.shape)


# In[31]:


y=flulabels.seasonal_vaccine
X=flu_train


# In[32]:


y.value_counts().plot(kind="bar")


# In[33]:


y.shape


# In[34]:


# Build the following - Score, predict, classfication Report, ROC Curve
# binary Logistic Regression
# Decision Tree
# Random Forest
# Gradient boosting
# Naive Bayes
# Support Vector Machine


# In[35]:


from sklearn.metrics import classification_report,RocCurveDisplay
from sklearn.linear_model import LogisticRegression


# In[36]:


logit=LogisticRegression(max_iter=2000)


# In[37]:


logitmodel=logit.fit(X,y)


# In[38]:


logitmodel.score(X,y)


# In[39]:


logitpredict=logitmodel.predict(X)


# In[40]:


pd.crosstab(y,logitpredict)


# In[41]:


print(classification_report(y,logitpredict))


# In[42]:


RocCurveDisplay.from_predictions(y,logitpredict)


# In[43]:


from sklearn.tree import DecisionTreeClassifier


# In[44]:


tree=DecisionTreeClassifier(max_depth=10)


# In[45]:


treemodel=tree.fit(X,y)


# In[46]:


treemodel.score(X,y)


# In[47]:


from sklearn.model_selection import cross_val_score


# In[48]:


cross_val_score(tree,X,y)


# In[49]:


np.mean([0.73886185, 0.74503931, 0.74255757, 0.74705111, 0.74255757])


# In[50]:


treepredict=treemodel.predict(X)


# In[51]:


pd.crosstab(y,treepredict)


# In[52]:


print(classification_report(y,treepredict))


# In[53]:


RocCurveDisplay.from_predictions(y,treepredict)


# In[54]:


from sklearn.ensemble import RandomForestClassifier


# In[55]:


RF=RandomForestClassifier(n_estimators=1000,max_depth=12)


# In[56]:


RFmodel=RF.fit(X,y)


# In[57]:


RFmodel.score(X,y)


# In[58]:


cross_val_score(RF,X,y)


# In[59]:


RFpredict=RFmodel.predict(X)


# In[60]:


pd.crosstab(y,RFpredict)


# In[61]:


print(classification_report(y,RFpredict))


# In[62]:


RocCurveDisplay.from_predictions(y,RFpredict)


# In[63]:


from sklearn.ensemble import GradientBoostingClassifier


# In[64]:


gbm=GradientBoostingClassifier(n_estimators=3000)


# In[65]:


gbmmodel=gbm.fit(X,y)


# In[66]:


gbmmodel.score(X,y)


# In[67]:


gbmpredict=gbmmodel.predict(X)


# In[68]:


pd.crosstab(y,gbmpredict)


# In[69]:


print(classification_report(y,gbmpredict))


# In[70]:


RocCurveDisplay.from_predictions(y,gbmpredict)


# In[71]:


from sklearn.naive_bayes import CategoricalNB


# In[72]:


nb=CategoricalNB()


# In[73]:


nbmodel=nb.fit(X,y)


# In[74]:


nbmodel.score(X,y)


# In[75]:


nbpredict=nbmodel.predict(X)


# In[76]:


pd.crosstab(y,nbpredict)


# In[77]:


print(classification_report(y,nbpredict))


# In[78]:


RocCurveDisplay.from_predictions(y,nbpredict)


# In[79]:


from sklearn.svm import SVC


# In[80]:


svm=SVC()


# In[81]:


svmmodel=svm.fit(X,y)


# In[82]:


svmmodel.score(X,y)


# In[ ]:




