#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from math import sin, cos, sqrt, atan2, radians


# In[2]:


df = pd.read_csv(r'C:\Users\Amshu\Downloads\Folder\Practicum\Data_final.csv',index_col=0)


# Delimiting columns

# In[3]:


new1 = df['Category'].str.split(',',expand=True)

new1.rename(columns={0: 'category0', 1: 'category1', 2 : 'category2'},inplace=True)

new2 = df['Cuisine'].str.split(',',expand=True)

new2.rename(columns={0: 'cuisine0', 1: 'cuisine1', 2 : 'cuisine2'},inplace=True)


# Concatenating processed columns

# In[4]:


df['distance'] = 0

df = pd.concat([df,new1,new2],axis=1)

df.drop(columns=['Category','Cuisine'],inplace=True)


# In[5]:


df.head()


# In[32]:


t1 = df.groupby([df.index, 'category0'])['title'].first().unstack()
t2 = df.groupby([df.index, 'category1'])['title'].first().unstack()
t3 = df.groupby([df.index, 'category2'])['title'].first().unstack()


# In[33]:


t1 = t1.replace(np.nan,0)
t2 = t2.replace(np.nan,0)
t3 = t3.replace(np.nan,0)


# In[34]:


t1[t1 != 0] = 1
t2[t2 != 0] = 1
t3[t3 != 0] = 1


# In[35]:


col_list = t1.columns.to_list()+t2.columns.to_list()+t3.columns.to_list()


# In[36]:


np.unique(col_list)

set(t2.columns.to_list()).difference(t1.columns.to_list())

set(t1.columns.to_list()).difference(t2.columns.to_list())

set(t1.columns.to_list()).difference(t3.columns.to_list())


# In[37]:


t1['08M']=0
t2['0UW']=0
t2['1PF']=0
t2['G2D']=0
t2['MDL']=0
t2['YUW']=0
t3['0UW']=0
t3['1PF']=0
t3['G2D']=0
t3['MDL']=0
t3['YUW']=0
t3['M6X']=0
t3['X48']=0
t3['ZCO']=0


# In[38]:


t2.columns

t3.columns

t1.columns


# In[39]:


t2 = t2[t1.columns]
t3 = t3[t1.columns]


# In[40]:


def fill_missing(x):
    """x is a group after group by `number`"""
    return x.reindex(
               list((x.name, v) for v in range(x.index[0][1], x.index[-1][1]+1))
           ).fillna(0)


# In[41]:


t2=t2.reindex(list(range(t1.index.min(),143)),fill_value=0)

t3=t3.reindex(list(range(t1.index.min(),143)),fill_value=0)


# In[42]:


t=t1+t2+t3
print(t)


# In[43]:


u1 = df.groupby([df.index, 'cuisine0'])['title'].first().unstack()
u2 = df.groupby([df.index, 'cuisine1'])['title'].first().unstack()
u3 = df.groupby([df.index, 'cuisine2'])['title'].first().unstack()


# In[44]:


u1 = u1.replace(np.nan,0)
u2 = u2.replace(np.nan,0)
u3 = u3.replace(np.nan,0)


# In[45]:


u1[u1 != 0] = 1
u2[u2 != 0] = 1
u3[u3 != 0] = 1


# In[46]:


print(u1)


# In[47]:


col_list1 = u1.columns.to_list()+u2.columns.to_list()+u3.columns.to_list()
np.unique(col_list1)

set(u2.columns.to_list()).difference(u1.columns.to_list())
set(u3.columns.to_list()).difference(u1.columns.to_list())
set(u1.columns.to_list()).difference(u2.columns.to_list())
set(u2.columns.to_list()).difference(u3.columns.to_list())


# In[48]:


u1['KKK']=0
u1['MQN']=0
u2['28E']=0
u2['4CF']=0
u2['4EE']=0
u2['7C5']=0
u2['IMW']=0
u2['JL9']=0
u2['LBP']=0
u2['S1H']=0
u2['VCE']=0
u3['28E']=0
u3['2Q6']=0
u3['4CF']=0
u3['4EE']=0
u3['7C5']=0
u3['8L3']=0
u3['IMW']=0
u3['JL9']=0
u3['LBP']=0
u3['MV1']=0
u3['O2F']=0
u3['QIX']=0
u3['S1H']=0
u3['VCE']=0
u3['ZB5']=0
u3['KKK']=0
u3['MQN']=0


# In[49]:


u2 = u2[u1.columns]
u3 = u3[u1.columns]


# In[50]:


print(u2)


# In[51]:


u2=u2.reindex(list(range(u1.index.min(),143)),fill_value=0)

u3=u3.reindex(list(range(u1.index.min(),143)),fill_value=0)


# In[52]:


u=u1+u2+u3
print(u)


# In[53]:


comb = pd.concat([t,u], axis=1)
print(comb)


# In[54]:


dic = {'YUW':'Study','1PF':'Religion','OU2':'History','M6X':'Wildlife','X48':'Nature','9CG':'Culture','ZCO':'Alcohol','0UW':'Sport','Y0B':'Art','G2D':'Shopping','MDL':'Eating','08M':'Architecture','KO4':'','7C5':'None','MV1':'Irish','IMW':'American','S1H':'Chocolate','QIX':'Buffet','4EE':'Indian','2Q6':'CafÃ©','JL9':'Mexican','ZB5':'FastFood','O2F':'Bar','28E':'Italian','A09':'European','8L3':'Asian','VCE':'Mediterrainean','4CF':'French','LBP':'Chinese','5N6':'','KKK':'MiddleEastern','MQN':'Thai'}


# In[56]:


comb = comb.rename(columns=dic)


# In[57]:


dfa = pd.read_csv(r'C:\Users\Amshu\Downloads\Folder\Practicum\Data_location.csv',index_col=0)


# Add first Category:

# In[58]:


q0= input()

query0 = str(q0) + ' == 1'


# Add Second Category

# In[59]:


q1= input()

query1 = str(q1) + ' == 1'


# Add First Cuisine

# In[60]:


q2= input()

query2 = str(q2) + ' == 1'


# Add Second Cuisine

# In[61]:


q3= input()

query3 = str(q3) + ' == 1'


# In[62]:


query= query0 + ' or ' + query1 + ' or ' + query2 + ' or ' + query3


# In[63]:


combo=comb.query(query)


# In[65]:


combo['title']=dfa['title']
combo['latitude']=df['latitude']
combo['longitude']=df['longitude']
combo['distance']=0
print(combo)


# User input fn

# In[66]:


def reqts(user_lat,user_lon):
    return user_lat, user_lon 


# In[67]:


user_lat, user_lon = reqts(53.3516,-6.2639)


# Dist fn

# In[68]:


# approximate radius of earth in km
R = 6373.0

def dist(lat1,lon1,lat2,lon2):
    x1 = radians(lat1)
    y1 = radians(lon1)
    x2 = radians(lat2)
    y2 = radians(lon2)
    
    dlon = y2 - y1
    dlat = x2 - x1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return(distance)


# In[71]:


test1=[]

for index, row in combo.iterrows():
    a=dist(user_lat,user_lon,row[-3],row[-2])
    test1.append(a)

combo['distance']=test1

combo.fillna(value=0, inplace=True)


# In[72]:


combo


# ML Optimizing

# In[73]:


test=combo


# In[74]:


test.drop(columns=['latitude','longitude'],inplace=True)


# In[75]:


test


# In[89]:


#test['title'] = df['title']

test['score']=0

test1 = [] 

#Get Weights of Coeffs
for i,row in test.iterrows():
    score= (0.5/row['distance']) + 1.2*row[q0] + 1.4*row[q1] + 0.8*row[q2] + 0.6*row[q3]
    test1.append(score)
    
test['score']=test1


# In[90]:


test


# In[91]:


score_cls = []

for i, row in test.iterrows():
    if row.score < 1:
        a = 0
    else:
        a=1
    score_cls.append(a)

test['target'] = score_cls


# In[92]:


test


# In[93]:


x1=test.nlargest(3, 'score')
print('The 3 recommended locations are:')
x1['title']


# In[94]:


print('The 3 scores for the 3 locations are:')
x1['score']


# ***

# Application of ML

# In[95]:


import statsmodels.formula.api as smf
from sklearn.linear_model import  LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


# In[96]:


x = test[['distance', q0, q1, q2, q3]].values

y = test['score'].values


# In[97]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[98]:


columns1=['Distance',q0,q1,q2,q3]


# ***

# Linear Regn

# In[99]:


Y = test['score']

X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.33, random_state=42)

model_r = LinearRegression()

model_r.fit(X_train,y_train)

y_pred_r = model_r.predict(X_test)

print(r2_score(y_test,y_pred_r),mean_squared_error(y_test,y_pred_r))

print(model_r.coef_)


# In[100]:


list(zip(columns1, model_r.coef_))


# Stats model

# In[101]:


form='score ~ distance ' + ' +' + q0 +' +' + q1 + '+'+ q2 +' +' + q3


# In[102]:


lm1 = smf.ols(formula=form, data=test).fit()


# In[103]:


lm1.params


# In[104]:


lm1.summary()


# In[105]:


lm1.rsquared


# In[106]:


lm1.pvalues


# In[107]:


regr = RandomForestRegressor(max_depth=2, random_state=42,n_estimators=100)
regr.fit(X_train,y_train)
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
           oob_score=False, random_state=0, verbose=0, warm_start=False)
print(regr.feature_importances_)
y_pred_rf = regr.predict(X_test)


# In[108]:


print(r2_score(y_test,y_pred_rf),mean_squared_error(y_test,y_pred_rf))


# In[114]:


bag=BaggingRegressor(base_estimator=None, n_estimators=100, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=42, verbose=0)
bag.fit(X_train,y_train)
#print(bag.estimators_features_)
y_pred_bag = bag.predict(X_test)
print(r2_score(y_test,y_pred_bag),mean_squared_error(y_test,y_pred_bag))
print(bag.estimators_features_)


# In[110]:


grad= GradientBoostingRegressor(random_state=42)
grad.fit(X_train,y_train)
print(grad.feature_importances_)
y_pred_grad = grad.predict(X_test)
print(r2_score(y_test,y_pred_grad),mean_squared_error(y_test,y_pred_grad))


# In[111]:


fig, ax = plt.subplots()
ax.scatter(y_test,y_pred_r)
#ax.scatter(y_test,y_pred_rf)
#ax.scatter(y_test,y_pred_bag)
#ax.scatter(y_test,y_pred_grad)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


# In[ ]:




