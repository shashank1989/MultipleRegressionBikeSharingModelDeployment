#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 15:31:11 2021

@author: shashankbhatnagar
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")
import pickle


data = pd.read_csv("day.csv")

#Data Cleaning - OutlierAnalysis

#Before Outlieranalysis

outliers = []
out_summary = []
out_cols = ['temp','atemp','hum','windspeed']

#print("Before outliers treatment\n\n",data[out_cols].describe())

#Outliers columns identification 

for i in out_cols:
    Q3,Q1 = np.percentile(data[i],[75,25])
    IQR = Q3-Q1
    lower_bound = Q1-1.5*IQR
    upper_bound = Q3+1.5*IQR

    if ((data[i].min() < lower_bound) or (data[i].max() > upper_bound)):
        out_summary.append("attribute \"{}\" with min value : {} -> max value : {} -> IQR {} -> lower_bound : {} match is {} -> upper_bound : {} match is {}".format(i,data[i].min(),data[i].max(),IQR,Q1-1.5*IQR,(data[i].min() < lower_bound),Q3+1.5*IQR,data[i].max() > upper_bound))
        outliers.append(i)

# List of outliers satisfying lower or upper bound        
for i in range(0,len(out_summary)):
    #print("\nOutlier column with stats : \n\n{}\n".format(out_summary[i]))
    pass
    
#Outliers Treatment

for i in outliers:
    Q3,Q1 = np.percentile(data[i],[75,25])
    IQR = Q3-Q1
    lower_bound = Q1-1.5*IQR
    upper_bound = Q3+1.5*IQR
    data[i][data[i]<=lower_bound] = lower_bound
    data[i][data[i]>=upper_bound] = upper_bound


#print("After outliers treatment\n\n",data[out_cols].describe())


#Converting weathersit and season columns to labelled column for interpretation
# season (1:spring, 2:summer, 3:fall, 4:winter)
# weathersit (1 : cloudy 2 :Mist 3 :LightRain 4:HeavyRain )
data['season'] = data['season'].apply(lambda x :'spring' if x == 1 else ('summer' if x == 2 else ('fall' if x==3 else 'winter')))
data['weathersit'] = data['weathersit'].apply(lambda x :'cloudy' if x == 1 else ('Mist' if x == 2 else ('LightRain' if x==3 else 'HeavyRain')))
data['mnth'] = data['mnth'].replace({1: 'jan',2: 'feb',3: 'mar',4: 'apr',5: 'may',6: 'jun',7: 'jul',8: 'aug',9: 'sept',10: 'oct',11: 'nov',12: 'dec'})
data['weekday'] = data['weekday'].replace({0: 'sun',1: 'mon',2: 'tue',3: 'wed',4: 'thu',5: 'fri',6: 'sat'})

data.head()

## 3. Categorical Encoding

#mnth

data = pd.get_dummies(data=data,columns=['mnth'],drop_first=True)
data = pd.get_dummies(data=data,columns=['weekday'],drop_first=True)
data = pd.get_dummies(data=data,columns=['season'],drop_first=True)
data = pd.get_dummies(data=data,columns=['weathersit'],drop_first=True)

#data = data.drop(columns=['mnth','weekday','season','weathersit'],axis=1)

#Spliting testing and training data 

df_train,df_test = train_test_split(data,test_size=.3,random_state = 42)

#print(df_train.shape)
#print(df_test.shape)

#Scaling the values

scaled_list = ['temp','atemp','hum','windspeed']

#Using StandardScaler 
scaler = StandardScaler()

# Fit & Transform Training set
df_train[scaled_list] = scaler.fit_transform(df_train[scaled_list])
df_test[scaled_list] = scaler.transform(df_test[scaled_list])

df_train.head()

xtrain = df_train.drop(columns=['cnt'],axis=1)
xtest  = df_test.drop(columns=['cnt'],axis=1)
ytrain = df_train[['cnt']]
ytest  = df_test[['cnt']]


# Function to return the OLS model input parameter will be the DataFrame

def ols_model(df_ytrain,df_xtrain_rfe):
    """
    # Input parameter wll be the training dataset and output will be the ols model
    # we need to add constant to the xtrain and before fitting to OLS
    """
    df_xtrain_sm = stats.add_constant(df_xtrain_rfe)
    lm = stats.OLS(df_ytrain,df_xtrain_sm)
    model = lm.fit()
    return model,df_xtrain_sm

# Function to return the Varince Inflation Factor 

def variance_inflation(df_xtrain):
    """
    # Input parameter will be the training dataset and the output will be the vif data frame
    """
    vif = pd.DataFrame()
    vif['features'] = df_xtrain.columns
    vif['VIF'] = [variance_inflation_factor(df_xtrain.values,i) for i in range(df_xtrain.shape[1])]
    vif['VIF'] = round(vif['VIF'],2)
    vif = vif.sort_values(by="VIF",ascending=False)
    return vif

## Manual Elimination order

"""
High pvalue , high VIF - remove first
High pvalue , low VIF - remove second
Low  pvalue , high VIF - removed third
"""


rfe_support_and_eda_analysis_list = ['yr', 'workingday', 'temp', 'mnth_mar', 'mnth_may', 'mnth_sept',
       'season_spring', 'season_winter', 'weathersit_Mist', 'windspeed',
       'hum']

xtrain_rfe = xtrain[rfe_support_and_eda_analysis_list]

xtrain_rfe = xtrain[rfe_support_and_eda_analysis_list]
model8,df_xtrain_sm8 = ols_model(ytrain,xtrain_rfe)
vif = variance_inflation(xtrain_rfe)
#print(model8.summary())



pickle.dump(model8,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

pickle.dump(scaler,open('scaler.pkl','wb'))

scaler_model = pickle.load(open('scaler.pkl','rb'))




#Predicting Model 


sql_data = pd.DataFrame()

sql_data['yr'] = [1]
sql_data['workingday'] = [1]
sql_data['temp'] = [14.11]
sql_data['mnth_mar'] = [0]
sql_data['mnth_may'] = [0]
sql_data['mnth_sept'] = [0]
sql_data['season_spring'] = [1]
sql_data['season_winter'] = [0]
sql_data['weathersit_Mist'] = [1]
sql_data['windspeed'] = [10]
sql_data['hum'] = [80.5]
sql_data['atemp'] = [13.12]

sql_data[scaled_list] = scaler_model.transform(sql_data[scaled_list])

#print(sql_data)


sql_data = sql_data[['yr', 'workingday', 'temp', 'mnth_mar', 'mnth_may', 'mnth_sept',
       'season_spring', 'season_winter', 'weathersit_Mist', 'windspeed',
       'hum']]


sql_data_sm = stats.add_constant(sql_data, has_constant='add')

sql_data_pred = model.predict(sql_data_sm)

print(sql_data_pred[0])




