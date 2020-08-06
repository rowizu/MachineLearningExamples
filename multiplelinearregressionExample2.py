#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:14:03 2020

@author: ruwaidazuhairy
"""

"""
This is an example to predict a car price based on 
Output for a hosiery mill over a 4-year period, using Multiple Linear Regression Model, 
Random Forest Regressor, Decision Tree regressor, and K-nearest Neighbours
"""

"""
Dataset: 
Source: Shaliny Goyal, "Car Price Prediction", 
https://www.kaggle.com/goyalshalini93/car-data
Description: A Chinese automobile company Teclov_chinese aspires to enter 
the US market by setting up their manufacturing unit there and producing cars 
locally to give competition to their US and European counterparts. They have 
contracted an automobile consulting company to understand the factors on which 
the pricing of cars depends. Specifically, they want to understand the factors 
affecting the pricing of cars in the American market, since those may be very 
different from the Chinese market.

Variables/Columns:
  car_ID
symboling          
CarName          
fueltype         
aspiration     
doornumber       
carbody        
drivewheel       
enginelocation   
wheelbase      
carlength        
carwidth       
carheight   
curbweight        
enginetype        
cylindernumber  
enginesize        
fuelsystem      
boreratio       
 stroke          
compressionratio 
horsepower     
peakrpm        
citympg            
highwaympg   
price
"""

#Importing Libraries
import pandas as pd  
import matplotlib.pyplot as plt 
import seaborn as sns
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

#Importiung the dataset
df=pd.read_csv('CarPrice_Assignment.csv')
print(df.shape)
print(df.describe())
print(df.info())

#Checking column with missing data
print(df.columns[df.isnull().any()])

#Handeling Categorical Data 
dataset=df
dataset[['carCompany','carName']] = df.CarName.str.split(' ',1,expand=True)
dataset['carCompany']=dataset['carCompany'].str.replace('toyouta','toyota')
dataset['carCompany']=dataset['carCompany'].str.replace('vokswagen','vw')
dataset['carCompany']=dataset['carCompany'].str.replace('volkswagen','vw')
dataset['carCompany']=dataset['carCompany'].str.replace('maxda','mazda')
dataset['carCompany']=dataset['carCompany'].str.replace('porschce','porsche')
dataset['carCompany']=dataset['carCompany'].str.replace('porcshce','porsche')
dataset['carCompany']=dataset['carCompany'].str.replace('Nissan','nissan')
del dataset['carName']
del dataset['CarName']
dataset.info()
#get correlations of each features in dataset
corrmat = dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#Adjusting Dataset upon feature selection
dataset=dataset[['enginesize','horsepower','wheelbase','boreratio','fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','enginetype','cylindernumber','fuelsystem','carCompany','price']]
dataset=pd.get_dummies(dataset,columns=['fueltype','fuelsystem','enginelocation','drivewheel','aspiration','doornumber','carbody','enginetype','cylindernumber','carCompany'])
print(dataset.info())

 #drop columns
cols_to_drop=dataset.corr()[(dataset.corr()['price']<=0.5) & (dataset.corr()['price']>=-0.5)]
cols_to_drop=cols_to_drop.reset_index()['index']
cols_to_drop=list(cols_to_drop)
dataset.drop(cols_to_drop,axis=1,inplace=True)
print(dataset.info())

#get correlations of each features in dataset
corrmat = dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#drop columns
dataset.drop(['horsepower','wheelbase','boreratio'],axis=1,inplace=True)
print(dataset.info())


#get correlations of each features in dataset
corrmat = dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#drop columns
dataset.drop(['drivewheel_fwd'],axis=1,inplace=True)
print(dataset.info())

#get correlations of each features in dataset
corrmat = dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#shuffling rows
df1=dataset.sample(frac=1)

#Assigning variables
y=dataset.iloc[:,1].values
x_dataset=dataset
x_dataset.drop(['price'],axis=1,inplace=True)
x_dataset.info()
X=x_dataset.iloc[:,:].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# Fitting Multiple Linear Regression to the Training set
model_name= 'Multible Linear Regression'
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set result
y_pred= regressor.predict(X_test)

# Evaluating the model 
def evaluate(name,ytest,ypred):
    """
    This function is to evaluate models
    """
    r=r2_score(ytest, ypred)*100
    mse=mean_squared_error(ytest, ypred)
    mae=mean_absolute_error(ytest, ypred)
    rmse=sqrt(mean_absolute_error(ytest, ypred))
    print("The " + name + "\n R2_Score: %.2f%%"% r +" \n Mean Squared Error is "
          + str(mse) + " \n Mean Absolute Error is "+str(mae)+ 
          " \n and root mean squared error is "+str(rmse)+'\n')

evaluate(model_name, y_test, y_pred)


#Standardisation
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.transform(X_test)

#using Random Forest Regressor to predict the values
#Import RFR model
from sklearn.ensemble import RandomForestRegressor

# set seed to make results reproducible
RF_SEED = 30
# training data fit
regressor1 = RandomForestRegressor(n_estimators=1000, random_state=RF_SEED)
regressor1.fit(train_scaled, y_train)
 
y_pred2 = regressor1.predict(test_scaled)


# Evaluating the model 

evaluate('Random Forest Regressor', y_test, y_pred2)

#using Decision Tree Regressor to predict the values
from sklearn.tree import DecisionTreeRegressor
regressor4 = DecisionTreeRegressor()
regressor4.fit(train_scaled, y_train)

y_pred4=regressor4.predict(test_scaled)

# Evaluating the model 

evaluate('Decision Tree Regressor', y_test, y_pred4)

#Using K Nearest Neighbors
#applying the model
from sklearn.neighbors import KNeighborsRegressor
regressor3 = KNeighborsRegressor()
regressor3.fit(train_scaled, y_train)
y_pred3 = regressor3.predict(test_scaled)

#evaluating the model
evaluate('K Nearest Neighbors Model', y_test, y_pred3)
