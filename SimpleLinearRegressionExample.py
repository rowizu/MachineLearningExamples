#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:21:46 2020
Simple Linear Regression
@author: ruwaidazuhairy

This is an example to predict the student grades based on 
 studying hours.

"""

"""Dataset: Dataset:  millcost.dat

Source: J. Dean (1941), "Statistical Cost Functions of a Hosiery Mill", 
Studies in Business Administration, Vol. 14, #3.

Description: Monthly Production Costs and Output for a hosiery
mill over a 4-year period (Data approximated from graph).

Variables/Columns
Month   1-2
Production   12-16   (Thousands of dozens of pairs)
Cost    20-24 ($1000s)"""

# Data Preprocessing

# Importing the libraries

import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('millcost_data.csv')

X = dataset.iloc[:,:1].values
y = dataset.iloc[:,-1].values


plt.scatter(X,y)
plt.show()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the Linear regresiion to the training set
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, y_train)

# Predecting the results 
y_pred=regressor.predict(X_test)

# Visualizaing the results 
plt.scatter(X_train, y_train, color='purple')
plt.plot(X_train, regressor.predict(X_train),color='pink')
plt.title('Monthly Production Costs vs Output for a hosiery mill over a 4year period (training set)')
plt.xlabel('Output for a hosiery mill')
plt.ylabel('Monthly Production Costs')
plt.show()

plt.scatter(X_test, y_test, color='purple')
plt.plot(X_test, y_pred ,color='pink')
plt.title('Monthly Production Costs vs Output for a hosiery mill over a 4year period (testing set)')
plt.xlabel('Output for a hosiery mill')
plt.ylabel('Monthly Production Costs')
plt.show()

# Evaluating the model 
from sklearn.metrics import r2_score
print("R2_Score: %.2f%%" % (r2_score(y_test, y_pred)*100))

from sklearn.metrics import mean_squared_error
print("R2_Score: %.2f%%" % mean_squared_error(y_test, y_pred))