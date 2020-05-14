# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 14:04:13 2019

@author: prajn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import metrics
import timeit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv("biodata.csv")

to_drop=['cmpdsynonym','inchikey','iupacname','meshheadings','aids','cidcdate','dois']
dataset.drop(to_drop, inplace=True, axis=1)

del dataset['cmpdname']
del dataset['mf']
del dataset['cid']
del dataset['xlogp']
del dataset['polararea']
'''dataset['xlogp'] = pd.to_numeric(dataset['xlogp'], errors='coerce')
dataset['xlogp'] = dataset['xlogp'].fillna(0)
dataset['xlogp']=dataset['xlogp'].astype(float)'''
X=dataset.iloc[:,[1,2,4,3,5,6,7]].values
y=dataset.iloc[:,0].values



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


plt.scatter(y_test, y_pred)
plt.xlabel('TARGET WEIGHTS')
plt.ylabel('MOLECULAR WEIGHTS')
plt.show()

accuracy = metrics.r2_score(y_test, y_pred)
print ("accuracy :", accuracy*100,"%")