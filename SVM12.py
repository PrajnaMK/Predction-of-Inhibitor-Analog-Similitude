# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 22:44:45 2019

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
from sklearn import preprocessing
from sklearn import utils
from sklearn.cluster import KMeans
import seaborn as sns


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
y=y.astype(float)
X=X.astype(float)

lab_enc1 = preprocessing.LabelEncoder()
training_scores_encoded2 = lab_enc1.fit_transform(y)
#print(training_scores_encoded2)
#print(utils.multiclass.type_of_target(y))
#print(utils.multiclass.type_of_target(y.astype('int')))
#print(utils.multiclass.type_of_target(training_scores_encoded2))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

#implementing the svm algorithm
scores=[]
from sklearn.svm import SVC


#clss = SVC(kernel='poly',C=1000.0, gamma=0.001, degree=6) # run this line to see the otuput for polynomial kernel
clss = SVC(kernel='linear',C=10000.0, gamma=0.0001, degree=9)
cv = KFold(n_splits=5, random_state=20, shuffle=False)
for train_index, test_index in cv.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], training_scores_encoded2[train_index], training_scores_encoded2[test_index]
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
   

    clss.fit(X_train, y_train)
    scores.append(clss.score(X_test, y_test))
    
#print(len(X_train))


cm=cross_val_predict(clss, X, training_scores_encoded2, cv=5)


plt.scatter(training_scores_encoded2,cm)
plt.xlabel('TARGET weights')
plt.ylabel('molecular weight')
plt.show()

accuracy = metrics.r2_score(training_scores_encoded2, cm)
print ("accuracy :", accuracy*100,"%")

from sklearn.metrics import confusion_matrix
confusion_matrix(training_scores_encoded2, cm).ravel()
