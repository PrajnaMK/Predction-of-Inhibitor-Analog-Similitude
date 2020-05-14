# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 18:34:59 2019

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

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, training_scores_encoded2, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


plt.scatter(y_test,y_pred)
plt.xlabel('TARGET weights')
plt.ylabel('molecular weight')
plt.show()

accuracy = metrics.r2_score(y_test, y_pred)
print ("accuracy :", accuracy*100,"%")

#print(cm)