# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 00:43:32 2019

@author: prajn
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
import timeit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn import utils
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()
import pandas as pd


dataset = pd.read_csv("biodata.csv")



to_drop=['cmpdsynonym','inchikey','iupacname','meshheadings','aids','cidcdate','dois']
dataset.drop(to_drop, inplace=True, axis=1)

del dataset['cmpdname']
del dataset['mf']
del dataset['cid']
del dataset['xlogp']
del dataset['polararea']

X=dataset.iloc[:,[0,1,2,4,3,5,6,7]].values
#y=dataset.iloc[:,0].values
#y=y.astype(float)
X=X.astype(float)


lab_enc1 = preprocessing.LabelEncoder()
training_scores_encoded2 = lab_enc1.fit_transform(y)
#print(training_scores_encoded2)
#print(utils.multiclass.type_of_target(y))
#print(utils.multiclass.type_of_target(y.astype('int')))
#print(utils.multiclass.type_of_target(training_scores_encoded2))

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)

#print(pca.components_)

#print(pca.explained_variance_)

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',linewidth=2,shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)



pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)



X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal');



