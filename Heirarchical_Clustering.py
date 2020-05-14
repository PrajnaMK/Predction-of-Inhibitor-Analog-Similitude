# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('biodata.csv')
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
X=dataset.iloc[:,[0,1,2,4,3,5,6,7]].values
#y=dataset.iloc[:,0].values
y=y.astype(float)
X=X.astype(float)

# Using the dendrogram to find the optimal number of clusters

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import DivisiveClustering
hc = DivisiveCl(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.title('Clusters of Analogs)
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()