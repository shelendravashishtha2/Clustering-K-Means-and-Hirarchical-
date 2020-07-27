import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data = pd.read_csv('Mall_Customers.csv')
data.head()
"""Label Encoding Genre Column"""
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
data[['Genre']]  = lb.fit_transform(data[['Genre']])
X = data.iloc[:,1:]
wcss = []
test = pd.DataFrame(X)
test.hist()
from sklearn.cluster import KMeans
/*plotting Within Cluster Sum of Squares to find optimal number of clusters in K Means*/
for i in range(1,11):
    km = KMeans(n_clusters = i,init = 'k-means++',random_state = 42)
    km.fit(X)
    wcss.append(km.inertia_)
plt.plot(wcss,marker = '*')
plt.xlabel('Number Of Clusters')
plt.ylabel('WCSS value per cluster')
plt.show()
/*found 4 or 5 clusters to be optimal ... Finding predicted Y based on 4 clusters */
km = KMeans(n_clusters = 4,init ='k-means++',random_state = 42)
y_km = km.fit_predict(X)
/*Performing Hierarchical Clustering*/
/*Plotting Dendrogram to find optimal numbers of clusters */
import scipy.cluster.hierarchy as dg
dgr = dg.dendrogram(dg.linkage(X,method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customer')
plt.ylabel('Euclidean Distance')
plt.show()
from sklearn.cluster import AgglomerativeClustering 
amc = AgglomerativeClustering(n_clusters = 3,affinity = 'euclidean',linkage = 'ward')
y_hc = amc.fit_predict(X)