import pandas as pd
import matplotlib.pylab as plt

crime_data = pd.read_csv("C:/Users/madhu/OneDrive/Desktop/360digiTMG/Data Science/9 Data Mining Unsupervised Learning - Hierarchical Clustering/Hands-on Material/crime_data.csv")
crime_data.describe()
crime_data.info()

# Normalization function
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)


# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(crime_data.iloc[:, 1:])
df_norm.describe()


from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch # for creating dendrogram 

linkage1 = linkage(df_norm, method = "complete", metric = "euclidean")

plt.figure(figsize=(15, 8)); plt.title('Hierarchical Clustering Dendrogram'); plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(linkage1, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 8 # font size for the x axis labels
)
plt.show()

#Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)
crime_data['clust'] = cluster_labels

crime_data = crime_data.iloc[:, [5,0,1,2,3,4]]
crime_data.head()

crime_data.iloc[:, 2:].groupby(crime_data.clust).mean()

crime_data.to_csv("Crime_data.csv", encoding = "utf-8")

import os
os.getcwd()
