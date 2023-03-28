import pandas as pd
import matplotlib.pylab as plt

air = pd.read_excel('C:/Users/madhu/OneDrive/Desktop/360digiTMG/Data Science/9 Data Mining Unsupervised Learning - Hierarchical Clustering/Hands-on Material/EastWestAirlines.xlsx', sheet_name="data")

air = air.drop(['ID#'], axis = 1)
air.describe()
air.info()

def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return (x)

air_norm = norm_func(air)
air_norm.describe()

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(air_norm, method = 'complete', metric = 'euclidean')

plt.figure(figsize=(30, 20)); plt.title('Hierarchical Clustering Dendrogram'); plt.xlabel('Index'); plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 90,  # rotates the x axis labels
    leaf_font_size = 10,   # font size for the x axis labels
    p = 5, truncate_mode = 'level'
)
plt.show()

from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters=3, linkage = 'complete', affinity = 'euclidean').fit(air_norm)
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)
air['clust'] = cluster_labels

air = air.iloc[:, [11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
air.head()

air1 = air.iloc[:, 1:].groupby(air.clust).mean()
