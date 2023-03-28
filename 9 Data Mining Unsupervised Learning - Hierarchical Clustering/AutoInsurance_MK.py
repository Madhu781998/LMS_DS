import pandas as pd
import matplotlib.pylab as plt

ins = pd.read_csv('C:/Users/madhu/OneDrive/Desktop/360digiTMG/Data Science/9 Data Mining Unsupervised Learning - Hierarchical Clustering/Hands-on Material/AutoInsurance.csv')

ins1 = ins.iloc[:, 2:]
ins1.columns

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
ins1['Coverage'] = lb.fit_transform(ins1['Coverage'])
ins1['Education'] = lb.fit_transform(ins1['Education'])
F_ins = pd.get_dummies(ins1)

F_ins.describe()

def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

ins_norm = norm_func(F_ins)

ins_norm.describe()

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(ins_norm, method = "complete", metric = "euclidean")

plt.figure(figsize=(80, 60)); plt.title('Hierarchical Clustering Dendrogram'); plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10,
    p = 5, truncate_mode = 'level'# font size for the x axis labels
)
plt.show()

from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(ins_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

ins1['Clust'] = cluster_labels  

ins = ins1.iloc[:, [22, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]
ins.head()

ins = ins1.iloc[:, [1, 3, 4, 8, 11, 12, 13, 14, 15, 20 ]].groupby(ins.Clust).mean()
