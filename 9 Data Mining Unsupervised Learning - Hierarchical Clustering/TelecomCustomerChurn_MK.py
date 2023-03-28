import pandas as pd

import matplotlib.pylab as plt
from sklearn.cluster import KMeans

churn = pd.read_excel("C:/Users/madhu/OneDrive/Desktop/360digiTMG/Data Science/9 Data Mining Unsupervised Learning - Hierarchical Clustering/Hands-on Material/Telco_customer_churn.xlsx")
churn.describe()

churn1 = churn.drop(['Customer ID', 'Count', 'Quarter'], axis = 1)

churn_new = pd.get_dummies(churn1)

def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return(x)

churn_norm = norm_func(churn_new.iloc[:, 0:10])    

TWSS = []
k = list(range (2,9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(churn_norm)
    TWSS.append(kmeans.inertia_)

TWSS

plt.plot(k, TWSS, 'ro-'); plt.xlabel('No of Clusters'); plt.ylabel('total within SS')

C_model = KMeans(n_clusters = 3)
C_model.fit(churn_norm)

C_model.labels_

C_md = pd.Series(C_model.labels_)
churn1['clust'] = C_md 

churn1.head()

churn1 = churn1.iloc[:, [27,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]]

churn1.head()

churn1.iloc[:, [2,3,6,10,22,23,24,25,26,27]].groupby(churn1.clust).mean()
