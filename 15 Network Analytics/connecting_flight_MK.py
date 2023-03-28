import pandas as pd

import networkx as nx 

connecting_route = pd.read_csv("C:/Users/madhu/OneDrive/Desktop/360digiTMG/Data Science/15 Network Analytics/Hands-on Material/connecting_routes.csv",header=None)

#3.	Data Cleaning
#droping empty column
connecting_route.drop([6],axis=1,inplace=True)

# changing index cols with name() 
connecting_route.columns = ["flights", " ID", "main Airport" , "main Airport ID", "Destination","Destination  ID","haults","machinary"]


#3.	Data Cleaning

connecting_route.nunique()

connecting_route.isnull().sum()

#Model Building
g_data = nx.Graph()

g_data = nx.from_pandas_edgelist(connecting_route, source = 'main Airport', target = 'Destination')

print(nx.info(g_data))

d_c = nx.degree_centrality(g_data)  # Degree Centrality
print(d_c) 
#top 10  Degree Centrality
top_10_dc = sorted(d_c, key = d_c.get, reverse=True)[:10]
top_10_dc

#for graph 
pos = nx.spring_layout(g_data)
nx.draw_networkx(g_data, pos, node_size = 20, node_color = 'red')


# closeness centrality
closeness = nx.closeness_centrality(g_data)
print(closeness)
#top 10 closeness centrality
top_10_cc= sorted(closeness, key = closeness.get, reverse = True)[:10]
top_10_cc

## Betweeness Centrality 
b_c = nx.betweenness_centrality(g_data) # Betweeness_Centrality
print(b_c)
#top 10 Betweeness Centrality
top_10_bc= sorted(b_c, key=b_c.get, reverse=True)[:10]
top_10_bc

## Eigen-Vector Centrality
ev = nx.eigenvector_centrality(g_data) # Eigen vector centrality
print(ev)
#top 10 Eigen vector centrality
top_10_ev = sorted(ev, key = ev.get, reverse = True)[:10]
top_10_ev

# cluster coefficient
cluster_coeff = nx.clustering(g_data)
print(cluster_coeff)

# Average clustering
a_c = nx.average_clustering(g_data) 
a_c

print(a_c)