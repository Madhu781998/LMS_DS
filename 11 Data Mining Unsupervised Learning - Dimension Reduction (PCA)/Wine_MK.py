import pandas as pd    #### for data manupulation
import numpy as np     #### for numerical calculations
import matplotlib.pylab as plt    #### for data visualisation

wine = pd.read_csv('C:/Users/madhu/OneDrive/Desktop/360digiTMG/Data Science/11 Data Mining Unsupervised Learning - Dimension Reduction (PCA)/Hands-on Material/wine.csv')
wine.describe()    ### for describing the data set  
wine.info()        ### for getting all information of the data 

from sklearn.decomposition import PCA     ##### for PCA caculations
from sklearn.preprocessing import scale   ##### For normalising the data

wine_norm = scale(wine)    ###### scaling or nomalising 
wine_norm

pca = PCA(n_components = 14)    ###### PCA calculations as per the columns in the given data we consider n components 
pca_values = pca.fit_transform(wine_norm)

var = pca.explained_variance_ratio_    #### for calculations of the variance in the PCA 
var

pca.components_                  ##### for calculating the weight of the PCA
pca.components_[0]

var1 = np.cumsum(np.round(var, decimals = 4)* 100)    ##### for calculations of the cumulative variance
var1

plt.plot(var1, color = "red")

pca_values

pca_data = pd.DataFrame(pca_values)   ### for converting array into data frame

pca_data.columns = "comp0", "comp1", "comp2", "comp3", "comp4", "comp5", "comp6", "comp7", "comp8", "comp9", "comp10", "comp11", "comp12", "comp13"
 
final = pca_data.iloc[:, 0:3]    #### considering only first three columns

wx = final.plot(x="comp0", y="comp1", kind="scatter", figsize=(15,10))     #### for plotting scatter plot between comp0 and comp1
