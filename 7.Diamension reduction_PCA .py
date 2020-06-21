### DIAMESNSION REDUCTION### OR ### PRINCIPLE COMPONENT ANALYSIS ###

# Business problem:Perform Principal component analysis and perform clustering using first 3 principal component scores (both heirarchial and k mean clustering)

#importing libraries
import numpy as np
import pandas as pd 

# Importing the dataset
data = pd.read_csv("F:/alldatasets/wine.csv")

data_new = data[:]
# deleting the type from the data_new
del data_new['Type']

## P C A ##
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
# Normalizing the numerical data 
data_normal = scale(data_new)
pca = PCA(n_components = 13)
pca_values = pca.fit_transform(data_normal)
pca_values
# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var
pca.components_[0]
# Cumulative variance 
var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1
# Variance plot for PCA components obtained 
plt.plot(var1,color="red")
# plot between PCA1 and PCA2 
x = pca_values[:,0]
y = pca_values[:,1]
z = pca_values[:2:13]
plt.scatter(x,y,color=["red","blue"])

from mpl_toolkits.mplot3d import Axes3D
Axes3D.scatter(np.array(x),np.array(y),np.array(z),c=["green","blue","red"])

################### Clustering  with using FIRST 3 PCA SCORES ##########################
pca_values= pd.DataFrame(pca_values)
new_df = pca_values.iloc[:,0:3] # Selecting first 3 PCA scored columns that contains 63.53 cumulative variance

# K- Value = Sqrt[n/2] ## OR ##
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist
k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(new_df)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(new_df.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,new_df.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=3) 
model.fit(new_df)

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
data['clust_pca']=md # creating a  new column and assigning it to new column 
data['clust_pca']
data.head()

data = data.iloc[:,[14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]

#####CLUSTERING WITH USING ORIGINAL DATA ######
##### HIERARCHICAL CLUSTERING #######
# Importing necessary libraries for hierarchical clustering
import scipy as sc
import scipy.cluster.hierarchy as sch # for creating dendrogram 
from scipy.cluster.hierarchy import linkage # For clustering
# p = np.array(data_norm) # converting into numpy array format 

# Cluster performing
z = linkage(data_normal, method="complete",metric="euclidean")

# Dendrogram Visualisation
plt.figure(figsize=(20, 10));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z,leaf_rotation=0.,leaf_font_size=7.,)# rotates the x axis labels # font size for the x axis labels
plt.show()

# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(data_normal) 

cluster_labels=pd.Series(h_complete.labels_)

data['clust_hierarchical']=cluster_labels # creating a  new column and assigning it to new column 

data = data.iloc[:,[15,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]

### K MEANS CLUSTERING USING ORIGINAL DATA ####
#necessary libraries for K means clustering
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 
import numpy as np
###### screw plot or elbow curve ############
k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(data_normal)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(data_normal[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,data_normal.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=3) 
model.fit(data_normal)

model.labels_ # getting the labels of clusters assigned to each row 
mdl=pd.Series(model.labels_)  # converting numpy array into pandas series object 
data['clust_kmeans']=mdl # creating a  new column and assigning it to new column 

data = data.iloc[:,[16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]

### Comparingh the 3 type of clustering methods using mean##
data.iloc[:,4:].groupby(data.clust_pca).mean()
data.iloc[:,4:].groupby(data.clust_hierarchical).mean()
data.iloc[:,4:].groupby(data.clust_kmeans).mean()

### SAVING THE CSV FILE ###
import os
os.getcwd()
os.chdir("F:/pythonwd")

# creating a csv file 
data.to_csv("wine_clust.csv",encoding="utf-8")


