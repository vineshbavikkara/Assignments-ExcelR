# Business problem:Perform Clustering for the crime data and identify the number of clusters formed and draw inferences.
##### HIERARCHICAL CLUSTERING #######
# Importing necessary libraries
import numpy as np
import pandas as pd 
import matplotlib.pylab as plt
import scipy as sc

# Importing the data set
data = pd.read_csv("F:/alldatasets/crime_data.csv")
data.columns

# Rename the columns
data.columns = ('city','murder','assault','urbanpop','rape')

# Identifyuing there have any null value
data.isnull().sum()

# Creating a new data with excluding the city names
crime= data
del data
crime_1= crime
del crime_1['city']

# Basic statistics
crime_1.describe() # thecking there have any scaling issue

# Normalizing the data for solving scaling issue
def norm_fun(i):
    x = (i- i.min()) / (i.max() - i.min())
    return x

#### Standardizing the data for solving scaling issue#####
def std_fun(i):
    x = (i - i.mean()) / (i.std())
    return x 

#Creating a normalised data frame
crime_norm = norm_fun(crime) # values lies betweenn 0 to 1 scale

# OR Creating a standardised dataframe
crime_std = std_fun(crime_1) # values lies between -3 to 3 scale

import scipy.cluster.hierarchy as sch # for creating dendrogram 
from scipy.cluster.hierarchy import linkage # For clustering
help(linkage)

type(crime_norm)
# p = np.array(data_norm) # converting into numpy array format 

# Cluster performing
z = linkage(crime_norm, method="complete",metric="euclidean")

# Dendrogram Visualisation
plt.figure(figsize=(20, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z,leaf_rotation=0.,leaf_font_size=7.,)# rotates the x axis labels # font size for the x axis labels
plt.show()

help(linkage)

# Now applying AgglomerativeClustering choosing 4 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=4,	linkage='complete',affinity = "euclidean").fit(crime_norm) 

cluster_labels=pd.Series(h_complete.labels_)

data = data = pd.read_csv("F:/alldatasets/crime_data.csv")
data.columns = ('city','murder','assault','urbanpop','rape')

data['clust']=cluster_labels # creating a  new column and assigning it to new column 

data = data.iloc[:,[5,0,1,2,3,4]]
data.head()

# getting aggregate mean of each cluster
data.iloc[:,2:].groupby(data.clust).median()


import os
os.getcwd()
os.chdir("F:/pythonwd")

# creating a csv file 
data.to_csv("Crime.csv",encoding="utf-8")

##############################################################################
#############################################################################
# Business Problem : Perform clustering (Both hierarchical and K means clustering) for the airlines data to obtain optimum number of clusters.Draw the inferences from the clusters obtained.
# Importing necessary libraries
import numpy as np
import pandas as pd 
import matplotlib.pylab as plt
import scipy as sc

# Importing the data set
data = pd.read_csv("F:/Airlines.csv")
data.columns

# Identifyuing there have any null value
data.isnull().sum()

# Basic statistics
data.describe() # thecking there have any scaling issue

# Normalizing the data for solving scaling issue
def norm_fun(i):
    x = (i- i.min()) / (i.max() - i.min())
    return x

#### Standardizing the data for solving scaling issue#####
def std_fun(i):
    x = (i - i.mean()) / (i.std())
    return x 

#Creating a normalised data frame excluding the dummy variable Award
data_norm = norm_fun(data.iloc[:,[1,2,3,4,5,6,7,8,9,10]]) # values lies betweenn 0 to 1 scale
data_norm['Award'] = data.Award # Attaching the excluded variableAward wiith normalisedd data frame

# OR Creating a standardised dataframe
data_std = std_fun(data.iloc[:,1:11]) # values lies between -3 to 3 scale

import scipy.cluster.hierarchy as sch # for creating dendrogram 
from scipy.cluster.hierarchy import linkage # For clustering
help(linkage)

# Cluster performing
z = linkage(data_norm, method="complete",metric="euclidean")

# Dendrogram Visualisation
sch.dendrogram(z,leaf_rotation=0.,leaf_font_size=7.,)# rotates the x axis labels # font size for the x axis labels
plt.figure(figsize=(20, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
plt.show()

help(linkage)

# Now applying AgglomerativeClustering choosing 4 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(data_norm) 

cluster_labels=pd.Series(h_complete.labels_)

data['cluster']=cluster_labels # creating a  new column and assigning it to new column 

data = data.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
data.head()

# getting aggregate mean of each cluster
data.iloc[:,2:].groupby(data.cluster).median()
d=pd.DataFrame(data.iloc[:,2:].groupby(data.cluster).median())

import os
os.getcwd()
os.chdir("F:/pythonwd")

# creating a csv file 
data.to_csv("Airlines.csv",encoding="utf-8")

##############################################################################
########## K- MEANS CLUSTRING##########################
import pandas as pd
import matplotlib.pylab as plt
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 
import numpy as np

# Generating random uniform numbers 
X = np.random.uniform(0,1,1000)
Y = np.random.uniform(0,1,1000)
df_xy =pd.DataFrame(columns=["X","Y"])
df_xy.X = X
df_xy.Y = Y
df_xy.plot(x="X",y = "Y",kind="scatter")
model1 = KMeans(n_clusters=5).fit(df_xy)
model1.labels_
model1.cluster_centers_
df_xy.plot(x="X",y = "Y",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)


# Kmeans on  Data set 
data = pd.read_csv("F:\\Airlines.csv")
# Normalization function 
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
data_norm = norm_func(data.iloc[:,1:11])
data_norm["Award"] = data.Award 

data_norm.head(10)  # Top 10 rows

###### screw plot or elbow curve ############
k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(data_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(data_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,data_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=5) 
model.fit(data_norm)

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
data['clust']=md # creating a  new column and assigning it to new column 
data.head()

data = data.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]

data.iloc[:,2:].groupby(data.clust).mean()

data.to_csv("Crime_cluster.csv")
################################################################################
