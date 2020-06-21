######## K N N CLASSIFICATION ###########
# Business problem : Prepare a model for glass classification using KNN

# Importing Libraries 
import pandas as pd
import numpy as np
data = pd.read_csv("F:\\glass.csv")
data.Type.unique()
data['Type'].value_counts()

# creating a defenition for changing from number to glass type name.
def change_type(num):
    if num is 1:
        return 'building_windows_float_processed'
    if num is 2:
        return 'building_windows_non_float_processed'
    if num is 3:
        return 'vehicle_windows_float_processed'
    if num is 5:
        return 'containers'
    if num is 6:
        return 'tableware'
    else:
        return 'headlamps'
    
data.Type = data.Type.apply(change_type) # appllying the created defenition in Type variable

# Training and Test data using 
from sklearn.model_selection import train_test_split
train,test = train_test_split(data,test_size = 0.2) # 0.2 => 20 percent of entire data in to test data

# KNN using sklearn 
# Importing Knn algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC

# for 3 nearest neighbours 
neigh = KNC(n_neighbors= 3)

# Fitting with training data 
neigh.fit(train.iloc[:,0:9],train.iloc[:,9])

# train accuracy 
print ('train accuracy is',np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9])) # 80.7%
train_acc = np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9])

# test accuracy
print ('test accuracy is',np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9])) # 74.4 %
test_acc= np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9])

# for 5 nearest neighbours
neigh = KNC(n_neighbors=5)

# fitting with training data
neigh.fit(train.iloc[:,0:9],train.iloc[:,9])

# train accuracy 
train_acc = np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9]) #74.3%

# test accuracy
test_acc = np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9]) #74.4%

# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values 
 
for i in range(3,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:9],train.iloc[:,9])
    train_acc = np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9])
    test_acc = np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9])
    acc.append([train_acc,test_acc])

import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"bo-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"ro-")

########################################################################################
########################################################################################
# Business problem : Implement a KNN model to classify the animals in to categories

# Importing data 
import pandas as pd
data = pd.read_csv("F:\\zoo.csv")
data.type.unique()
data['type'].value_counts()
del data['animal name'] 

# Training and Test data spliting 
from sklearn.model_selection import train_test_split
train,test = train_test_split(data,test_size = 0.2)

# KNN using sklearn 
from sklearn.neighbors import KNeighborsClassifier as KNC
# for 3 nearest neighbours 
neigh = KNC(n_neighbors= 3)
# Fitting with training data 
neigh.fit(train.iloc[:,0:16],train.iloc[:,16])

# train accuracy 
import numpy as np
train_acc = np.mean(neigh.predict(train.iloc[:,0:16])==train.iloc[:,16]) #97.5%
# test accuracy
test_acc= np.mean(neigh.predict(test.iloc[:,0:16])==test.iloc[:,16]) # 95.2

# for 5 nearest neighbours
neigh = KNC(n_neighbors=5)

# fitting with training data
neigh.fit(train.iloc[:,0:16],train.iloc[:,16])

# train accuracy 
train_acc = np.mean(neigh.predict(train.iloc[:,0:16])==train.iloc[:,16]) #90%

# test accuracy
test_acc = np.mean(neigh.predict(test.iloc[:,0:16])==test.iloc[:,16]) #95.2%

# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 20 nearest neighbours(odd numbers) and 
# storing the accuracy values 
 
for i in range(3,20,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:9],train.iloc[:,9])
    train_acc = np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9])
    test_acc = np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9])
    acc.append([train_acc,test_acc])

# Accuracy plots 
import matplotlib.pyplot as plt # library to do visualizations 
plt.plot(np.arange(3,20,2),[i[0] for i in acc],"bo-") # train accuracy plot
plt.plot(np.arange(3,20,2),[i[1] for i in acc],"ro-") # test accuracy plot
##############################################################################
