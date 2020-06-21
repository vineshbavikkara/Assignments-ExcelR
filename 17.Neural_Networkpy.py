######## Neural Network ##########

# Business problem : Prepare a model for strength of concrete data using Neural Networks

# Reading data 
import numpy as np
import pandas as pd
Concrete = pd.read_csv("F:\\concrete_data.csv")
Concrete.head()
Concrete.isnull().sum()

# Plots and Correlation between variables for each independent variables 
import seaborn as sns
sns.boxplot(Concrete.cement)
sns.boxplot(Concrete.slag)
sns.boxplot(Concrete.ash)
sns.boxplot(Concrete.water)
sns.boxplot(Concrete.superplastic)
sns.boxplot(Concrete.age)
# Scatter plot between all possible independent variables and histogram for each independent variable
sns.pairplot(Concrete)
Concrete.hist()
# Correlation values 
Concrete.corr()

column_names = list(Concrete.columns)
predictors = column_names[0:8]
target = column_names[8]

'''
##### ####
# Model Building II
#MLP CLASSIFIER
# Creating X and Y
Y = np.asarray(Concrete[target], dtype="|S6")
Y.shape
X = np.asarray(Concrete[predictors])
X.shape

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(50), random_state=None)
clf.fit(X,Y)
pred_values = clf.predict(X)
pred_values = int(pred_values)
np.sqrt(np.mean((pred_values-Concrete[target])))
#### #####
'''
# Model Building I
# conda install -c hesi_m keras --> new version 
# pip install keras # old version
# pip instal tensorflow 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation,Layer,Lambda

def prep_model(hidden_dim):
    model = Sequential()
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],kernel_initializer="normal",activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    model.add(Dense(hidden_dim[-1]))
    model.compile(loss="mean_squared_error",optimizer="adam",metrics = ["accuracy"])
    return (model)

first_model = prep_model([8,50,1])
first_model.fit(np.array(Concrete[predictors]),np.array(Concrete[target]),epochs=900)
pred_train = first_model.predict(np.array(Concrete[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])

# RMSE
rmse_value = np.sqrt(np.mean((pred_train-Concrete[target])**2)) # RMSE is 4.99
# Accuracy
import matplotlib.pyplot as plt
plt.plot(pred_train,Concrete[target],"bo")
np.corrcoef(pred_train,Concrete[target]) # r = 0.947, we got high correlation 

#################################################################################################

# Business problem : Build a Neural Network model for 50_startups data to predict profit 

#Import data
import pandas as pd
file = "F:\\50_Startups.csv"
data = pd.read_csv(file)
data.columns
# Assigning numbers to categorical variable; State
data['State'].unique()
string_columns=["State"]
from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in string_columns:
    data[i] = number.fit_transform(data[i]) # 0 for california, 1 for florida and 2 for New York

# Plots and Correlation between variables for each independent variables 
import seaborn as sns
sns.boxplot(data['R&D Spend'])
sns.boxplot(data.Administration)
sns.boxplot(data['Marketing Spend'])
sns.boxplot(data.State)
sns.boxplot(data.Profit)

# checking the categorical variable is balanced or not
data.State.value_counts() # balanced data

# Scatter plot between all possible independent variables and histogram for each independent variable
sns.pairplot(data)
data.hist()
# Correlation values 
data.corr()

column_names = list(data.columns)
predictors = column_names[0:4]
target = column_names[4]

'''
##### ####
# Model Building II
#MLP CLASSIFIER
# Creating X and Y
Y = np.asarray(Concrete[target], dtype="|S6")
Y.shape
X = np.asarray(Concrete[predictors])
X.shape

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(50), random_state=None)
clf.fit(X,Y)
pred_values = clf.predict(X)
pred_values = int(pred_values)
np.sqrt(np.mean((pred_values-Concrete[target])))
#### #####
''''

# Model Building 
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation,Layer,Lambda

def prep_model(hidden_dim):
    model = Sequential()
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],kernel_initializer="normal",activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    model.add(Dense(hidden_dim[-1]))
    model.compile(loss="mean_squared_error",optimizer="adam",metrics = ["accuracy"])
    return (model)

first_model = prep_model([4,500,1]) 
first_model.fit(np.array(data[predictors]),np.array(data[target]),epochs=1000)
pred_train = first_model.predict(np.array(data[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])

# RMSE
rmse_value = np.sqrt(np.mean((pred_train-data[target])**2)) # RMSE is 11076

second_model = prep_model([4,2000,1]) 
second_model.fit(np.array(data[predictors]),np.array(data[target]),epochs=1500)
pred_train = second_model.predict(np.array(data[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])

# RMSE
rmse_value = np.sqrt(np.mean((pred_train-data[target])**2)) # RMSE is 9198

# Accuracy
import matplotlib.pyplot as plt
plt.plot(pred_train,data[target],"bo")
np.corrcoef(pred_train,data[target]) # r = 0.98, we got high correlation

##############################################################################################

# Business problem: Predict the burned are of forest fires with Neural Network.

# import data
import pandas as pd
data = pd.read_csv('F:/forestfires.csv', sep = ',')
data.drop(data.iloc[:,11:30],axis=1,inplace=True) # Dropping the uncessary column 
data.head()

#dealing categorical variables 
data.month.unique()
def change(txt):
    if 'jan' in txt:
        return 0
    if 'feb' in txt:
        return 1
    if 'mar' in txt:
        return 2
    if 'apr' in txt:
        return 3
    if 'may' in txt:
        return 4
    if 'jun' in txt:
        return 5
    if 'jul' in txt:
        return 6
    if 'aug' in txt:
        return 7
    if 'sep' in txt:
        return 8
    if 'oct' in txt:
        return 9
    if 'nov' in txt:
        return 10
    else :
        return 11
data['month'] = data['month'].apply(change)
data['month'].value_counts()

data.day.unique()
def day_to_number(txt):
    if 'sun' in txt:
        return 0
    if 'mon' in txt:
        return 1
    if 'tue' in txt:
        return 2
    if 'wed' in txt:
        return 3
    if 'thu' in txt:
        return 4
    if 'fri' in txt:
        return 5
    else :
        return 6
data['day'] = data['day'].apply(day_to_number)
data['day'].value_counts()


data.columns
data.isnull().sum()
data.isnull().values.any() # No missing values 

# Checking the category output variable
data.size_category.unique()
#  small as 0 and large as 1
data.loc[data.size_category=="small","size_category"] = 0
data.loc[data.size_category=="large","size_category"] = 1

# EDA
import matplotlib.pylab as plt
data.columns
plt.boxplot(data.FFMC)
plt.boxplot(data.DMC)
plt.boxplot(data.DC)
plt.boxplot(data.ISI)
plt.boxplot(data.temp)
plt.boxplot(data.RH)
plt.boxplot(data.wind)
plt.boxplot(data.rain)
plt.boxplot(data.atrea)
plt.hist(data.size_category, kind = 'bar')
plt.hist(data.month)
plt.hist(data.day)
import seaborn as sns
sns.pairplot(data)
data.hist()

# checking it is imbalanced or not
data.size_category.value_counts() # it is imbalanced data
data.size_category.value_counts().plot(kind="bar") # Plot

## Immplimenting Over sampling technique for Handling imbalanced
# importing dependent and independent features
columns = data.columns.tolist()
#Store the value we are predicting
target = "size_category"
# filter the columns to remove data we do not want
columns = [c for c in columns if c not in ['size_category']]
#Define a random state
import numpy as np
state = np.random.RandomState(42)
X = data[columns] 
y = pd.DataFrame(data[target])
X.shape #(517, 28)
y.shape #(517,)

small = data[data['size_category']==0]
large = data[data['size_category']==1]
small.shape #(378, 29)
large.shape #(139, 29)

# for over sampling technique
from sklearn.datasets import make_classification as mc
from imblearn.combine import SMOTETomek
X, y = mc(n_samples=517,n_classes=2,weights=[0.8,0.2], n_features=11,
          shift=0.0, scale=1.0,shuffle=True, random_state=None)
smk = SMOTETomek(random_state=42)
X_res,y_res = smk.fit_resample(X,y)
bal_X = pd.DataFrame(X_res)
bal_y = pd.DataFrame(y_res)
# shape of new  balanced data
print('Original dataset shape is',data.shape) # shape of imbalanced data
bal_y.columns = ["size_category"]
small = bal_y[bal_y['size_category']==0]
large = bal_y[bal_y['size_category']==1]
small.shape #(410, 1)
large.shape #(410, 1)
# join the X and y 
data_bal = bal_X.join(bal_y)
data_bal.shape # (820, 12)

# Train, Test split
from sklearn.model_selection import train_test_split
train,test = train_test_split(data_bal,test_size = 0.3,random_state=42)
trainX = train.drop(["size_category"],axis=1)
trainY = train["size_category"]
testX = test.drop(["size_category"],axis=1)
testY = test["size_category"]

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(820,100))

mlp.fit(trainX,trainY)
prediction_train=mlp.predict(trainX)
prediction_test = mlp.predict(testX)

# Accuracy check
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(testY,prediction_test))
np.mean(trainY==prediction_train) # train accuracy is 1
np.mean(testY==prediction_test) # test accuracy is 0.96
################################################################################################
