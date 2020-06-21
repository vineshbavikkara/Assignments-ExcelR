####### SUPPORT VECTOR MACHINE (S V M)########
# Business problem : classify the Size_Categorie using SVM
# Import Dataset
import pandas as pd
data = pd.read_csv("F:\\forestfires.csv")
del data['month'],data['day']
data.columns
data.head()
data.describe()

# checking output variable size_category
data.size_category.unique()
data.size_category.value_counts() # Imbalanced data

import seaborn as sns
sns.boxplot(x="size_category",y="FFMC",data=data,palette = "hls")
sns.boxplot(x="DMC",y="size_category",data=data,palette = "hls")
sns.boxplot(x="DC",y="size_category",data=data)
sns.boxplot(x="ISI",y="size_category",data=data,palette = "hls")
sns.boxplot(x="temp",y="size_category",data=data,palette = "hls")

from sklearn.model_selection import train_test_split
train,test = train_test_split(data,test_size = 0.2)
test.head()
train_X = train.iloc[:,0:28]
train_y = train.iloc[:,28]
test_X  = test.iloc[:,0:28]
test_y  = test.iloc[:,28]

from sklearn.svm import SVC
help(SVC)
# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

# kernel = linear
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

import numpy as np
np.mean(pred_test_linear==test_y) # Accuracy = 99.038

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)

np.mean(pred_test_poly==test_y) # Accuracy = 99.038

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y) # Accuracy = 73.076

# kernel = sigmoid
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(train_X,train_y)
pred_test_sig = model_sig.predict(test_X)

np.mean(pred_test_sig==test_y) # Accuracy = 71.15

# Conclusion : selecting model_lineaar for classification.
####################################################################################
####################################################################################

# Business problem : Prepare a classification model using SVM for salary data

# Importing Dataset
import pandas as pd
train = pd.read_csv("F:\\SalaryData_Train.csv")
test = pd.read_csv("F:\\SalaryData_Test.csv")
del train['educationno'], test['educationno']

# checking data type of all variables
col=pd.DataFrame(train.columns)
col_dtypes = pd.DataFrame(train.dtypes)

train = train.iloc[:,[12,11,7,6,5,4,3,2,1,10,9,8,0]]
test = test.iloc[:,[12,11,7,6,5,4,3,2,1,10,9,8,0]]

# For summary
train.describe()

# creating Dummy variable for categorical variables on train data
train_dummy_native= pd.get_dummies(train.native,prefix='native',prefix_sep='_')
train_dummy_sex = pd.get_dummies(train.sex,prefix='sex',prefix_sep='_')
train_dummy_race = pd.get_dummies(train.race,prefix='race',prefix_sep='_')
train_dummy_relationship = pd.get_dummies(train.relationship,prefix='relationship',prefix_sep='_')
train_dummy_occupation = pd.get_dummies(train.occupation,prefix='occupation',prefix_sep='_')
train_dummy_maritalstatus = pd.get_dummies(train.maritalstatus,prefix='maritalstatus',prefix_sep='_')
train_dummy_education = pd.get_dummies(train.education,prefix='education',prefix_sep='_')
train_dummy_workclass = pd.get_dummies(train.workclass,prefix='workclass',prefix_sep='_')

# joining the dummy variables with train data
train = train.join([train_dummy_native,train_dummy_sex,train_dummy_race
              ,train_dummy_relationship,train_dummy_occupation
              ,train_dummy_maritalstatus,train_dummy_education,train_dummy_workclass])

# deleting the categorical variable from train data
train.dtypes
del train["native"],train["sex"],train["race"],train["relationship"],train["occupation"],train["maritalstatus"],train["education"],train["workclass"]

# creating Dummy variable for categorical variables on test data
test_dummy_native= pd.get_dummies(test.native,prefix='native',prefix_sep='_')
test_dummy_sex = pd.get_dummies(test.sex,prefix='sex',prefix_sep='_')
test_dummy_race = pd.get_dummies(test.race,prefix='race',prefix_sep='_')
test_dummy_relationship = pd.get_dummies(test.relationship,prefix='relationship',prefix_sep='_')
test_dummy_occupation = pd.get_dummies(test.occupation,prefix='occupation',prefix_sep='_')
test_dummy_maritalstatus = pd.get_dummies(test.maritalstatus,prefix='maritalstatus',prefix_sep='_')
test_dummy_education = pd.get_dummies(test.education,prefix='education',prefix_sep='_')
test_dummy_workclass = pd.get_dummies(test.workclass,prefix='workclass',prefix_sep='_')

# joining the dummy variables with test data
test = test.join([test_dummy_native,test_dummy_sex,test_dummy_race
              ,test_dummy_relationship,test_dummy_occupation
              ,test_dummy_maritalstatus,test_dummy_education,test_dummy_workclass])

# deleting the categorical variable from test data
test.dtypes
del test["native"],test["sex"],test["race"],test["relationship"],test["occupation"],test["maritalstatus"],test["education"],test["workclass"]

# checking output variable; Salary
train.Salary.unique()
train.Salary.value_counts() # Imbalanced data

# save CSV file to cwd
import os
os.getcwd()
train.to_csv("trainn.csv",encoding = "utf-8")
test.to_csv("testt.csv", encoding="utf-8")

# EDA Part : Boxplot representation
import seaborn as sns
train.columns
sns.boxplot(x="age",y="Salary",data=train,palette = "hls")
sns.boxplot(x="hoursperweek",y="Salary",data=train,palette = "hls")
sns.boxplot(x="capitalloss",y="Salary",data=train,palette = "hls")
sns.boxplot(x="capitalgain",y="Salary",data=train,palette = "hls")

# train_X,train_y and test_X,test_y split
train_X = train.iloc[:,1:]
train_y = train.iloc[:,0]
test_X  = test.iloc[:,1:]
test_y  = test.iloc[:,0]

from sklearn.svm import SVC
help(SVC)
# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid'

# kernel = linear
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

import numpy as np
np.mean(pred_test_linear==test_y)

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)

np.mean(pred_test_poly==test_y) 

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y)  

# kernel = sigmoid
model_sig = SVC(kernel = "sigmoid")
model_sig.fit(train_X,train_y)
pred_test_sig = model_sig.predict(test_X)

np.mean(pred_test_sig==test_y) 

#################################################################################