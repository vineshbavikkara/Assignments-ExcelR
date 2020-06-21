####### RONDOM FOREST ######

# Business problem : A Random Forest can be built with target variable Sales

# Reading the Data 
import pandas as pd
import numpy as np
data = pd.read_csv("F:\\Company_data.csv")
del data['Unnamed: 0']
data.head()
data.columns
colnames = list(data.columns)
predictors = colnames[1:]
target = colnames[0]
X = data[predictors]
Y = data[target]

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=15,criterion="entropy")
# n_estimators -> Number of trees ( you can increase for better accuracy)
# n_jobs -> Parallelization of the computing and signifies the number of jobs 
# running parallel for both fit and predict
# oob_score = True means model has done out of box sampling to make predictions

np.shape(data) # (400, 11) 

#### Attributes that comes along with RandomForest function
rf.fit(X,Y) # Fitting RandomForestClassifier model from sklearn.ensemble 
rf.estimators_ # 
rf.classes_ # class labels (output)
rf.n_classes_ # Number of levels in class labels 
rf.n_features_  # Number of input features in model 8 here.

rf.n_outputs_ # Number of outputs when fit performed

rf.oob_score_  # 0.76
rf.predict(X)

data['rf_pred'] = rf.predict(X)
data['Class_variable']= data.Sales
data["Class_variable"]
cols = ['rf_pred','Class_variable']
data[cols].head()

from sklearn.metrics import confusion_matrix
confusion_matrix(data['Class_variable'],data['rf_pred'])

pd.crosstab(data['Class_variable'],data['rf_pred'])

print("Accuracy is",(197+200)/(197+200+1+2)*100) # Accuracy is 99.25

###########################################################################################

# Business problem : A Random Forest can be built with target variable Sales
# Reading the Data 
import pandas as pd
import numpy as np
data = pd.read_csv("F:\\Fraud_ckeck_data.csv")
del data['Unnamed: 0']
data.head()
data.columns
colnames = list(data.columns)
predictors = colnames[1:]
target = colnames[0]
X = data[predictors]
Y = data[target]

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=16,criterion="gini")
np.shape(data) # (600, 6) 
#### Attributes that comes along with RandomForest function
rf.fit(X,Y) # Fitting RandomForestClassifier model from sklearn.ensemble 
rf.estimators_ # For getting estimators
rf.classes_ # class labels (output)
rf.n_classes_ # Number of levels in class labels 
rf.n_features_  # Number of input features in model 8 here.

rf.n_outputs_ # Number of outputs when fit performed

rf.oob_score_  # 0.73
rf.predict(X)

data['rf_pred'] = rf.predict(X)
data['Class_variable']= data.taxable_income
data["Class_variable"].head(10)
cols = ['rf_pred','Class_variable']
data[cols].head()
data[cols].tail(10)

from sklearn.metrics import confusion_matrix
confusion_matrix(data['Class_variable'],data['rf_pred'])
pd.crosstab(data['Class_variable'],data['rf_pred'])

print("Accuracy is",(476+109)/(476+109+0+15)*100) # Accuracy is 97.5

########################################################################################