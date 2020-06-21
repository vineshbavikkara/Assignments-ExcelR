import pandas as pd 
import numpy  as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sb

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split # train and test split
from sklearn import metrics
from sklearn import preprocessing


#Importing Data
bank = pd.read_csv('F:/alldatasets/bankdatanew.csv')
bank.head()

# creating dummy columns for the categorical columns 
bank.columns
bank_dummies = pd.get_dummies(bank[["job","marital","education","default","housing","loan","contact","month","poutcome","y"]])
# Dropping the columns for which we have created dummies
bank.drop(["job","marital","education","default","housing","loan","contact","month","poutcome"],inplace=True,axis = 1)
bank.drop(["y"],inplace =True,axis=1)
bank_dummies.drop(['y_no'],inplace= True,axis=1)

# adding the columns to the bank data frame 
bank = pd.concat([bank,bank_dummies],axis=1)

# checking there have any null value
bank.isnull().sum() 

# E D A
# Getting the barplot for the categorical columns
#Importing the bank data again to getting the barplot of the categorical data
data= pd.read_csv("F:/alldatasets/bankdatanew.csv")

sb.countplot(x="y",data=data,palette="hls")# Checking the output variable data is balanced or not

sb.countplot(x="job",data=data,palette="hls")
pd.crosstab(data.job,data.y).plot(kind="bar")

sb.countplot(x="marital",data=data,palette="hls")
pd.crosstab(data.marital,data.y).plot(kind="bar")

sb.countplot(x="education",data=data,palette="hls")
pd.crosstab(data.education,data.y).plot(kind="bar")

sb.countplot(x="default",data=data,palette="hls")
pd.crosstab(data.default,data.y).plot(kind="bar")

sb.countplot(x="housing",data=data,palette="hls")
pd.crosstab(data.housing,data.y).plot(kind="bar")

sb.countplot(x="loan",data=data,palette="hls")
pd.crosstab(data.loan,data.y).plot(kind="bar")

sb.countplot(x="contact",data=data,palette="hls")
pd.crosstab(data.contact,data.y).plot(kind="bar")

sb.countplot(x="month",data=data,palette="hls")
pd.crosstab(data.month,data.y).plot(kind="bar")

sb.countplot(x="poutcome",data=data,palette="hls")
pd.crosstab(data.poutcome,data.y).plot(kind="bar")


# Data Distribution - Boxplot of continuous variables with respect to each category of categorical columns

sb.boxplot(x="y",y="age",data=data,palette="hls")
sb.boxplot(x="y",y="balance",data=data,palette="hls")
sb.boxplot(x="y",y="duration",data=data,palette="hls")
sb.boxplot(x="y",y="day",data=data,palette="hls")
sb.boxplot(x="y",y="campaign",data=data,palette="hls")
sb.boxplot(x="y",y="pdays",data=data,palette="hls")
sb.boxplot(x="y",y="previous",data=data,palette="hls")

sb.boxplot(x="job",y="age",data=data,palette="hls")
sb.boxplot(x="job",y="balance",data=data,palette="hls")
sb.boxplot(x="job",y="duration",data=data,palette="hls")
sb.boxplot(x="job",y="day",data=data,palette="hls")
sb.boxplot(x="job",y="campaign",data=data,palette="hls")
sb.boxplot(x="job",y="pdays",data=data,palette="hls")
sb.boxplot(x="job",y="previous",data=data,palette="hls")

# chgnging column names 
bank['y'] = bank['y_yes']
del bank['y_yes']
bank['job_admin']=bank['job_admin.']
del bank['job_admin.']
bank['job_blue_collar']= bank['job_blue-collar']
del bank['job_blue-collar']
bank['job_self_employed']=bank['job_self-employed']
del bank['job_self-employed']

#### Model building [WAY 1] ######
import statsmodels.formula.api as sm
bank.columns

logit_model = sm.logit('y ~ age+balance+day+duration+campaign+pdays+previous+job_admin+job_blue_collar+job_entrepreneur+job_housemaid+job_management+job_retired+job_self_employed+job_services+job_student+job_technician+job_unemployed+job_unknown+marital_divorced+marital_married+marital_single+education_primary+education_secondary+education_tertiary+education_unknown+default_no+default_yes+housing_no+housing_yes+loan_no+loan_yes+contact_cellular+contact_telephone+contact_unknown+month_apr+month_aug+month_dec+month_feb+month_jan+month_jul+month_jun+month_mar+month_may+month_nov+month_oct+month_sep+poutcome_failure+poutcome_other+poutcome_success+poutcome_unknown',data = data).fit()
logit_model.summary()

#summary
logit_model.summary()
y_pred = logit_model.predict(bank)
y_pred
bank["pred_prob"] = y_pred
# Creating new column for storing predicted class of y (output variable)
# filling all the cells with zeroes
bank["y_val"] = np.zeros(45211)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
bank.loc[y_pred>=0.5,"y_val"] = 1
bank.y_val

from sklearn.metrics import classification_report
classification_report(bank.y_val,bank.y)

# confusion matrix 
confusion_matrix = pd.crosstab(bank['y'],bank.y_val)
confusion_matrix
accuracy = (38940+1833)/(45211) # 90.2
accuracy

# ROC curve 
from sklearn import metrics
# fpr => false positive rate
# tpr => true positive rate
fpr, tpr, threshold = metrics.roc_curve(bank.y, y_pred)


# the above function is applicable for binary classification class 

plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
 
roc_auc = metrics.auc(fpr, tpr) # area under ROC curve 
roc_auc
#################################################################################
#### Model building [WAY 2] ######
from sklearn.linear_model import LogisticRegression
bank.shape
X = bank.iloc[:,0:51]
Y = bank.iloc[:,51]
classifier = LogisticRegression()
classifier.fit(X,Y)

classifier.coef_ # coefficients of features 
classifier.predict_proba (X) # Probability values 

y_pred = classifier.predict(X)
y_pred
a = pd.DataFrame(y_pred)
bank["y_pred"] = y_pred
y_prob = pd.DataFrame(classifier.predict_proba(X.iloc[:,:]))
new_df = pd.concat([bank,y_prob],axis=1)
new_data= pd.concat([bank,y_pred],axis=1)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y,y_pred)
print (confusion_matrix)
type(y_pred)
accuracy = sum(Y==y_pred)/bank.shape[0]
pd.crosstab(y_pred,Y)

################################################################################
### Dividing data into train and test data sets
from sklearn.model_selection import train_test_split

train,test = train_test_split(bank,test_size=0.3)

# checking na values 
train.isnull().sum();test.isnull().sum()

# Building a model on train data set 

train_model = sm.logit('y ~ age+balance+day+duration+campaign+pdays+previous+job_admin+job_blue_collar+job_entrepreneur+job_housemaid+job_management+job_retired+job_self_employed+job_services+job_student+job_technician+job_unemployed+job_unknown+marital_divorced+marital_married+marital_single+education_primary+education_secondary+education_tertiary+education_unknown+default_no+default_yes+housing_no+housing_yes+loan_no+loan_yes+contact_cellular+contact_telephone+contact_unknown+month_apr+month_aug+month_dec+month_feb+month_jan+month_jul+month_jun+month_mar+month_may+month_nov+month_oct+month_sep+poutcome_failure+poutcome_other+poutcome_success+poutcome_unknown',data = train).fit()

#summary
train_model.summary()
train_pred = train_model.predict(train)
train_pred

# Creating new column for storing predicted class of y variable (output variable)

# filling all the cells with zeroes
train.shape[0] # for finding the number of raws
train["train_pred"] = np.zeros(31647)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
train.loc[train_pred>0.5,"train_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(train['y'],train.train_pred)
confusion_matrix
accuracy_train = (27254+1286)/(train.shape[0]) 
accuracy_train  # 0.9018

# Prediction on Test data set
test_pred = train_model.predict(test)

# Creating new column for storing predicted class of y

# filling all the cells with zeroes
test.shape[0]
test["test_pred"] = np.zeros(13564)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
test.loc[test_pred>0.5,"test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test['y'],test.test_pred)
confusion_matrix
accuracy_test = (11693+544)/(test.shape[0]) 
accuracy_test # 90.2%

#####################################################################################