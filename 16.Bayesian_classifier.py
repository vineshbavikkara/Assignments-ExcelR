##### Bayesian Classifier #######

# Business problem: Prepare a classification model using Naive Bayes for salary data 

# Importing the Data
import pandas as pd
salary_train = pd.read_csv("F:\\SalaryData_Train.csv")
salary_test = pd.read_csv("F:\\SalaryData_Test.csv")

# Assigning numbers to categorical variables
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]
from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in string_columns:
    salary_train[i] = number.fit_transform(salary_train[i])
    salary_test[i] = number.fit_transform(salary_test[i])

# deleting variable Same type of variable educationno, considering only education because these are same meaning variables
del salary_train['educationno'],salary_test['educationno']

# checking train data balanced or imbalanced data 
salary_train['Salary'].value_counts() # it is imbalanced data

# Making Imbalanced data to balanced data ; 1. Over sampling 2. Under sampling 3.SMOTE
# Here using Under sampling technique

#1. sorting in Ascending order
salary_train_sorted= salary_train.sort_index(by=['Salary'], ascending=[True])
salary_train_sorted  = salary_train_sorted.reset_index(drop=True)
salary_train_sorted.Salary.value_counts() # <=50K => 22653 and >50K =>7508 then 22653-7508 =15145

#2. removing the 15145 data of <=50K from Salary variable. And clearing the imbalaced problem of data 
salary_train_bal = salary_train_sorted.iloc[15145:,:]
salary_train_bal = salary_train_bal.reset_index(drop=True) # reset index
salary_train_bal.Salary.value_counts() # now its a balanced data

'''#For SMOTE #
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
generate the data
X,y = make_classification(n_classes = 2, weights=[0.1,0.9], n_features=20, n_samples=30161)
apply the SMOTE over-sampling
sm = SMOTE(ratio = 'auto',kind = 'regular')
X_resampled, y_resampled = sm.fit.sample(X, y)'''

# Creating train input ,train output and test input , test output
colnames = salary_train_bal.columns
len(colnames[0:13])
trainX = salary_train_bal[colnames[0:12]]
trainY = salary_train_bal[colnames[12]]
testX  = salary_test[colnames[0:12]]
testY  = salary_test[colnames[12]]

# Gaussian Naive bayes classification
from sklearn.naive_bayes import GaussianNB as GB
sgnb = GB()
spred_gnb = sgnb.fit(trainX,trainY).predict(testX)
# Accuracy checking
from sklearn.metrics import confusion_matrix
confusion_matrix(testY,spred_gnb)
print ("Accuracy is :",(10690+1174)/(10690+1174+2526+670)) # 78.78%

from sklearn.naive_bayes import MultinomialNB as MB
smnb = MB()
spred_mnb = smnb.fit(trainX,trainY).predict(testX)
# Accuracy checking
confusion_matrix(testY,spred_mnb)
print("Accuracy is :",(10891+780)/(10891+780+2920+469))  # 77.5%  #There have no changes visible in the  test accuracy after performing Multinomial NB for classification by using imbalanced data or balanced data

#############################################################################################
# Business problem : Build a naive Bayes model on the data set for classifying the ham and spam
# Loading textual data
import pandas as pd
import numpy as np
email_data = pd.read_csv("F:\\sms_raw_NB.csv",encoding = "ISO-8859-1")

# cleaning data 
import re
stop_words = []
with open("F:\\stop_words.txt") as f:
    stop_words = f.read()

# splitting the entire string by giving separator as "\n" to get list of 
# all stop words
stop_words = stop_words.split("\n")

# cleaning data
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

email_data.text = email_data.text.apply(cleaning_text) # applyin def on mail data

# removing empty rows 
email_data.shape
email_data = email_data.loc[email_data.text != " ",:]

# CountVectorizer
# Convert a collection of text documents to a matrix of token counts

# TfidfTransformer
# Transform a count matrix to a normalized tf or tf-idf representation.

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split
email_train,email_test = train_test_split(email_data,test_size=0.3)

# Preparing email texts into word count matrix format 
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
emails_bow = CountVectorizer(analyzer=split_into_words).fit(email_data.text)

# For all messages
all_emails_matrix = emails_bow.transform(email_data.text)
all_emails_matrix.shape # (5559,6661)

# For training messages
train_emails_matrix = emails_bow.transform(email_train.text)
train_emails_matrix.shape # (3891,6661)

# For testing messages
test_emails_matrix = emails_bow.transform(email_test.text)
test_emails_matrix.shape # (1668,6661)

####### Without TFIDF matrices ############
# Preparing a naive bayes model on training data set 

# Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB as MB
classifier_mb = MB()
classifier_mb.fit(train_emails_matrix,email_train.type)
train_pred_m = classifier_mb.predict(train_emails_matrix)
# train accuracy
print ('Accuracy is:',np.mean(train_pred_m==email_train.type))# 98.95%
accuracy_train_m = np.mean(train_pred_m==email_train.type)
# test accuracy
test_pred_m = classifier_mb.predict(test_emails_matrix)
accuracy_test_m = np.mean(test_pred_m==email_test.type) # 96.64%

# Gaussian Naive Bayes 
from sklearn.naive_bayes import GaussianNB as GB
classifier_gb = GB()
classifier_gb.fit(train_emails_matrix.toarray(),email_train.type.values) # we need to convert tfidf into array format which is compatible for gaussian naive bayes
train_pred_g = classifier_gb.predict(train_emails_matrix.toarray())
# train accuracy
accuracy_train_g = np.mean(train_pred_g==email_train.type) # 90.67%
# test accuracy
test_pred_g = classifier_gb.predict(test_emails_matrix.toarray())
accuracy_test_g = np.mean(test_pred_g==email_test.type) # 84.29%

