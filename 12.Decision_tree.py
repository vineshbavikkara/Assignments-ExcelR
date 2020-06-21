##### DECISION TREE CLASSIFIER #######
#Business problem: Decision tree on Company data 
# Importing the data
import pandas as pd
data = pd.read_csv("F:\\Company_Data.csv")
data.columns
data.head()

# Sorting the data in ascending order for the splitting of the target(dependent) variable.
data=data.sort_index(by=['Sales'],ascending=True)
data = data.reset_index(drop=True)

# splitting the target Variable in to 2 parts
def split(num):
    if num < 7.5:
        return 'low_sale'
    else:
        return 'high_sale'
data.Sales = data.Sales.apply(split) # Applying split defenition on target variable
data.Sales.unique()
data.Sales.value_counts()

##converting the categorical variable into numeric
#converting Shelveloc into 0,1 and 2. Bad=0,Medium=1 and Good=2
data.ShelveLoc.unique()
data.ShelveLoc.value_counts()
def convert_ShelveLoc(txt):
    if 'Bad' in txt:
        return 0
    if 'Medium' in txt:
        return 1
    else:
        return 2
data.ShelveLoc = data.ShelveLoc.apply(convert_ShelveLoc) 

#converting urban variable in to 0 and 1. if yes =1 and No =0
data.Urban.value_counts()
data.Urban.unique()
def convert_yes_no(txt):
    if 'No' in txt:
        return 0
    else:
        return 1

data.Urban = data.Urban.apply(convert_yes_no)
#converting US variable in to 0 and 1. if yes =1 and No =0
data.US.value_counts()
data.US = data.US.apply(convert_yes_no)

# working directory details
import os
os.getcwd()
os.chdir("F:/pythonwd")

# creating a csv file 
data.to_csv("Company_data.csv",encoding="utf-8")


# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train,test = train_test_split(data,test_size = 0.2)

from sklearn.tree import  DecisionTreeClassifier
help(DecisionTreeClassifier)

colnames = list(data.columns)
predictors = colnames[1:]
target = colnames[0]

model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])
preds = model.predict(test[predictors])
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)

# Accuracy = train
import numpy as np
np.mean(train.Sales == model.predict(train[predictors]))

# Accuracy = Test
np.mean(preds==test.Sales)

# tree visualisation
from sklearn import tree
tree.plot_tree(model)
 
################################################################################################
################################################################################################
#Business problem: Decision tree on Fraud check data 
#importng data
import pandas as pd
data = pd.read_csv("F:/Fraud_check.csv")
data.head()
print('column names :',data.columns)
data.columns =('undergrad','marital_status','taxable_income','city_population','work_exp','urban')

# splitting the target data taxable_income
def split_target(num):
    if num <=30000:
        return 'Risky'
    else:
        return 'Good'
    
data.taxable_income = data.taxable_income.apply(split_target)
data.taxable_income.value_counts()

# splitting the Yes_No category assigning Yes=1 and No=0
def convert_yes_no(txt):
    if 'NO' in txt:
        return 0
    else:
        return 1
    
data.undergrad = data.undergrad.apply(convert_yes_no) # Converting undergrad variable
data.undergrad.value_counts()
data.urban = data.urban.apply(convert_yes_no) # Converting the urban variable
data.urban.value_counts()

# splitting the marital_status variable from category to numeric assigning Single=1, Divorced=0, Married=2
def convert_text(txt):
    if 'Single' in txt:
        return 1
    if 'Divorced'in txt:
        return 0
    else:
        return 2
data.marital_status= data.marital_status.apply(convert_text)

data=data.iloc[:,[2,0,1,3,4,5]]

# Create a csv data file
import os
os.getcwd()
data.to_csv("Fraud_ckeck_data.csv",encoding="utf-8")

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train,test = train_test_split(data,test_size = 0.2)


from sklearn.tree import  DecisionTreeClassifier
help(DecisionTreeClassifier)
colnames = list(data.columns)
predictors = colnames[1:]
target = colnames[0]
model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])
preds = model.predict(test[predictors])
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)

# Accuracy = train
import numpy as np
np.mean(train.taxable_income == model.predict(train[predictors])) # 100%

# Accuracy = Test
np.mean(preds==test.taxable_income) # 67.5%

# tree visualisation
from sklearn import tree
tree.plot_tree(model)

# building a new model after channging criterion from entropy to gini
model1 = DecisionTreeClassifier(criterion = 'gini')
model1.fit(train[predictors],train[target])
preds = model1.predict(test[predictors])
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)

# Accuracy = train
import numpy as np
np.mean(train.taxable_income == model.predict(train[predictors])) # 100%

# Accuracy = Test
np.mean(preds==test.taxable_income) # 69.5%

# tree visualisation
from sklearn import tree
tree.plot_tree(model1)
