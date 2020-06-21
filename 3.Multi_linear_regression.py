# Importing nesessari libraries
import numpy as np
import pandas as pd 
import matplotlib.pylab as plt
import statsmodels.formula.api as smf
import seaborn as sns

# Importing the dataset
data = pd.read_csv("F:/alldatasets/50_Startups.csv")
#Editing column names
data.columns=('RandD','admin','marketing','state','profit')
# Deleting  variable called state
del data['state']

# EDA

#Boxplot
sns.boxplot(data['r&d'])
sns.boxplot(data['admin'])
sns.boxplot(data['marketing'])
sns.boxplot(data['profit'])

# Removing 0 value data rows
data=data.drop(data.index[[19,47]],axis=0)

#Pair Plot
sns.pairplot(data, hue="state")
sns.pairplot(data) 
data.corr()

# summary
data.describe()

#Model Building on Levels
ml1=smf.ols('profit~RandD+admin+marketing',data=data).fit()
ml1.summary()

# Identifying the influencing value on the model
import statsmodels.api as sm
sm.graphics.influence_plot(ml1) # 46th raw has highly influencing

# Variable Influencing factor (VIF): For identify collinearity problem
# calculating VIF's values of independent variables
rsq_RandD = smf.ols('RandD~admin+marketing',data=data).fit().rsquared
rsq_RandD
vif_RandD = 1/(1-rsq_RandD)
vif_RandD

rsq_admin = smf.ols('admin~RandD+marketing',data=data).fit().rsquared
vif_admin = 1/(1-rsq_admin)

rsq_marketing =smf.ols('marketing~RandD+admin',data=data).fit().rsquared
vif_marketing = 1/(1-rsq_marketing)
# Storing vif values in a data frame
d1 = {'Variables':['RandD','admin','marketing'],'VIF':[vif_RandD,vif_admin,vif_marketing]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

# Added varible plot 
sm.graphics.plot_partregress_grid(ml1)

# removing the influencing raw based on influence plot
data1=data.drop(data.index[[45]],axis=0)
# model running on the new data1
ml1=smf.ols('profit~RandD+admin+marketing',data=data1).fit()
ml1.summary()


#model on Administration
ml_ad=smf.ols('profit~admin',data=data1).fit()
ml_ad.summary() # the coefficient of administration is not significant

#model on marketing
ml_mar=smf.ols('profit~marketing',data=data1).fit()
ml_mar.summary()

# Model on R & D spending
ml_RD = smf.ols('profit~RandD',data = data1).fit()
ml_RD.summary()

#ml5 Considering all variables except admin
ml5=smf.ols('profit~RandD+marketing',data=data1).fit()
ml5.summary()

#ml_6
data['marketing_squre'] = data.marketing*data.marketing
ml6 =smf.ols('profit~RandD+marketing_squre',data=data).fit()
ml6.summary()
pred6 = ml6.predict(data)
pred6
ERROR = data.profit-pred6
RMSE = np.sqrt(np.mean(ERROR*ERROR))
RMSE

# Identifying the influencing value on the model6
import statsmodels.api as sm
sm.graphics.influence_plot(ml6) # 46th raw has highly influensing

#considering data 1 as afer removing 46 th raw
data['marketing_squre'] = data.marketing*data.marketing
ml6 =smf.ols('profit~RandD+marketing_squre',data=data1).fit()
ml6.summary()
pred6 = ml6.predict(data1)
pred6
ERROR = data.profit-pred6
RMSE = np.sqrt(np.mean(ERROR*ERROR))
RMSE #rmse = 7080.72 selecting model6 for prediction then;

# Confidence values 99%
print(ml6.conf_int(0.01)) # 99% confidence level

# Added varible plot 
sm.graphics.plot_partregress_grid(ml6)


######  Linearity #########
# Observed values VS Fitted values
plt.scatter(data1.profit,pred6,c='red');plt.xlabel("observed_values");plt.ylabel("fitted_values")

##### Residuals VS Fitted Values#### Homoscedasticity #######

plt.scatter(pred6,ml6.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


########    Normality plot for residuals ######
# histogram
plt.hist(ml6.resid_pearson) # Checking the standardized residuals are normally distributed

# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(ml6.resid_pearson, dist="norm", plot=pylab)
##########################################################################

### Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
data_train,data_test  = train_test_split(data1,test_size = 0.2) # 20% size

# preparing the model on train data 

model_train = smf.ols("profit~RandD+(marketing*marketing)",data=data_train).fit()
model_train.summary()

# train_data prediction
train_pred = model_train.predict(data_train)
train_pred

# train residual values 
train_resid  = train_pred - data.profit
train_resid

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))
train_rmse

# prediction on test data set # prediction on unknown data
test_pred = model_train.predict(data_test)
test_pred
# test residual values 
test_resid  = test_pred - data_test.profit

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
test_rmse
##############################################################################






























