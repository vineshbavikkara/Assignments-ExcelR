# Importing nesessari libraries
import numpy as np
import pandas as pd 
import matplotlib.pylab as plt
import statsmodels.formula.api as smf
import seaborn as sns

# Importing the dataset
data = pd.read_csv("F:/Computer.csv")
data.describe()

# creating Dummy variable on CD data 
dummies_cd = pd.get_dummies(data['cd'], prefix='cd')
dummies_cd
data_with_dummycd = data.join(dummies_cd)
data_with_dummycd 

# creating Dummy variable on multi data 
dummies_multi = pd.get_dummies(data['multi'],prefix='multi')
dummies_multi
data_with_dummycdmulti =data_with_dummycd.join(dummies_multi)

# creating Dummy variable on premium data 
dummies_premium = pd.get_dummies(data['premium'],prefix = 'premium')
dummies_premium
data_with_dummies = data_with_dummycdmulti.join(dummies_premium)

# Final data with all dummies and avoiding categorical data from the data frame
dta = data_with_dummies[:]
del dta['cd'], dta['multi'], dta['premium']
data=dta[:]
del dta
data.columns

# EDA
#Boxplot
sns.boxplot(data['price'])
sns.boxplot(data['speed'])
sns.boxplot(data['hd'])
sns.boxplot(data['ram'])
sns.boxplot(data['screen'])
sns.boxplot(data['ads'])
sns.boxplot(data['trend'])

# Scatter plot between variables
plt.scatter(data.multi_yes,data.price)


#Pair Plot and Correlation
sns.pairplot(data1)
correlation = pd.DataFrame(data.corr())



#Model Building on Levels
data.columns
ml1=smf.ols('price ~ speed+hd+ram+screen+ads+trend+cd_no+cd_yes+multi_no+multi_yes+premium_no+premium_yes',data=data).fit()
ml1.summary()

# Identifying the influencing value on the model
import statsmodels.api as sm
sm.graphics.influence_plot(ml1) #  raw has highly influencing

# Added varible plot 
sm.graphics.plot_partregress_grid(ml1)

# ml2 model on all the variables excepting dummy variables
ml2=smf.ols('price ~ speed+hd+ram+screen+ads+trend',data=data).fit()
ml2.summary()

#ml3 Considering only dummy variables
ml3 = smf.ols('(price)~ cd_no+cd_yes+multi_no+multi_yes+premium_no+premium_yes',data=data).fit()
ml3.summary()

# ml4
data['ram_square']= data.ram*data.ram
ml4= smf.ols('np.log(price) ~ np.log(speed)+np.log(hd)+np.log(ram)+np.log(screen)+np.log(ads)+trend+cd_no+cd_yes+multi_no+multi_yes+premium_no+premium_yes',data=data).fit()
ml4.summary()
ml4.params
# Finding influencing raw and remove it from
sm.graphics.influence_plot(ml4) #  raw has highly influencing
data1 = data.drop(data.index[[4327]],axis=0) # removing influencing rows
ml5= smf.ols('np.log(price) ~ np.log(speed)+np.log(hd)+np.log(ram)+np.log(screen)+np.log(ads)+trend+cd_no+cd_yes+multi_no+multi_yes+premium_no+premium_yes',data=data1).fit()
ml5.summary()


# selecting model 4

pred_log = ml5.predict(data)
pred_log
pred4 = np.exp(pred_log)
pred4
ERROR = data.price-pred4
RMSE = np.sqrt(np.mean(ERROR*ERROR))
RMSE


# Confidence values 99%
ml4.conf_int(0.01) # 99% confidence level

######  Linearity #########
# Observed values VS Fitted values
plt.scatter(data.price,pred4,c='red');plt.xlabel("observed_values");plt.ylabel("fitted_values")

##### Residuals VS Fitted Values#### Homoscedasticity #######

plt.scatter(pred4,ml3.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


########    Normality plot for residuals ######
# histogram
plt.hist(ml4.resid_pearson) # Checking the standardized residuals are normally distributed

# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(ml4.resid_pearson, dist="norm", plot=pylab)
