### FORECASTING #### MODEL BASED ###
# Business problem : Forecast the CocaCola prices

#importing dataset
import pandas as pd
data = pd.read_excel("F:/Cocacola.xlsx")
data.columns

# Time plot
data.plot()

# creating dummy variable 
dummy= pd.get_dummies(data.Quarter)
d= data.join(dummy)

# creating time index column
import numpy as np
d['t'] = np.arange(1,43)

# creating t- square column
d['t_squre'] = d.t*d.t

# log value of Sales
d['log_sales']= np.log(d.Sales)

# Time plot
d.Sales.plot()

# Spliting Train and Test
Train = d.head(39)
Test = d.tail(4)


####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear # 538.44

##################### Exponential ##############################

Exp = smf.ols('log_sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp # 441.36

#################### Quadratic ###############################

Quad = smf.ols('Sales~t+t_squre',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squre"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad #504.79

################### Additive seasonality ########################

add_sea = smf.ols('Sales~Q1_86+Q1_87+Q1_88+Q1_89+Q1_90+Q1_91+Q1_92+Q1_93+Q1_94+Q1_95+Q1_96+Q2_86+Q2_87+Q2_88+Q2_89+Q2_90+Q2_91+Q2_92+Q2_93+Q2_94+Q2_95+Q2_96+Q3_86+Q3_87+Q3_88+Q3_89+Q3_90+Q3_91+Q3_92+Q3_93+Q3_94+Q3_95+Q4_86+Q4_87+Q4_88+Q4_89+Q4_90+Q4_91+Q4_92+Q4_93+Q4_94+Q4_95',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea # 1605.86

################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Sales~t+t_squre+Q1_86+Q1_87+Q1_88+Q1_89+Q1_90+Q1_91+Q1_92+Q1_93+Q1_94+Q1_95+Q1_96+Q2_86+Q2_87+Q2_88+Q2_89+Q2_90+Q2_91+Q2_92+Q2_93+Q2_94+Q2_95+Q2_96+Q3_86+Q3_87+Q3_88+Q3_89+Q3_90+Q3_91+Q3_92+Q3_93+Q3_94+Q3_95+Q4_86+Q4_87+Q4_88+Q4_89+Q4_90+Q4_91+Q4_92+Q4_93+Q4_94+Q4_95',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad # 403.02

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_sales~Q1_86+Q1_87+Q1_88+Q1_89+Q1_90+Q1_91+Q1_92+Q1_93+Q1_94+Q1_95+Q1_96+Q2_86+Q2_87+Q2_88+Q2_89+Q2_90+Q2_91+Q2_92+Q2_93+Q2_94+Q2_95+Q2_96+Q3_86+Q3_87+Q3_88+Q3_89+Q3_90+Q3_91+Q3_92+Q3_93+Q3_94+Q3_95+Q4_86+Q4_87+Q4_88+Q4_89+Q4_90+Q4_91+Q4_92+Q4_93+Q4_94+Q4_95',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea #2067.93

##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_sales~t+Q1_86+Q1_87+Q1_88+Q1_89+Q1_90+Q1_91+Q1_92+Q1_93+Q1_94+Q1_95+Q1_96+Q2_86+Q2_87+Q2_88+Q2_89+Q2_90+Q2_91+Q2_92+Q2_93+Q2_94+Q2_95+Q2_96+Q3_86+Q3_87+Q3_88+Q3_89+Q3_90+Q3_91+Q3_92+Q3_93+Q3_94+Q3_95+Q4_86+Q4_87+Q4_88+Q4_89+Q4_90+Q4_91+Q4_92+Q4_93+Q4_94+Q4_95',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea # 1937.08

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse
# so rmse_add_sea_quad has the least value among the models prepared so far 
# selecting this model for Forecasting

##############################################################################################################
##############################################################################################################
# Business problem :Forecast Airlines Passengers data set

#importing dataset
import pandas as pd
data = pd.read_excel("F:/Airlines.xlsx")
data.columns

# Converting the normal index of data to time stamp for getting year on x axis while running time plot
data.index=pd.to_datetime(data.Month,format="%b-%y")
# time series plot - included Year in X-axis 
data.Passengers.plot() 

# Converting the index to normal
import numpy as np
data.index = np.arange(0,96)

# Creating a Date column to store the actual Date format for the given Month column
from datetime import datetime,time
data["Date"] = pd.to_datetime(data.Month,format="%b-%y")

# Extracting Day, weekday name, month name, year from the Date column using 
# Date functions from pandas 

data["month"] = data.Date.dt.strftime("%b") # month extraction
#data["Day"] = data.Date.dt.strftime("%d") # Day extraction
#data["wkday"] = data.Date.dt.strftime("%A") # weekday extraction
data["year"] = data.Date.dt.strftime("%Y") # year extraction

# Creating a new column in data joining the month with year
data['Month'] = data.Date.dt.strftime("%Y_%b")
del data['Date']

## EDA on Time series data ##
# Heat map visualization 
import seaborn as sns
heatmap_y_month = pd.pivot_table(data=data,values="Passengers",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")

# Boxplot for ever
sns.boxplot(x="month",y="Passengers",data=data)
sns.boxplot(x="year",y="Passengers",data=data)
sns.factorplot("month","Passengers",data=data,kind="box")

# Line plot for Passengers based on year  and for each month
sns.lineplot(x="year",y="Passengers",hue="month",data=data)


# moving average for the time series to understand better about the trend character in data
import matplotlib.pylab as plt
data.Passengers.plot(label="org") # time series plot
for i in range(2,24,6):
    data["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)
    
# Time series decomposition plot 
from statsmodels.tsa.seasonal import seasonal_decompose
decompose_ts_add = seasonal_decompose(data.Passengers,model="additive")
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(data.Passengers,model="multiplicative")
decompose_ts_mul.plot()

# ACF plots and PACF plots on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
tsa_plots.plot_acf(data.Passengers,lags=10)
tsa_plots.plot_pacf(data.Passengers)

# creating dummy variable 
dummy= pd.get_dummies(data.Month)
d= data.join(dummy)

# creating time index column
import numpy as np
d['t'] = np.arange(1,97)

# creating t- square column
d['t_squre'] = d.t*d.t

# log value of Sales
d['log_Passengers']= np.log(d.Passengers)

# Time plot
d.Passengers.plot() # Assumptions based on time plot ; it have multiplicative seasonality with linear upward trend

# Spliting Train and Test
Train = d.head(84)
Test = d.tail(12)


####################### L I N E A R ##########################
import statsmodels.formula.api as smf

linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear # 53.199

##################### Exponential ##############################
Exp = smf.ols('log_Passengers~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp # 46.05

#################### Quadratic ###############################
Quad = smf.ols('Passengers~t+t_squre',data = Train).fit()
pred_Quad = Quad.predict(Test)
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad)))**2)
rmse_Quad # 15.97
################### Additive seasonality ########################
X = Train.iloc[:,2:98] 
Y = Train['Passengers']
 
# with sklearn
from sklearn.linear_model import LinearRegression
regr =LinearRegression()
regr.fit(X, Y)
regr.coef_
pred_passengers=regr.predict(X) # for getting predicted Passengers on train data

#getting predicted passengers data on  test data set
Z= Test.iloc[:,2:98]
pred_passengers_test = regr.predict(Z) # predicting passengers using test dataset 
pred_passengers_test

pred_add_sea = pred_passengers_test # just for renaming purpose

rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea # 138.69

################## Additive Seasonality Quadratic ############################
X = Train.iloc[:,2:100]
Y = Train['Passengers']
 
# with sklearn
from sklearn.linear_model import LinearRegression
regr =LinearRegression()
regr.fit(X, Y)
regr.coef_
pred_passengers=regr.predict(X) # for getting predicted Passengers on train data

#getting predicted passengers data on  test data set
Z= Test.iloc[:,2:100]
pred_passengers_test = regr.predict(Z) # predicting passengers using test dataset 
pred_passengers_test

pred_add_sea_quad = pred_passengers_test # just for renaming purpose

rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad ))**2))
rmse_add_sea_quad #48.05

################## Multiplicative Seasonality ##################
X = Train.iloc[:,2:98]
Y = Train['log_Passengers']
 
# with sklearn
from sklearn.linear_model import LinearRegression
regr =LinearRegression()
regr.fit(X, Y)
regr.coef_
pred_passengers=regr.predict(X) # for getting predicted Passengers on train data

#getting predicted passengers data on  test data set
Z= Test.iloc[:,2:98]
pred_passengers_test = regr.predict(Z) # predicting passengers using test dataset 
pred_passengers_test

pred_multi_sea = np.exp(pred_passengers_test) # just for renaming purpose

rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_multi_sea ))**2))
rmse_Mult_sea # 146.6

#################Multiplicative Additive Seasonality ###########
X = Train.iloc[:,2:99]
Y = Train['log_Passengers']
 
# with sklearn
from sklearn.linear_model import LinearRegression
regr =LinearRegression()
regr.fit(X, Y)
regr.coef_
pred_passengers=regr.predict(X) # for getting predicted Passengers on train data

#getting predicted passengers data on  test data set
Z= Test.iloc[:,2:99]
pred_passengers_test = regr.predict(Z) # predicting passengers using test dataset 
pred_passengers_test

pred_multi_add_sea = np.exp(pred_passengers_test) # just for renaming purpose

rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_multi_add_sea))**2))
rmse_Mult_add_sea # 46.05

################## Multiplicative Quadratic trend ################
X = Train.iloc[:,2:100]
Y = Train['log_Passengers']
 # with sklearn
from sklearn.linear_model import LinearRegression
regr =LinearRegression()
regr.fit(X, Y)
regr.coef_
pred_passengers=regr.predict(X) # for getting predicted Passengers on train data

#getting predicted passengers data on  test data set
Z= Test.iloc[:,2:100]
pred_passengers_test = regr.predict(Z) # predicting passengers using test dataset 
pred_passengers_test

pred_multi_sea_quad = np.exp(pred_passengers_test) # just for renaming purpose

rmse_Mult_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_multi_sea_quad))**2))
rmse_Mult_sea_quad # 49.33

################## Testing #######################################
data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea","rmse_Mult_sea_quad"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea,rmse_Mult_sea_quad])}
table_rmse=pd.DataFrame(data)
table_rmse
# so rmse_Exp  has the least value among the models prepared so far 
# selecting Qudratic model for Forecasting.

###############################################################################################################
###############################################################################################################
### FORECASTING ###### DATA DRIVEN APPROACHES ####
# Business problem : Forecast Airlines Passengers data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing 
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
from datetime import datetime,time
#from sm.tsa.statespace import sa
data = pd.read_excel("F:\\Airlines.xlsx")
data.rename(columns={"Passengers":"passengers"},inplace=True)   
# Converting the normal index of passengers data to time stamp 
data.index = pd.to_datetime(data.Month,format="%b-%y")
data.passengers.plot()# time series plot 
# Creating a Date column to store the actual Date format for the given Month column
data["Date"] = pd.to_datetime(data.Month,format="%b-%y")

# Extracting Day, weekday name, month name, year from the Date column using 
# Date functions from pandas 

data["month"] = data.Date.dt.strftime("%b") # month extraction
#data["Day"] = data.Date.dt.strftime("%d") # Day extraction
#data["wkday"] = data.Date.dt.strftime("%A") # weekday extraction
data["year"] = data.Date.dt.strftime("%Y") # year extraction

# Some EDA on Time series data 
# Heat map visualization 
heatmap_y_month = pd.pivot_table(data=data,values="passengers",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")

# Boxplot
sns.boxplot(x="month",y="passengers",data=data)
sns.boxplot(x="year",y="passengers",data=data)
# factor plot
sns.factorplot("month","passengers",data=data,kind="box")

# Line plot for passengers based on year  and for each month
sns.lineplot(x="year",y="passengers",hue="month",data=data)


# moving average for the time series to understand better about the trend character in data
data.passengers.plot(label="org")
for i in range(2,20,6):
    data["passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=4)

# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(data.passengers,model="additive")
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(data.passengers,model="multiplicative")
decompose_ts_mul.plot()

# ACF plots and PACF plots on Original data sets 
tsa_plots.plot_acf(data.passengers,lags=9)
tsa_plots.plot_pacf(data.passengers)

# data.index.freq = "MS" 
# splitting the data into Train and Test data and considering the last 12 months data as train
# and left over data as test data 

Train = data.head(84)
Test = data.tail(12)
# to change the index value in pandas data frame 
#Test.set_index(np.arange(1,13),inplace=True)

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)

# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["passengers"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.passengers) # 14.235

# Holt method 
hw_model = Holt(Train["passengers"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.passengers) # 11.841

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["passengers"],seasonal="add",trend="add",seasonal_periods=12,damped=True).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.passengers) # 1.618

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["passengers"],seasonal="mul",trend="add",seasonal_periods=12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.passengers) # 2.819

# Lets us use auto_arima from p
from statsmodels.tsa.arima_model import ARIMA
# fit model
model = ARIMA(Train['passengers'], order=(1,0,0))
model_fit = model.fit(disp=0)
model_fit.summary()
# AIC ==> 847.60
# BIC ==>854.89

# plot residual errors
import matplotlib.pylab as plt
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
residuals.plot(kind='kde')
residuals.describe()

pred_test.index = Test.index
MAPE(pred_test,Test.passengers)
#############################################################################################