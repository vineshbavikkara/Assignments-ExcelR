# Business problem is to predict weight gained using calories consumed data
# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# importing the dataset
calor = pd.read_csv('F:/alldatasets/calories_consumed.csv')
calor.columns = ('weight','calories')
calor
calor.describe()

# To find the distribution of the data usind histogram
plt.hist(calor.weight)
plt.hist(calor.calories)

#box plot
plt.boxplot(calor.weight,vert=False)
plt.boxplot(calor.calories,vert=False)

#scatter plot
plt.scatter(x=calor['calories'],y=calor['weight']);plt.xlabel=("calories_consumed");plt.ylabel=("weight_gained")

# correlation value between calories_consumed and weight_gained
np.corrcoef(calor.calories,calor.weight)
calor.calories.corr(calor.weight)

# importing statsmodels.formula.api for preparing regression model
import statsmodels.formula.api as smf
model = smf.ols('weight~calories',data = calor).fit()
model.params # for getting parameters of the model
model.summary() # for getting model summary

# Confidence interval
model.conf_int(0.01)

# For getting Predicted weight gaine Values
pred = model.predict(calor.iloc[:,1])
pred
pred = model.predict(calor)
pred
# Scatter plot with regression line
plt.scatter(x=calor['calories'],y=calor['weight'],color='green');plt.plot(calor['calories'],pred,color='blue');plt.xlabel('Calories consumed');plt.ylabel('weight gainde')
# For Error and RMSE
error = pd.DataFrame(calor.weight-pred)
RMSE= np.sqrt(np.mean((pred-calor.weight)**2))
RMSE
# Transforming the data for getting more accuracy
# log transformation on calories
model1 = smf.ols("weight ~ np.log(calories)",data= calor).fit()
model1.params
model1.summary()
pred1 = model1.predict(calor)
RMSE1= np.sqrt(np.mean((pred1-calor.weight)**2))
plt.scatter(x=calor['calories'],y=calor['weight'],color='green');plt.plot(calor['calories'],pred1,color='blue');plt.xlabel('Calories consumed');plt.ylabel('weight gained')

# ExponentialTransformation (log value on weight )
model2 = smf.ols("np.log(weight)~calories",data= calor).fit()
model2.params
model2.summary()
pred_log = model2.predict(calor)
pred_log
pred2 = pd.DataFrame(np.exp(pred_log))
pred2
RMSE2= np.sqrt(np.mean((pred2-calor.weight)**2))
RMSE
plt.scatter(x=calor['calories'],y=calor['weight'],color='green');plt.plot(calor['calories'],pred2,color='blue');plt.xlabel('Calories consumed');plt.ylabel('weight gained')

# Quadratic model
calor["calories_sqr"] = calor.calories*calor.calories
model3 = smf.ols('weight~calories+calories_sqr',data=calor).fit()
model3.params
model3.summary()
pred3 = model3.predict(calor)
pred3
RMSE3= np.sqrt(np.mean((pred3-calor.weight)**2))
plt.scatter(x=calor['calories'],y=calor['weight'],color='green');plt.plot(calor['calories'],pred3,color='blue');plt.xlabel('Calories consumed');plt.ylabel('weight gained')

# model4 --> Transformation is squaring the calories data considering only the squared value
calor["calories_sqr"] = calor.calories*calor.calories
model4 = smf.ols('weight~calories_sqr',data=calor).fit()
model4.params
model4.summary()
pred4 = model4.predict(calor)
pred4
RMSE4= np.sqrt(np.mean((pred4-calor.weight)**2))
RMSE4
plt.scatter(x=calor['calories'],y=calor['weight'],color='green');plt.plot(calor['calories'],pred4,color='blue');plt.xlabel('Calories consumed');plt.ylabel('weight gained')

# Predicted vs actual values scatter plot
plt.scatter(x=pred4,y=calor.weight);plt.xlabel("Predicted");plt.ylabel("Actual")

#Correlation between predicted and actual 
np.corrcoef(calor.weight,pred4)

# confidence inteval of Model4
model4.conf_int(0.01)

# selecting Model4, because of high R-squared value and leat RMSE
# getting residuals of the entire data set
student_resid = model4.resid_pearson 
student_resid
# histogram for residual values 
plt.hist(model4.resid_pearson)
#scatter plot of standardised residuals
plt.plot(model4.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

################################################################################













