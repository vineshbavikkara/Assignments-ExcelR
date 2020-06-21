##############################################################################
#  Business Problem is to determine whether there is any significant difference in the diameter of the cutlet between two units.
#  Hypothesis are ;
#  Ho : No signifiacnt Difference
#  Ha : Have significant Difference

import numpy as np
import pandas as pd 

#  Importing the dataset
cutlet = pd.read_csv('F:/alldatasets/Cutlets.csv')
cutlet
cutlet.columns
cutlet.columns=("unitA","unitB")

# The data is continuous and have 2 population with eachother

# Checking Normality Using AD (Anderson Darling)Test
# Ho : Data is normal
# Ha : Data is not normal 
import scipy
import scipy.stats 
from scipy import stats # same as import scipy.stats 
cutlet.columns
scipy.stats.anderson(cutlet.unitA,dist='norm')
# p value of 5% significance level is  0.719.
#p high null fly => Accept Ho : Data is normal.

scipy.stats.anderson(cutlet.unitB,dist='norm')
#p value of 5% significance level is  0.719.
#p high null fly => Accept Ho : Data is normal.

# Checking Normality Using Shapiro test.
scipy.stats.shapiro(cutlet.unitA) # p=0.3199 > 0.05, Data is normal
scipy.stats.shapiro(cutlet.unitB) # p=0.5224 > 0.05, Data is normal

# The data is normal,and we have no external standards available

# Then we have to check variance are equal or not
# Ho : Variance are equal
# Ha : Variance are different

help(scipy.stats.levene) # for help. levene for finding the variance equal or not
scipy.stats.levene(cutlet.unitA,cutlet.unitB)
# p value = 0.417 > 0.05 . Then Accept Ho : Variance are equal.


### 2 Sample T test for Equal Variance ###
#  Ho : Diameter of the cutlet between two units is same
#  Ha : Diameter of the cutlet between two units is same
scipy.stats.ttest_ind(cutlet.unitA,cutlet.unitB, nan_policy = 'omit')
help(scipy.stats.)

# P value = 0.472 > 0.05
# Accepting Ho : Diameter of the cutlet between two units is same 
# No significant difference in the diameter of the cutlet between two units 

########################################################################

########################################################################
# Business problem is to determine whether there is  any difference
#            in the average Turn Around Time (TAT) of reports of the 
#                               laboratories on their preferred list

# Hypothesis are ;
# Ho : Average TAT among the different laboratories is same  
# Ha : Average TAT among the different laboratories is Different  

# importing Pandas library
import pandas as pd
# importing data set
lab =  pd.read_csv("F:/alldatasets/LabTAT.csv")
lab
lab.columns = ('lab1','lab2','lab3','lab4')

# the data is continous
# then we have to check the normality

#importing Scipy library
import scipy
import scipy.stats

# Normality test
# Ho : Data is normal
# Ha : Data is not normal
lab.columns
scipy.stats.shapiro(lab.lab1) # p=0.550 > 0.05, accept Ho:lab1 data is normal
scipy.stats.shapiro(lab.lab2) # p=0.550 > 0.86, accept Ho:lab2 data is normal
scipy.stats.shapiro(lab.lab3) # p=0.550 > 0.42, accept Ho:lab3 data is normal

# All data are normal, The data are continous and,
# comparing 4 population with each other. Then;
# We have to check the variance are equal or not

# checking the variance are equal or not 
# Ho : Variance are equal
# Ha : Variance are unequal
scipy.stats.levene(lab.lab1,lab.lab2,lab.lab3,lab.lab4)
# p = 0.051 > 0.05
# Accept Ho : Variance are equal

#ANOVA test (Analysis of Variance ) for unequal variance
help(scipy.stats.f_oneway)
scipy.stats.f_oneway(lab.lab1,lab.lab2,lab.lab3,lab.lab4)
# pvalue= 2.1156708949992414e-57 < 0.05
# Reject Ho means accepting Ha
# Conclusion: Average TAT among the different laboratories is Different

########################################################################

########################################################################

# Business problem is male-female buyer rations are similar across regions or not
# Hypothesis are ;
# Ho : All proportion are equal
# Ha : Not all proportion are equal

import numpy as np
import pandas as pd

br = pd.read_csv("F:/alldatasets/BuyerRatio.csv")
br.columns = ('sex','east', 'west', 'north', 'south')
br

# Assigning male is 0 , females is 1

## Chi-square test
import scipy
import scipy.stats

scipy.stats.chi2_contingency(br)
help(scipy.stats.chi2_contingency)
# p = 0.7919 > 0.05 
# Accept Ho 
# All proportion are equal
# Conclusion : male-female buyer rations are similar across regions
################################################################

#####################################################################

# Business problem is to check whether the defective % varies by centre or not
# Ho : defective % are same
# Ha : defective % varies by centre

import numpy as np
import pandas as pd

form = pd.read_excel('F:/alldatasets/Costomer+OrderForm. Stack data.xlsx')

# Defective as 0 and Error Free as 1

count=pd.crosstab(form["Defective"],form["Country"])
count
# Chi-square test
import scipy.stats
scipy.stats.chi2_contingency(count)
# p value is 0.277 > 0.05 
# Accept Ho : defective % are same
# Conclusion: the defective % is not varies by centre

#########################################################################

##########################################################################

# Business problem is to check % of males versus females walking in to the store is differ based on day of the week
# Ho : % of males versus females walking in to the store is same
# Ha : % of males versus females walking in to the store is different

import numpy as np
import pandas as pd

fal = pd.read_excel("F:/alldatasets/faltoons_stack.xlsx")
fal
# the data is discrete and comparing 2 population with each other
# Here we use 2 Proportion Test

#importing packages to do 2 proportion test
from statsmodels.stats.proportion import proportions_ztest

#we do the cross table
#we do the cross table and see How many males and females on week days and week end
tab = fal.groupby(['Person', 'Day']).size()
count=pd.crosstab(fal["Person"],fal["Day"])

count = np.array([233, 167]) #How many females and males on weekend
nobs = np.array([520, 280]) #Total number of  Females and males

proportions_ztest(count, nobs,alternative='two-sided')
# p = 6.261142877946052e-05 < 0.05
# Accepting Ha : % of males versus females walking in to the store is different
# conclusion :
#############################################################################












