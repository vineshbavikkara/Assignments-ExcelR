#### ASSOCISTION RULE ####

#PROBLEM : Try different values of support and confidence. Observe the change in number of rules for different support,confidence values

# importing necessery libraries
import numpy as np
import pandas as pd

# implementing Apriori algorithm from mlxtend

# conda install -c conda-forge mlxtend
# Install mlxtend --> open anaconda prompt -> type 'pip install mlxtend'
from mlxtend.frequent_patterns import apriori,association_rules

# open data
groceries = []
# As the file is in transaction data we will be reading data directly 
with open("F:/alldatasets/groceries.csv") as f:
    groceries = f.read()

# splitting the data into separate transactions using separator as "\n"
groceries = groceries.split("\n")

# spliting the each items with ',' in the data by using a new loop
groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))
    

# for  i in gloceries list:
#    all_groceries_list = all_groceries_list+i
# separating each element as new raw based on groceries_list
all_groceries_list = []
all_groceries_list = [i for item in groceries_list for i in item]

from collections import Counter # For finding frequencies
item_frequencies = Counter(all_groceries_list)

# sorting the  data based on item column # for finding highest frequency
#item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])
item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

# barplot of top 10 
import matplotlib.pyplot as plt

plt.bar(height = frequencies[:11], x = list(range(0,11)),color='rgbkymc');plt.xticks(list(range(0,11),),items[0:11]);plt.xlabel("items");plt.ylabel("Count")


# Creating Data Frame for the transactions data 
# Purpose of converting all list into Series object Coz to treat each list element as entire element not to separate 
groceries_series  = pd.DataFrame(pd.Series(groceries_list))
groceries_series = groceries_series.iloc[:9835,:] # removing the last empty transaction

groceries_series.columns = ["transactions"]

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = groceries_series['transactions'].str.join(sep='*').str.get_dummies(sep='*')

frequent_itemsets = apriori(X, min_support=0.005, max_len=3,use_colnames = True)
frequent_itemsets.shape

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support',ascending = False,inplace=True)
plt.bar(x = list(range(1,11)),height = frequent_itemsets.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11]);plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending = False,inplace=True)
rules= pd.DataFrame(rules)

# Consedering different support  and confidence values then comparing the changes in the number of rules;
frequent_itemsets1 = apriori(X, min_support=0.004, max_len=3,use_colnames = True)
frequent_itemsets1.shape
rules1 = association_rules(frequent_itemsets1, metric="lift", min_threshold=1)
rules1.sort_values('lift',ascending = False,inplace=True)
rules1= pd.DataFrame(rules1)

frequent_itemsets2 = apriori(X, min_support=0.003, max_len=3,use_colnames = True)
frequent_itemsets2.shape
rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=1)
rules2.sort_values('lift',ascending = False,inplace=True)
rules2= pd.DataFrame(rules2)

###########################################################################################

#PROBLEM : Change the minimum length in apriori algorithm

# importing necessery libraries
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules

# open data
data = pd.read_csv('F:/alldatasets/book.csv')

frequent_itemsets = apriori(data, min_support=0.005, max_len=3,use_colnames = True)
frequent_itemsets.shape

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support',ascending = False,inplace=True)

import matplotlib.pylab as plt
plt.bar(x = list(range(1,26)),height = frequent_itemsets.support[1:26],color='rgmyk');plt.xticks(list(range(1,26)),frequent_itemsets.itemsets[1:26]);plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending = False,inplace=True)
rules= pd.DataFrame(rules)


# Consedering different maximum length(max_len is refleccted in Anticident and Consequents list on rules) and comparing the changes in the number of rules;
frequent_itemsets1 = apriori(data, min_support=0.004,max_len=4,use_colnames = True)
frequent_itemsets1.shape
rules1 = association_rules(frequent_itemsets1, metric="lift", min_threshold=1)
rules1.sort_values('lift',ascending = False,inplace=True)
rules1= pd.DataFrame(rules1)

frequent_itemsets2 = apriori(data, min_support=0.001, max_len=2,use_colnames = True)
frequent_itemsets2.shape
rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=1)
rules2.sort_values('lift',ascending = False,inplace=True)
rules2= pd.DataFrame(rules2)

###########################################################################################
# Problem : Visulize the obtained rules using different plots 

# importing necessery libraries
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules

# open data
data1 = pd.read_csv('F:/alldatasets/my_movies.csv')
del data1['V1'], data1['V2'], data1['V3'], data1['V4'], data1['V5']

frequent_itemsets = apriori(data1, min_support=0.005, max_len=3,use_colnames = True)
frequent_itemsets.shape

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support',ascending = False,inplace=True)

import matplotlib.pylab as plt
plt.bar(x = list(range(1,26)),height = frequent_itemsets.support[1:26],color='rgmyk');plt.xticks(list(range(1,26)),frequent_itemsets.itemsets[1:26]);plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head(20)
rules.sort_values('lift',ascending = False,inplace=True)
rules= pd.DataFrame(rules)

# Consedering different maximum length and comparing the changes in the number of rules;
frequent_itemsets1 = apriori(data1, min_support=0.004,max_len=4,use_colnames = True)
frequent_itemsets1.shape
rules1 = association_rules(frequent_itemsets1, metric="lift", min_threshold=1)
rules1.sort_values('lift',ascending = False,inplace=True)
rules1= pd.DataFrame(rules1)

frequent_itemsets2 = apriori(data1, min_support=0.001, max_len=2,use_colnames = True)
frequent_itemsets2.shape
rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=1)
rules2.sort_values('lift',ascending = False,inplace=True)
rules2= pd.DataFrame(rules2)
