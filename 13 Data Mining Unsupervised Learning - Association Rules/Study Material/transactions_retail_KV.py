import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

# Importing transactions_retail1 dataset.
df = pd.read_csv("C:/Users/ketan/OneDrive/Desktop/360digiTMG/13 Data Mining Unsupervised Learning - Association Rules/Hands-on Material/transactions_retail1.csv", header = None)
df.head()
df_sample = df.iloc[1:4000 , : ]
df1 = df_sample.replace(np.nan, '', regex = True) # NAN Values are replaced with empty strings

transactions= []

for i in range(1,3999):
     transactions.append([str(df1.values[i,j]) for j in range(0,6)])

from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)

df_new = pd.DataFrame(te_ary, columns=te.columns_)
df_new

frequent_itemsets = apriori(df_new, min_support = 0.005, use_colnames = True)
frequent_itemsets.shape

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace=True)

import matplotlib.pylab as plt

plt.bar(x = list(range(1,11)),height = frequent_itemsets.support[1:11])
plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11],rotation = "vertical")
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.shape

rules.head(20)
rules.sort_values('lift',ascending = False,inplace=True)

# Changing the min support value

frequent_itemsets_1 = apriori(df_new,min_support = 0.0075,use_colnames = True)
frequent_itemsets_1.shape

# Most Frequent item sets based on support 
frequent_itemsets_1.sort_values('support',ascending = False,inplace=True)

import matplotlib.pylab as plt

plt.bar(x = list(range(1,11)),height = frequent_itemsets_1.support[1:11]);plt.xticks(list(range(1,11)),frequent_itemsets_1.itemsets[1:11]);plt.xlabel('item-sets');plt.ylabel('support')


rules_1 = association_rules(frequent_itemsets_1, metric="lift", min_threshold=1)
rules_1.shape

rules_1.head(20)
rules_1.sort_values('lift',ascending = False,inplace=True)

# Changing the min_threshold value

frequent_itemsets_2 = apriori(df_new,min_support = 0.0075,use_colnames = True)
frequent_itemsets_2.shape

# Most Frequent item sets based on support 
frequent_itemsets_2.sort_values('support',ascending = False,inplace=True)

import matplotlib.pylab as plt

plt.bar(x = list(range(1,11)),height = frequent_itemsets_2.support[1:11],color='red');plt.xticks(list(range(1,11)),frequent_itemsets_1.itemsets[1:11]);plt.xlabel('item-sets');plt.ylabel('support')

rules_2 = association_rules(frequent_itemsets_2, metric="lift", min_threshold=1.5)
rules_2.shape

rules_2.head(20)
rules_2.sort_values('lift',ascending = False,inplace=True)

# We can present all of the rules formed to the client and he/she can apply the number of rules which leads to maximization in profit.

