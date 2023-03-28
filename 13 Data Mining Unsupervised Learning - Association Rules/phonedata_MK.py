import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

phone = []
with open("C:/Users/madhu/OneDrive/Desktop/360digiTMG/Data Science/13 Data Mining Unsupervised Learning - Association Rules/Hands-on Material/myphonedata.csv") as f:
    phone = f.read()

# splitting the data into separate transactions using separator as "\n"
phone = phone.split("\n")

phone_list = []

for i in phone:
    phone_list.append(i.split(","))

all_phone_list = [i for item in phone_list for i in item]

from collections import Counter # ,OrderedDict

item_frequencies = Counter(all_phone_list)

# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

# barplot of top 10 
import matplotlib.pyplot as plt

plt.bar(height = frequencies[0:11], x = list(range(0, 11)), color = 'red')
plt.xticks(list(range(0, 11), ), items[0:11])
plt.xlabel("items")
plt.ylabel("Count")
plt.show()


# Creating Data Frame for the transactions data
phone_series = pd.DataFrame(pd.Series(phone_list))
phone_series = phone_series.iloc[:9835, :] # removing the last empty transaction

phone_series.columns = ["transactions"]

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = phone_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')

frequent_itemsets = apriori(X, min_support = 0.0075, max_len = 4, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color ='blue')
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=90)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)

