#pip install mlxtend

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

books = pd.read_csv('C:/Users/ketan/OneDrive/Desktop/360digiTMG/13 Data Mining Unsupervised Learning - Association Rules/Hands-on Material/book.csv')
frequent_books = apriori(books, min_support = 0.0075, max_len = 4, use_colnames = True)

frequent_books.sort_values('support', ascending = False, inplace = True)

import matplotlib.pylab as plt
 
plt.bar(x = list(range(0, 11)), height = frequent_books.support[0:11], color ='blue')
plt.xticks(list(range(0, 11)), frequent_books.itemsets[0:11], rotation=90)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_books, metric = "lift", min_threshold = 1)
rules.head()
rules = rules.sort_values('lift', ascending = False).head(10)

