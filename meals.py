import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import json

def dataframe_analize(df,df_name):
	print('-'*(len(df_name)+40+2))
	print('-'*20,df_name.upper(),'-'*20)
	print(df.head(5))
	print('.'*20)
	print(df.tail(5))
	print('DATA TYPES FOR COLUMNS','-'*20)
	print(df.dtypes)
	print('COLUMS DESCRIPTION','-'*20)
	print(df.describe())
	print('NULL VALUES?','-'*20)
	print(df.isnull().sum())

		
def create_encoding(df,column):
	values_list = df[column].tolist()
	values_set = set (values_list)
	encoding_dict = {}
	for i,value in enumerate(values_set):
		encoding_dict[value] = i+1  # no zero codes (parameters will always be nullified)
	return(encoding_dict)
	
#
# Meals csv -> encoded meal info
#
#cuisine_encode  = {'Continental':1,'Indian':2,'Italian':3,'Thai':4}
#category_encode = {'Beverages':1,'Biryani':2,'Desert':3,'Extras':4,'Fish':5,'Other Snacks':6,'Pasta':7,'Pizza':8,'Rice Bowl':9,'Salad':10,'Sandwich':11,'Seafood':12,'Soup':13,'Starters':14}
df_meals = pd.read_csv('Dataset/meal_info.csv')
dataframe_analize(df_meals,'meal-info')
#
cuisine_encode  = create_encoding(df_meals,'cuisine')
category_encode = create_encoding(df_meals,'category')
df_meals['encode_cuisine']  = df_meals.apply(lambda x: cuisine_encode[x['cuisine']], axis=1)
df_meals['encode_category'] = df_meals.apply(lambda x: category_encode[x['category']], axis=1)
print('cuisine:',len(cuisine_encode))
print('category:',len(category_encode))
#
# Scatterplot distribuzione meals tra cuisine e category
#

from collections import Counter
z = list(zip(df_meals['cuisine'].to_list(),df_meals['category'].to_list()))
c = Counter(z) 
print('Counter for Meals: category x cuisine',c)
s = [10*c[(x,y)] for x,y in z]
fig = plt.figure(figsize=(16, 10), dpi=80)
grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)
ax_main = fig.add_subplot(grid[:-1, :-1])
ax_right = fig.add_subplot(grid[:-1, -1], yticklabels=[])
ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[])
ax_main.scatter('category', 'cuisine', s=s, data=df_meals, edgecolors='face', linewidths=.5)
#
ax_bottom.hist(df_meals.category, 40, histtype='stepfilled', orientation='vertical', color='green')
ax_bottom.invert_yaxis()
ax_right.hist(df_meals.cuisine, 40, histtype='stepfilled', orientation='horizontal', color='green')
#
ax_main.set(title='Scatterplot MEALS category vs cuisine', xlabel='category', ylabel='cuisine')
ax_main.title.set_fontsize(20)
plt.show()
