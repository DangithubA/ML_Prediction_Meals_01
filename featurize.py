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
# Matrice di correlazione
# 
def correllogram_plot(df_sales,descr):
	plt.figure(figsize=(12, 10), dpi=80)
	corr_map = df_sales.corr()
	mask = np.zeros_like(corr_map)
	mask[np.triu_indices_from(mask)] = True
	sns.heatmap(corr_map, mask=mask, xticklabels=corr_map.columns, yticklabels=corr_map.corr().columns, center=0,annot=True)
	plt.title('Correlogram of sales '+descr, fontsize=22)
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	locs, labels = plt.xticks()
	plt.setp(labels, rotation=45)
	plt.show()

	
pd.set_option('display.max_columns', None)

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
ax_bottom.hist(df_meals.category, 40, histtype='stepfilled', orientation='vertical') #, color='green')
ax_bottom.invert_yaxis()
ax_right.hist(df_meals.cuisine, 40, histtype='stepfilled', orientation='horizontal') #, color='green')
#
ax_main.set(title='Scatterplot MEALS category vs cuisine', xlabel='category', ylabel='cuisine')
ax_main.title.set_fontsize(20)
plt.show()

# Centers csv -> encoded center info
#
#city_encode    = {456:1,461:2,473:3,478:4,485:5,515:6,517:7,522:8,526:9,541:10,553:11,556:12,561:13,562:14,576:15,577:16,579:17,590:18,593:19,596:20,599:21,602:22,604:23,609:24,614:25,615:26,620:27,628:28,632:29,638:30,647:31,648:32,649:33,651:34,654:35,658:36,659:37,675:38,676:39,679:40,680:41,683:42,685:43,693:44,695:45,698:46,699:47,700:48,702:49,703:50,713:51}
#region_encode  = {23:1,34:2,35:3,56:4,71:5,77:6,85:7,93:8}
#type_encode    = {'TYPE_A':1,'TYPE_B':2,'TYPE_C':3}
df_centers = pd.read_csv('Dataset/fulfilment_center_info.csv')
dataframe_analize(df_centers,'fullfilment-centers')
#
city_encode   = create_encoding(df_centers,'city_code')
region_encode = create_encoding(df_centers,'region_code')
type_encode   = create_encoding(df_centers,'center_type')
df_centers['encode_city']   = df_centers.apply(lambda x: city_encode[x['city_code']], axis=1)
df_centers['encode_region'] = df_centers.apply(lambda x: region_encode[x['region_code']], axis=1)
df_centers['encode_type']   = df_centers.apply(lambda x: type_encode[x['center_type']], axis=1)
print('city:',len(city_encode))
print('region:',len(region_encode))
print('type:',len(type_encode))
#
# Scatterplot distribuzione centers tra region e city
#
z = list(zip(df_centers['encode_region'].to_list(),df_centers['encode_city'].to_list()))
c = Counter(z) 
print('Counter for centers: region x city',c)
s = [10*c[(x,y)] for x,y in z]
fig = plt.figure(figsize=(16, 10), dpi=80)
grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)
ax_main = fig.add_subplot(grid[:-1, :-1])
ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[])
ax_main.scatter('encode_region', 'encode_city', s=s, data=df_centers, edgecolors='face', linewidths=.5)
#
ax_bottom.hist(df_centers.encode_region, 40, histtype='stepfilled', orientation='vertical', color='green')
ax_bottom.invert_yaxis()
#
ax_main.set(title='Scatterplot CENTERS region vs city', xlabel='region', ylabel='city')
ax_main.title.set_fontsize(20)
plt.show()
#
# Sales csv -> dataframe (per sviluppo programma prendo solo i primi 1000)
#
def week_in_year(week):
	w = week % 52
	if w == 0:
		w = 52
	return(int(w))
#center_id_encode = {10:1,11:2,13:3,14:4,17:5,20:6,23:7,24:8,26:9,27:10,29:11,30:12,32:13,34:14,36:15,39:16,41:17,42:18,43:19,50:20,51:21,52:22,53:23,55:24,57:25,58:26,59:27,61:28,64:29,65:30,66:31,67:32,68:33,72:34,73:35,74:36,75:37,76:38,77:39,80:40,81:41,83:42,86:43,88:44,89:45,91:46,92:47,93:48,94:49,97:50,99:51,101:52,102:53,104:54,106:55,108:56,109:57,110:58,113:59,124:60,126:61,129:62,132:63,137:64,139:65,143:66,145:67,146:68,149:69,152:70,153:71,157:72,161:72,162:74,174:75,177:76,186:77}
#meal_id_encode   = {1062:1,1109:2,1198:3,1207:4,1216:5,1230:6,1247:7,1248:8,1311:9,1438:10,1445:11,1525:12,1543:13,1558:14,1571:15,1727:16,1754:17,1770:18,1778:19,1803:20,1847:21,1878:22,1885:23,1902:24,1962:25,1971:26,1993:27,2104:28,2126:29,2139:30,2290:31,2304:32,2306:33,2322:34,2444:35,2490:36,2492:37,2494:38,2539:39,2569:40,2577:41,2581:42,2631:43,2640:44,2664:45,2704:46,2707:47,2760:48,2826:49,2867:50,2956:51}

df_sales = pd.read_csv('Dataset/train.csv') # Legge il csv completo
#df_sales = pd.read_csv('Dataset/train.csv',nrows=1000) # PER PRIMO TEST USATE SOLO 1000 RIGHE
dataframe_analize(df_sales,'weekly sales')
#
center_id_encode = create_encoding(df_sales,'center_id')
meal_id_encode   = create_encoding(df_sales,'meal_id')
df_sales['center_type'] = df_sales.apply(lambda x: df_centers[df_centers['center_id']==x['center_id']]['encode_type'].item(),axis=1)
df_sales['center_region'] = df_sales.apply(lambda x: df_centers[df_centers['center_id']==x['center_id']]['encode_region'].item(),axis=1)
df_sales['center_city'] = df_sales.apply(lambda x: df_centers[df_centers['center_id']==x['center_id']]['encode_city'].item(),axis=1)
df_sales['center_op_area'] = df_sales.apply(lambda x: df_centers[df_centers['center_id']==x['center_id']]['op_area'].item(),axis=1)
df_sales['meal_cuisine'] = df_sales.apply(lambda x: df_meals[df_meals['meal_id']==x['meal_id']]['encode_cuisine'].item(),axis=1)
df_sales['meal_category'] = df_sales.apply(lambda x: df_meals[df_meals['meal_id']==x['meal_id']]['encode_category'].item(),axis=1)
df_sales['center_code']  = df_sales.apply(lambda x: center_id_encode[x['center_id']], axis=1)
df_sales['meal_code']  = df_sales.apply(lambda x: meal_id_encode[x['meal_id']], axis=1)
df_sales['week_in_year']  = df_sales.apply(lambda x: week_in_year(x['week']), axis=1)
df_sales.drop(columns='center_id',inplace = True)
df_sales.drop(columns='meal_id',inplace = True)
df_sales.drop(columns='id',inplace = True)
df_sales.drop(columns='week',inplace = True)
#
correllogram_plot(df_sales,'BASIC BEFORE')
#
# Azioni dopo analisi correlazione#
df_sales.drop(columns='base_price',inplace = True)
df_sales.drop(columns='center_region',inplace = True)
df_sales['with_promo']  = df_sales.apply(lambda x: (x['emailer_for_promotion'] or x['homepage_featured']), axis=1)
df_sales.drop(columns='emailer_for_promotion',inplace = True)
df_sales.drop(columns='homepage_featured',inplace = True)
#
print(df_sales)
print('meal_id:',len(meal_id_encode))
print('center_id:',len(center_id_encode))
#
# trova e rimuovi outliers sulle features
#
"""
outliers_opArea = df_sales[(np.abs(stats.zscore(df_sales['center_op_area'])) >= 3)]
print('Outliers per op area:',len(outliers_opArea.index))
if len(outliers_opArea.index) > 0:
	for i,s in outliers_opArea.iterrows():
		print(' '*10,s['center_op_area'])
	z_scores = np.abs(stats.zscore(df_sales['center_op_area']))
	not_outliers = (z_scores < 3)
	df_sales = df_sales[not_outliers]
#
outliers_checkoutprice = df_sales[(np.abs(stats.zscore(df_sales['checkout_price'])) >= 3)]
print('Outliers per checkout price:',len(outliers_checkoutprice.index))
if len(outliers_checkoutprice.index) > 0:
	for i,s in outliers_checkoutprice.iterrows():
		print(' '*10,s['checkout_price'])
	z_scores = np.abs(stats.zscore(df_sales['checkout_price']))
	not_outliers = (z_scores < 3)
	df_sales = df_sales[not_outliers]
"""
#
# trova e rimuovi outliers sulle labels
#
outliers_numOrders = df_sales[(np.abs(stats.zscore(df_sales['num_orders'])) >= 20)]
print('Outliers per num orders:',len(outliers_numOrders.index))
if len(outliers_numOrders.index) > 0:
	for i,s in outliers_numOrders.iterrows():
		print(' '*10,s['num_orders'])
	z_scores = np.abs(stats.zscore(df_sales['num_orders']))
	not_outliers = (z_scores < 20)
	df_sales = df_sales[not_outliers]
#





df_sales.to_pickle('sales')

#
# Scatterplot con/senza azioni di marketing, y=vendite settimana, x=costo di vendita (base price)
#
fig = plt.figure(figsize=(16, 10), dpi=80)
grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)
ax_main = fig.add_subplot(grid[:-1, :-1])
ax_right = fig.add_subplot(grid[:-1, -1], yticklabels=[])
ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[])
ax_main.scatter('checkout_price', 'num_orders', s=2**2, c=df_sales.with_promo.astype('category').cat.codes, alpha=.9, data=df_sales,
				cmap="brg", edgecolors='face', linewidths=.5)
#
ax_bottom.hist(df_sales.checkout_price, 40, histtype='stepfilled', orientation='vertical', color='green')
ax_bottom.invert_yaxis()
ax_right.hist(df_sales.num_orders, 40, histtype='stepfilled', orientation='horizontal', color='green')

#
ax_main.set(title='Scatterplot checkout price vs num orders (green= with promo)', xlabel='checkout price', ylabel='num orders')
ax_main.title.set_fontsize(20)
plt.show()

correllogram_plot(df_sales,'BASIC AFTER')

#
# Prepara seconda version con 1 HOT ENCODING ######################################################
#
col_list = ['meal_code','meal_cuisine','meal_category','center_code','center_type','center_city']
df_sales_1hot = df_sales.copy(deep=True)
df_sales_1hot = pd.get_dummies(df_sales_1hot,columns=col_list,drop_first=True)


#correllogram_plot(df_sales_1hot,'1HOT') # troppe colonne, correlogramma non leggibile

#
# PREPARO I DATASET ####################################################################
#
# metti a parte le label y
#
df_labels = df_sales['num_orders']
df_sales.drop(columns='num_orders',inplace = True)
df_sales_1hot.drop(columns='num_orders',inplace = True)

#
# Salva nomi colonne per elaborazioni successive (Linear Model)
#

# salva nomi colonne (per stampa coefficienti modello lineare)
with open('sales_columns.json','w') as file:
	json.dump(df_sales.columns.to_list(),file)
with open('sales_1hot_columns.json','w') as file:
	json.dump(df_sales_1hot.columns.to_list(),file)
	
#
# transforma pandas dataset in numpy arrays
#
np_sales      = df_sales.to_numpy()
np_sales_1hot = df_sales_1hot.to_numpy()
np_labels     = df_labels.to_numpy()
#
# split <train+validation>/test datasets
#
from sklearn.model_selection import train_test_split
x_trainVal,x_test,y_trainVal,y_test=train_test_split(np_sales,np_labels,test_size=0.2,random_state=729)
x_trainVal_1hot,x_test_1hot,y_trainVal_1hot,y_test_1hot=train_test_split(np_sales_1hot,np_labels,test_size=0.2,random_state=729)
#
# split train/validation datasets
#
x_train,x_val,y_train,y_val=train_test_split(x_trainVal,y_trainVal,test_size=0.125,random_state=638)
x_train_1hot,x_val_1hot,y_train_1hot,y_val_1hot=train_test_split(x_trainVal_1hot,y_trainVal_1hot,test_size=0.125,random_state=638)
#
# Nella grid search delle Neural networks i tempi sono troppo lunghi se utilizzo l'intero dataset (train+val)
# preparo quindi due dataset ridotti (per V1 e V2) con il 20% dei dati del train+val 
# (basta un solo sottoinsieme, senza ulteriori divisioni; ci pensa la grid search a fare i fold)
# ATTENZIONE: sempre senza toccare i dati di test e di validazione
#
x_gridsearch,x_out,y_gridsearch,y_out=train_test_split(x_train,y_train,test_size=0.8,random_state=729)
x_gridsearch_1hot,x_out,y_gridsearch_1hot,y_out=train_test_split(x_train_1hot,y_train_1hot,test_size=0.8,random_state=729)

print('-'*30)
print('training validation test')
print('basic',x_train.shape,x_val.shape,x_test.shape)
print('1Hot',x_train_1hot.shape,x_val_1hot.shape,x_test_1hot.shape)
print('gridsearch basic',x_gridsearch.shape)
print('gridsearch 1Hot ',x_gridsearch_1hot.shape)
#
# Salva tutti i dataset in files compressi (usa np.load per ricaricarli come dictionary di numpy arrays)
#
np.savez('datasets',x_train=x_train,x_val=x_val,x_test=x_test,y_train=y_train,y_val=y_val,y_test=y_test)
np.savez('datasets_1hot',x_train=x_train_1hot,x_val=x_val_1hot,x_test=x_test_1hot,y_train=y_train_1hot,y_val=y_val_1hot,y_test=y_test_1hot)
np.savez('datasets_gridsearch',x_train=x_gridsearch,y_train=y_gridsearch)
np.savez('datasets_1hot_gridsearch',x_train=x_gridsearch_1hot,y_train=y_gridsearch_1hot)

