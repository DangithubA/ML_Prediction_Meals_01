import json
import time
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sea
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor


from sklearn import metrics

datasets = np.load('datasets.npz')
datasets_1Hot = np.load('datasets_1hot.npz')
datasets_gridsearch = np.load('datasets_gridsearch.npz')
datasets_1Hot_gridsearch = np.load('datasets_1hot_gridsearch.npz')

#
# main routine per neural net
#
def neural_model(dataset_dict,gridsearch_dict,descr,plot=False):
	caption=descr
	print('-'*20,caption.upper(),'-'*20)
	
	X_train = dataset_dict['x_train']
	X_val   = dataset_dict['x_val']
	X_test  = dataset_dict['x_test']
	Y_train = dataset_dict['y_train']
	Y_val   = dataset_dict['y_val']
	Y_test  = dataset_dict['y_test']
	
	X_gridsearch = gridsearch_dict['x_train']
	Y_gridsearch = gridsearch_dict['y_train']
	
	#
	# GRID SEARCH E TRAINING
	#
	# utilizza il dataset ridooto per gridsearch (gridSearch tiene da parte un dataset di validazione per misurare la performance con differenti parameteri)
	estimator = MLPRegressor(random_state=314,
							max_iter=500,
							solver='adam',
							alpha=0.0002)
	param_grid = [{'hidden_layer_sizes': [(100,50,30),(50,30,20,10) ], 'activation': ['tanh','relu']}]
	gs = GridSearchCV(estimator,param_grid,scoring='neg_mean_absolute_error',verbose=3)
	start_time = time.process_time()
	gs.fit(X_gridsearch,Y_gridsearch)
	end_time = time.process_time()
	print('grid search time:',(end_time - start_time))
	#
	print('-'*10,'grid search results','-'*10)
	for p in gs.cv_results_:
		print(p,':',gs.cv_results_[p])
	print('-'*10,'grid search best parameters','-'*10)
	for p in gs.best_params_:
		print(p,':',gs.best_params_[p])

	#
	# Grafici x grid search
	#
	if plot:
		opt_num = len(gs.cv_results_['params'])
		line_list = ['r-','b-','g-','y-']
		for opt in range(opt_num):
			descr_plot = ""
			for par in gs.cv_results_['params'][opt].items():
				descr_plot = descr_plot + json.dumps(par)
			data_list = []
			for fold in range(5):
				data_list.append(gs.cv_results_['split'+str(fold)+'_test_score'][opt])
			plt.plot([1,2,3,4,5], data_list, line_list[opt], label=descr_plot)
		plt.legend()
		plt.show()	
	#
	# dopo grid search: full training con gli hyperparameters selezionati
	#
	print('Training con i migliori iper-parametri',caption.upper())
	nnm = MLPRegressor(	hidden_layer_sizes=gs.best_params_['hidden_layer_sizes'],
					random_state=314,
					max_iter=1000,
					activation=gs.best_params_['activation'],
					solver='adam',
					alpha=0.0002,
					verbose=False)
	start_time = time.process_time()
	nnm.fit(X_train, Y_train)
	end_time = time.process_time()
	print('training time after grid search:',(end_time - start_time))
	#
	# Report
	#

	Y_pred = nnm.predict(X_val)
	Y_true = Y_val
	dataset_type = 'validation'
		
	score = mean_absolute_error(Y_true,Y_pred)
	#print(caption+" Error on "+dataset_type+" (MSE) = ", mean_squared_error(Y_true,Y_pred))
	print(caption+" Error on "+dataset_type+" (MAE) = ", score)
	#print(caption+" Performance on "+dataset_type+" (R2) = ", r2_score(Y_true, Y_pred))
	#print(caption+" Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(Y_true, Y_pred)))

	if plot:
		plt.plot(Y_pred, (Y_true - Y_pred)/Y_true, 'o')
		plt.hlines(0, xmin=min(Y_pred), xmax=max(Y_pred), linestyles='--')
		plt.title('Residual % Plot '+caption)
		plt.show()
		"""
		df = pd.DataFrame({'Actual': Y_true, 'Predicted': Y_pred})
		df1 = df.head(25)
		df1.plot(kind='bar', figsize=(10, 8))
		plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
		plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
		plt.show()
		"""
		
	pickle.dump(nnm, open(descr+'.sav', 'wb'))
	
	return(score,gs.best_params_)

def nnm_ensemble(dataset_dict,gridsearch_dict,descr,hyper_params,plot=False):
	#https://scikit-learn.org/stable/modules/ensemble.html
	caption=descr
	print('-'*20,caption.upper()+' ENSEMBLE ','-'*20)
	print('parameters:', hyper_params)
	
	X_train = dataset_dict['x_train']
	X_val   = dataset_dict['x_val']
	X_test  = dataset_dict['x_test']
	Y_train = dataset_dict['y_train']
	Y_val   = dataset_dict['y_val']
	Y_test  = dataset_dict['y_test']
	
	X_gridsearch = gridsearch_dict['x_train']
	Y_gridsearch = gridsearch_dict['y_train']
	
	#
	# Grid Search
	#

	nnm_model = MLPRegressor(hidden_layer_sizes=hyper_params['hidden_layer_sizes'],
					random_state=314,
					max_iter=200,
					activation=hyper_params['activation'],
					solver='adam',
					alpha=0.0002,
					verbose=False)
	"""
	estimator = AdaBoostRegressor(base_estimator=nnm_model,
								random_state=314,
								loss='linear')
	param_grid = [{'n_estimators': [3,5], 'learning_rate': [1,0.5]}]
	gs = GridSearchCV(estimator,param_grid,scoring='neg_mean_absolute_error',verbose=3)
	start_time = time.process_time()
	gs.fit(X_gridsearch,Y_gridsearch)
	end_time = time.process_time()
	print('grid search time:',(end_time - start_time))
	#
	print('-'*10,'grid search results','-'*10)
	for p in gs.cv_results_:
		print(p,':',gs.cv_results_[p])
	print('-'*10,'grid search best parameters','-'*10)
	for p in gs.best_params_:
		print(p,':',gs.best_params_[p])
	#
	# Grafici x grid search
	#
	if plot:
		opt_num = len(gs.cv_results_['params'])
		line_list = ['r-','b-','g-','y-']
		for opt in range(opt_num):
			descr_plot = ""
			for par in gs.cv_results_['params'][opt].items():
				descr_plot = descr_plot + json.dumps(par)
			data_list = []
			for fold in range(5):
				data_list.append(gs.cv_results_['split'+str(fold)+'_test_score'][opt])
			plt.plot([1,2,3,4,5], data_list, line_list[opt], label=descr_plot)
		plt.legend()
		plt.show()	
	#
	# after grid search: full training with the selected hyperparameters
	#
	print('Training con i migliori iper-parametri',caption.upper(),'ENSEMBLE')
	nnm = AdaBoostRegressor (base_estimator=nnm_model,
							random_state=314,
							loss='linear',
							n_estimators=gs.best_params_['n_estimators'], 
							learning_rate=gs.best_params_['learning_rate'])
	"""
	#
	# Senza grid search: Training con parametri di default
	#
	print('Training con gli iper-parametri di default',caption.upper(),'ENSEMBLE')
	nnm = AdaBoostRegressor (base_estimator=nnm_model,
							random_state=314,
							loss='linear',
							n_estimators=3, 
							learning_rate=1)
	start_time = time.process_time()
	nnm.fit(X_train, Y_train)
	end_time = time.process_time()
	print('training time:',(end_time - start_time))
	#
	# Report
	#
	Y_pred = nnm.predict(X_val)
	Y_true = Y_val
	dataset_type = 'validation'
	
	score = mean_absolute_error(Y_true,Y_pred)
	#print(caption+" Error on "+dataset_type+" (MSE) = ", mean_squared_error(Y_true,Y_pred))
	print(caption+" Error on "+dataset_type+" (MAE) = ", score)
	#print(caption+" Performance on "+dataset_type+" (R2) = ", r2_score(Y_true, Y_pred))
	#print(caption+" Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(Y_true, Y_pred)))

	if plot:
		plt.plot(Y_pred, (Y_true - Y_pred)/Y_true, 'o')
		plt.hlines(0, xmin=min(Y_pred), xmax=max(Y_pred), linestyles='--')
		plt.title('Residual % Plot '+caption)
		plt.show()
		"""
		df = pd.DataFrame({'Actual': Y_true, 'Predicted': Y_pred})
		df1 = df.head(25)
		df1.plot(kind='bar', figsize=(10, 8))
		plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
		plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
		plt.show()
		"""
			
	pickle.dump(nnm, open('Neural_Ensemble_'+descr+'.sav', 'wb'))
	
	return(score)
	
#
# ESECUZIONE PROGRAMMA #########################################
#
	
nnm_result_basic = neural_model(datasets,datasets_gridsearch,'Neural_Basic',plot=True)
nnm_result_1Hot  = neural_model(datasets_1Hot,datasets_1Hot_gridsearch,'Neural_1Hot',plot=True)

#
# Confronta i punteggi e addestras l'Ensemble con il modello col punteggio migliore
#
if nnm_result_basic[0] < nnm_result_1Hot[0]: 
	ensemble_type = 'Basic'
	nnm_result_ensemble = nnm_ensemble(datasets,datasets_gridsearch,'Basic',nnm_result_basic[1],plot=True)
else:
	ensemble_type = '1Hot'
	nnm_result_ensemble = nnm_ensemble(datasets_1Hot,datasets_1Hot_gridsearch,'1Hot',nnm_result_1Hot[1],plot=True)

	
#
# REPORT fINALE
#
print('-'*20,'FINAL REPORT FOR NEURAL NETWORK MODELS','-'*20)
print('Neural Network Basic    '+' '*len(ensemble_type)+': MAE =',nnm_result_basic[0],'- parameters:',nnm_result_basic[1])
print('Neural Network 1Hot     '+' '*len(ensemble_type)+': MAE =',nnm_result_1Hot[0],'- parameters:',nnm_result_1Hot[1])
print('Neural Network Ensemble '+ensemble_type+': MAE =',nnm_result_ensemble)


