import json
import time
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sea
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor

with open('sales_columns.json') as file:
    col_list = json.load(file)
with open('sales_1hot_columns.json') as file:
    col_list_1hot = json.load(file)

datasets = np.load('datasets.npz')
datasets_1Hot = np.load('datasets_1hot.npz')


# Linear Regression  ############################################à
#
#
def linear_model(dataset_dict, columns, descr, plot=False):
    caption = descr
    print('-' * 20, caption.upper(), '-' * 20)

    colums_local = [x for x in columns]

    X_train = dataset_dict['x_train']
    X_val = dataset_dict['x_val']
    X_test = dataset_dict['x_test']
    Y_train = dataset_dict['y_train']
    Y_val = dataset_dict['y_val']
    Y_test = dataset_dict['y_test']
    #
    # Grid Search
    #
    estimator = ElasticNet(random_state=314,
                           max_iter=500,
                           alpha=0.1)
    param_grid = [{'l1_ratio': [0.2, 0.5, 0.7], 'normalize': [True, False]}]
    gs = GridSearchCV(estimator, param_grid, scoring='neg_mean_absolute_error', verbose=3)
    start_time = time.process_time()
    gs.fit(X_train, Y_train)
    end_time = time.process_time()
    print('grid search time:', (end_time - start_time))
    #
    print('-' * 10, 'grid search results', '-' * 10)
    for p in gs.cv_results_:
        print(p, ':', gs.cv_results_[p])
    print('-' * 10, 'grid search best parameters', '-' * 10)
    for p in gs.best_params_:
        print(p, ':', gs.best_params_[p])
    #
    # Grafici x grid search
    #
    if plot:
        opt_num = len(gs.cv_results_['params'])
        line_list = ['r-', 'b-', 'g-', 'y-', 'o-', 'v-']
        for opt in range(opt_num):
            descr_plot = ""
            for par in gs.cv_results_['params'][opt].items():
                descr_plot = descr_plot + json.dumps(par)
            data_list = []
            for fold in range(5):
                data_list.append(gs.cv_results_['split' + str(fold) + '_test_score'][opt])
            plt.plot([1, 2, 3, 4, 5], data_list, line_list[opt], label=descr_plot)
        plt.legend()
        plt.show()
    #
    # dopo grid search: training completo con gli hyperparameters selezionati
    #
    lr = ElasticNet(l1_ratio=gs.best_params_['l1_ratio'],
                    random_state=314,
                    max_iter=2000,
                    alpha=0.1,
                    normalize=gs.best_params_['normalize']
                    )
    start_time = time.process_time()
    lr.fit(X_train, Y_train)
    end_time = time.process_time()
    print('training time after grid search:', (end_time - start_time))

    #
    # Report
    #
    # print(len(lr.coef_),len(colums_local))
    # print(lr.coef_)
    # print(colums_local)
    pd.set_option('display.max_rows', None)
    coeff_df = pd.DataFrame(lr.coef_, colums_local, columns=['Coefficient'])
    print('LINEAR MODEL COEFFICENTS')
    print(coeff_df)
    pd.reset_option('display.max_rows')

    Y_pred = lr.predict(X_val)
    Y_true = Y_val
    dataset_type = 'validation'

    score = mean_absolute_error(Y_true, Y_pred)
    # print(caption+" Error on "+dataset_type+" (MSE) = ", mean_squared_error(Y_true,Y_pred))
    print(caption + " Error on " + dataset_type + " (MAE) = ", score)
    # print(caption+" Performance on "+dataset_type+" (R2) = ", r2_score(Y_true, Y_pred))
    # print(caption+" Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(Y_true, Y_pred)))

    if plot:
        plt.plot(Y_pred, (Y_pred - Y_true) / Y_true, 'o')
        plt.hlines(0, xmin=min(Y_pred), xmax=max(Y_pred), linestyles='--')
        plt.title('Residual % Plot ' + caption)
        plt.show()
        """
        df = pd.DataFrame({'Actual': Y_true, 'Predicted': Y_pred})
        df1 = df.head(25)
        df1.plot(kind='bar', figsize=(10, 8))
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        plt.show()
        """

    pickle.dump(lr, open('Linear_' + descr + '.sav', 'wb'))

    return (score, gs.best_params_)


def lr_ensemble(dataset_dict, columns, descr, hyper_params, plot=False):
    # https://scikit-learn.org/stable/modules/ensemble.html
    caption = descr
    colums_local = [x for x in columns]
    print('-' * 20, caption.upper() + ' ENSEMBLE ', '-' * 20)
    print('parameters:', hyper_params)

    X_train = dataset_dict['x_train']
    X_val = dataset_dict['x_val']
    X_test = dataset_dict['x_test']
    Y_train = dataset_dict['y_train']
    Y_val = dataset_dict['y_val']
    Y_test = dataset_dict['y_test']
    #
    # Grid Search
    #
    linear_model = lr = ElasticNet(l1_ratio=hyper_params['l1_ratio'],
                                   random_state=314,
                                   max_iter=2000,
                                   alpha=0.1,
                                   normalize=hyper_params['normalize']
                                   )
    """
    estimator = AdaBoostRegressor(base_estimator=linear_model,
                                random_state=314,
                                loss='linear')
    param_grid = [{'n_estimators': [5,10], 'learning_rate': [1,2]}]
    gs = GridSearchCV(estimator,param_grid,scoring='neg_mean_absolute_error',verbose=3)
    start_time = time.process_time()
    gs.fit(X_train,Y_train)
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
    lr = AdaBoostRegressor (base_estimator=linear_model,
                            random_state=314,
                            loss='linear',
                            n_estimators=gs.best_params_['n_estimators'], 
                            learning_rate=gs.best_params_['learning_rate'])
    """
    #
    # Senza Grid Search: full training con iperparamentei di default
    #
    #lr = AdaBoostRegressor(base_estimator=linear_model,
    #                       random_state=314,
    #                       loss='linear',
    #                       n_estimators=5,
    #                       learning_rate=1)
    lr = BaggingRegressor(base_estimator=linear_model,
                          bootstrap=True,
                          random_state=314,
                          n_estimators=5
                          )
    start_time = time.process_time()
    lr.fit(X_train, Y_train)
    end_time = time.process_time()
    print('training time after grid search:', (end_time - start_time))
    #
    # Report
    #

    Y_pred = lr.predict(X_val)
    Y_true = Y_val
    dataset_type = 'validation'

    score = mean_absolute_error(Y_true, Y_pred)
    # print(caption+" Error on "+dataset_type+" (MSE) = ", mean_squared_error(Y_true,Y_pred))
    print(caption + " Error on " + dataset_type + " (MAE) = ", score)
    # print(caption+" Performance on "+dataset_type+" (R2) = ", r2_score(Y_true, Y_pred))
    # print(caption+" Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(Y_true, Y_pred)))
    Y_pred = lr.predict(X_train)
    Y_true = Y_train
    score = mean_absolute_error(Y_true, Y_pred)
    print(caption + " TRAIN Error on " + dataset_type + " (MAE) = ", score)

    if plot:
        plt.plot(Y_pred, (Y_pred - Y_true) / Y_true, 'o')
        plt.hlines(0, xmin=min(Y_pred), xmax=max(Y_pred), linestyles='--')
        plt.title('Residual % Plot ' + caption)
        plt.show()
        """
        df = pd.DataFrame({'Actual': Y_true, 'Predicted': Y_pred})
        df1 = df.head(25)
        df1.plot(kind='bar', figsize=(10, 8))
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        plt.show()
        """

    pickle.dump(lr, open('Linear_Ensemble_' + descr + '.sav', 'wb'))

    return (score)


#
# ESECUZIONE PROGRAMMA #########################################
#

lr_result_basic = linear_model(datasets, col_list, 'Basic', plot=True)
lr_result_1Hot = linear_model(datasets_1Hot, col_list_1hot, '1Hot', plot=True)

#
# Confronta i punteggi e addestras l'Ensemble con il modello col punteggio migliore
#
if lr_result_basic[0] < lr_result_1Hot[0]:
    ensemble_type = 'Basic'
    lr_result_ensemble = lr_ensemble(datasets, col_list, 'Basic', lr_result_basic[1], plot=True)
else:
    ensemble_type = '1Hot'
    lr_result_ensemble = lr_ensemble(datasets_1Hot, col_list_1hot, '1Hot', lr_result_1Hot[1], plot=True)

#
# REPORT FINALE
#
print('-' * 20, 'FINAL REPORT FOR LINEAR REGRESSION MODELS', '-' * 20)
print('Linear Regression Basic    ' + ' ' * len(ensemble_type) + ': MAE =', lr_result_basic[0], '- parameters:',
      lr_result_basic[1])
print('Linear Regression 1Hot     ' + ' ' * len(ensemble_type) + ': MAE =', lr_result_1Hot[0], '- parameters:',
      lr_result_1Hot[1])
print('Linear Regression Ensemble ' + ensemble_type + ': MAE =', lr_result_ensemble)

