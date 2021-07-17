import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

linear_ensemble = '1Hot'
neural_ensemble = '1Hot'

datasets = np.load('datasets.npz')
datasets_1Hot = np.load('datasets_1hot.npz')

#### LINEAR ####
descr='Linear_Basic'
X_test  = datasets['x_test']
Y_test  = datasets['y_test']
lr = pickle.load(open(descr+'.sav', 'rb'))
Y_pred = lr.predict(X_test)
score = mean_absolute_error(Y_test,Y_pred)
print(descr,": Error on TEST (MAE) = ", score)

descr='Linear_1Hot'
X_test  = datasets_1Hot['x_test']
Y_test  = datasets_1Hot['y_test']
lr = pickle.load(open(descr+'.sav', 'rb'))
Y_pred = lr.predict(X_test)
score = mean_absolute_error(Y_test,Y_pred)
print(descr,": Error on TEST (MAE) = ", score)

descr='Linear_Ensemble_'+linear_ensemble
if linear_ensemble == '1Hot':
	X_test  = datasets_1Hot['x_test']
	Y_test  = datasets_1Hot['y_test']
else:
	X_test  = datasets['x_test']
	Y_test  = datasets['y_test']
lr = pickle.load(open(descr+'.sav', 'rb'))
Y_pred = lr.predict(X_test)
score = mean_absolute_error(Y_test,Y_pred)
print(descr,": Error on TEST (MAE) = ", score)

#### NEURAL ####
descr='Neural_Basic'
X_test  = datasets['x_test']
Y_test  = datasets['y_test']
nnm = pickle.load(open(descr+'.sav', 'rb'))
Y_pred = nnm.predict(X_test)
score = mean_absolute_error(Y_test,Y_pred)
print(descr,": Error on TEST (MAE) = ", score)

descr='Neural_1Hot'
X_test  = datasets_1Hot['x_test']
Y_test  = datasets_1Hot['y_test']
nnm = pickle.load(open(descr+'.sav', 'rb'))
Y_pred = nnm.predict(X_test)
score = mean_absolute_error(Y_test,Y_pred)
print(descr,": Error on TEST (MAE) = ", score)

descr='Neural_Ensemble_'+neural_ensemble
if neural_ensemble == '1Hot':
	X_test  = datasets_1Hot['x_test']
	Y_test  = datasets_1Hot['y_test']
else:
	X_test  = datasets['x_test']
	Y_test  = datasets['y_test']
nnm = pickle.load(open(descr+'.sav', 'rb'))
Y_pred = nnm.predict(X_test)
score = mean_absolute_error(Y_test,Y_pred)
print(descr,": Error on TEST (MAE) = ", score)