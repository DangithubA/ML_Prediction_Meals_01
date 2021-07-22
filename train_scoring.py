import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

linear_ensemble = '1Hot'
neural_ensemble = '1Hot'

datasets = np.load('datasets.npz')
datasets_1Hot = np.load('datasets_1hot.npz')

#### LINEAR ####
descr='Linear_Basic'
X_train  = datasets['x_train']
Y_train  = datasets['y_train']
lr = pickle.load(open(descr+'.sav', 'rb'))
Y_pred = lr.predict(X_train)
score = mean_absolute_error(Y_train,Y_pred)
print(descr,": Error on TRAIN (MAE) = ", score)

descr='Linear_1Hot'
X_train  = datasets_1Hot['x_train']
Y_train  = datasets_1Hot['y_train']
lr = pickle.load(open(descr+'.sav', 'rb'))
Y_pred = lr.predict(X_train)
score = mean_absolute_error(Y_train,Y_pred)
print(descr,": Error on TRAIN (MAE) = ", score)

descr='Linear_Ensemble_'+linear_ensemble
if linear_ensemble == '1Hot':
	X_train  = datasets_1Hot['x_train']
	Y_train  = datasets_1Hot['y_train']
else:
	X_train  = datasets['x_train']
	Y_train  = datasets['y_train']
lr = pickle.load(open(descr+'.sav', 'rb'))
Y_pred = lr.predict(X_train)
score = mean_absolute_error(Y_train,Y_pred)
print(descr,": Error on TRAIN (MAE) = ", score)

#### NEURAL ####
descr='Neural_Basic'
X_train  = datasets['x_train']
Y_train  = datasets['y_train']
nnm = pickle.load(open(descr+'.sav', 'rb'))
Y_pred = nnm.predict(X_train)
score = mean_absolute_error(Y_train,Y_pred)
print(descr,": Error on TRAIN (MAE) = ", score)

descr='Neural_1Hot'
X_train  = datasets_1Hot['x_train']
Y_train  = datasets_1Hot['y_train']
nnm = pickle.load(open(descr+'.sav', 'rb'))
Y_pred = nnm.predict(X_train)
score = mean_absolute_error(Y_train,Y_pred)
print(descr,": Error on TRAIN (MAE) = ", score)

descr='Neural_Ensemble_'+neural_ensemble
if neural_ensemble == '1Hot':
	X_train  = datasets_1Hot['x_train']
	Y_train  = datasets_1Hot['y_train']
else:
	X_train  = datasets['x_train']
	Y_train  = datasets['y_train']
nnm = pickle.load(open(descr+'.sav', 'rb'))
Y_pred = nnm.predict(X_train)
score = mean_absolute_error(Y_train,Y_pred)
print(descr,": Error on TRAIN (MAE) = ", score)
