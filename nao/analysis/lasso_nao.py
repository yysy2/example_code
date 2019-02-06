def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn
#-----------------BEGIN HEADERS-----------------
import numpy as np
from scipy.stats.stats import pearsonr
from sklearn import preprocessing
from sklearn import linear_model
import sys
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

##################READ IN DATA################
print("Reading in data")
data = []
data_train = pd.read_csv("data/train.csv")
header = data_train.columns.values

##################BASIC VISUALISATION/ANALYSIS################
print("Showing the head of training data")
print(data_train.head(15))

#Seperating x and y
print("Spliting x and y")
y_raw = data_train.iloc[:,0]
y_raw_name = data_train.iloc[:,0].name
x_raw = data_train.iloc[:, data_train.columns != y_raw_name]
x_raw_name = x_raw.columns
y_raw = pd.DataFrame(y_raw, columns=[y_raw_name])
y_train_raw = y_raw
x_train_raw = x_raw

#Printing dimensions of data
print("Dimensions of x_train, y_train")
print(np.shape(x_train_raw), np.shape(y_train_raw))

#Treating with NaNs
print("Is there nans?")
print(x_train_raw.isnull().values.any())

#Change to numpy arrays
x_train_raw = np.array(x_train_raw)[:,:]
y_train_raw = np.array(y_train_raw)[:,:]

best_param_list = []
for i in range(0,np.shape(y_train_raw)[0]):
  ## Doing Scaife pullout
  print("Now doing i", i)
  mypullout = i
  x_test = x_train_raw[mypullout,:]
  x_test = np.expand_dims(x_test,axis=0)

  x_train = np.vstack((x_train_raw[:mypullout,:],x_train_raw[(mypullout+1):,:]))
  y_train = np.vstack((y_train_raw[:mypullout,:],y_train_raw[(mypullout+1):,:]))
  x_train = np.array(x_train)
  y_train = np.array(y_train)
  #del(x_train_raw,y_train_raw)

  ## To change shape from (N,1) to (N)
  a, b = y_train.shape
  y_train = y_train.reshape(a)

  #'''
  ##################SCALER################
  ## Scale the data (mean 0, std 1)
  scaler = preprocessing.StandardScaler()
  x_train_norm = scaler.fit_transform(x_train)
  x_test_norm = scaler.transform(x_test)
  #'''

  ## Grid search options
  #n = np.logspace(np.log(30)/np.log(10),np.log(10000)/np.log(10),endpoint=True,num=150,base=10.0)
  #m = [0.00001, 0.0001, 0.001, 0.01]
  #alpha_op = np.hstack((np.array(m),n))
  #alpha_op = [1E-3, 5E-3, 1E-2, 5E-2, 1E-1, 5E-1, 1E0, 5E0, 1E1, 5E1, 1E2, 5E2, 1E3, 5E3, 1E4, 5E4, 1E5, 5E5, 1E6, 5E6, 1E7, 5E7, 1E8, 5E8, 1E9]
  #a = [1E-15]
  alpha_op = (np.logspace(np.log(0.5)/np.log(10),np.log(10)/np.log(10),endpoint=True,num=5,base=10.0))#.astype(int)
  #alpha_op = np.hstack((a,b))

  #'''
  ## Grid search
  clf = linear_model.Lasso(alpha=0, copy_X=True, fit_intercept=False, normalize=False, random_state=None, tol=0.001)
  param_dist = {'alpha': alpha_op}
  grid_search = GridSearchCV(clf, param_grid=param_dist, verbose=1, cv=6, scoring='neg_mean_absolute_error')
  grid_search.fit(x_train_norm, y_train)
  #print("Best scores are:", grid_search.best_params_, grid_search.best_score_)
  best_params = grid_search.best_params_
  best_score = grid_search.best_score_
  best_param_list.append(best_params)
  print(best_params)
  print(best_score)
  #'''

  ## Running
  reg = linear_model.Lasso(alpha=best_params.items()[0][1], copy_X=True, fit_intercept=False, normalize=False, random_state=None, tol=0.001)#best_params.items()[0][1]
  reg.fit(x_train_norm,y_train)

  ## Predicting
  p_test = reg.predict(x_test_norm)
  #print(p_test)

  e = open('ridgea_regression_scaife3.csv', 'a+')
  e.write(str(p_test[0]) + '\n')
  e.close()

  del(p_test, clf, reg, x_train_norm, x_train, y_train, a, b, x_test, mypullout, scaler, alpha_op, param_dist, grid_search, best_params)

print("best_params are: ")
for i in range(len(best_param_list)):
  print(best_param_list[i])

#print("alpha_op is: ")
#print(alpha_op)
