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

bigcoef = np.zeros((np.shape(x_train_raw)[0],np.shape(x_train_raw)[1]))
#best_param_list = []
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
  #alpha_op = np.logspace(np.log(1)/np.log(10),np.log(100000)/np.log(10),endpoint=True,num=10,base=10.0)
  #alpha_op = np.hstack((a,b))

  '''
  ## Grid search
  clf = linear_model.Ridge(alpha=0, copy_X=True, fit_intercept=False, max_iter=None, normalize=False, random_state=None, solver='auto', tol=0.001)
  param_dist = {'alpha': alpha_op}
  grid_search = GridSearchCV(clf, param_grid=param_dist, verbose=1, cv=6, scoring='neg_mean_absolute_error')
  grid_search.fit(x_train_norm, y_train)
  #print("Best scores are:", grid_search.best_params_, grid_search.best_score_)
  best_params = grid_search.best_params_
  best_score = grid_search.best_score_
  best_param_list.append(best_params)
  print(best_params)
  print(best_score)
  '''

  myalpha = [21981.1138131223, 13456.836055829, 35905.1237943963, 8238.2739188291, 1, 11903.2698171941, 29870.4251053972, 29870.4251053972, 70498.2257304827, 12656.2376106914, 3711.4001932522, 11195.0989464067, 8759.404249415, 407.9035502696, 13456.836055829, 962.7073087077, 9313.4998375453, 15213.1674249611, 18286.6717755351, 7287.1807962094, 7748.1476170246, 9902.6459738686, 8238.2739188291, 14308.0781353604, 10529.0598587288, 2568.6665835501, 10529.0598587288, 11903.2698171941, 5701.7238291636, 18286.6717755351, 29870.4251053972, 55160.0714737772, 1, 40591.3141418927, 13456.836055829, 8238.2739188291, 6853.638647767, 16175.5101496073, 13456.836055829]

  ## Running
  reg = linear_model.Ridge(alpha=myalpha[i], copy_X=True, fit_intercept=False, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)#best_params.items()[0][1]
  reg.fit(x_train_norm,y_train)
  bigcoef[i] = reg.coef_

  ## Predicting
  p_test = reg.predict(x_test_norm)
  #print(p_test)

  e = open('ridgea_regression_scaife3.csv', 'a+')
  e.write(str(p_test[0]) + '\n')
  e.close()

  del(p_test, reg, x_train_norm, x_train, y_train, a, b, x_test, mypullout, scaler)

seaice_index = np.load("seaice_index.npy")
sst_index = np.load("sst_index.npy")
z70_index = np.load("z70_index.npy")
seaice_index = seaice_index[0,:,:]
sst_index = sst_index[0,:,:]
z70_index = z70_index[0,:,:]

## Change bigcoef into seaice,z70,sst
seaice_coef = bigcoef[:,:np.shape(seaice_index)[0]]
z70_coef = bigcoef[:,np.shape(seaice_index)[0]:np.shape(seaice_index)[0]+np.shape(z70_index)[0]]
sst_coef = bigcoef[:,np.shape(seaice_index)[0]+np.shape(z70_index)[0]:]
seaice_coef_sum = np.sum(seaice_coef,axis=0)
z70_coef_sum = np.sum(z70_coef,axis=0)
sst_coef_sum = np.sum(sst_coef,axis=0)

mylon = 480
mylat = 241

seaice_coef_map = np.zeros((480,241))
z70_coef_map = np.zeros((480,241))
sst_coef_map = np.zeros((480,241))

for i in range(0,mylon):
  print(str((float(i)/float(mylon))*100.0) + '%')
  for j in range(0,mylat):
    for q in range(0,np.shape(seaice_index)[0]):
      if seaice_index[q,2] == i and seaice_index[q,1] == j:
        seaice_coef_map[i,j] = np.abs(seaice_coef_sum[q])
    for q in range(0,np.shape(z70_index)[0]):
      if z70_index[q,2] == i and z70_index[q,1] == j:
        z70_coef_map[i,j] = np.abs(z70_coef_sum[q])
    for q in range(0,np.shape(sst_index)[0]):
      if sst_index[q,2] == i and sst_index[q,1] == j:
        sst_coef_map[i,j] = np.abs(sst_coef_sum[q])

seaice_coef_map = seaice_coef_map.T
z70_coef_map = z70_coef_map.T
sst_coef_map = sst_coef_map.T

import matplotlib.pyplot as plt
from matplotlib import ticker
from netCDF4 import Dataset as NetCDFFile
from mpl_toolkits.basemap import Basemap, addcyclic, cm
mydata = NetCDFFile("/home/scottyiu/Desktop/datascience_final/final/tuned/weights/data_production/data_index/z70_nov.nc")
lats = mydata.variables['latitude'][:]
lons1 = mydata.variables['longitude'][:]
nlats = len(lats)
nlons = len(lons1)
m = Basemap(projection='mill', resolution='c', llcrnrlon=0, llcrnrlat=-90,urcrnrlon=360.01, urcrnrlat=90)
results_plot_seaice, lons = addcyclic(seaice_coef_map, lons1)
results_plot_z70, hold = addcyclic(z70_coef_map, lons1)
results_plot_sst, hold = addcyclic(sst_coef_map, lons1)
lons, lats = np.meshgrid(lons, lats)
x, y = m(lons, lats)

## SEAICE
fig=plt.figure(figsize=(18,10))
ax = fig.add_axes([0.09,0.10,0.80,0.80]) #0.5, 0.5, 0.9, 0.85
clevs = np.linspace(0,0.006,25)
orig_cmap = plt.cm.coolwarm
cs = m.contourf(x,y,results_plot_seaice,clevs,cmap=orig_cmap, extend='max')

m.drawcoastlines(linewidth=1.25)
m.fillcontinents(color='0.8')
m.drawparallels(np.arange(-80,81,20),labels=[1,0,0,0], fontsize=40)
m.drawmeridians(np.arange(0,360,60),labels=[0,0,0,1], fontsize=40)

cbar = m.colorbar(pad="15%",location='bottom')
cbar.set_label('contribution', rotation=0, fontsize=30)
tick_locator = ticker.MaxNLocator(nbins=8)
cbar.locator = tick_locator
cbar.ax.tick_params(labelsize=15)
cbar.update_ticks()

plt.title('seaice_contribution', fontsize=55, y=1.02)
plt.savefig('seaice_contribution.png')
plt.close()

## Z70
fig=plt.figure(figsize=(18,10))
ax = fig.add_axes([0.09,0.10,0.80,0.80]) #0.5, 0.5, 0.9, 0.85
clevs = np.linspace(0,0.003,25)
orig_cmap = plt.cm.coolwarm
cs = m.contourf(x,y,results_plot_z70,clevs,cmap=orig_cmap, extend='max')

m.drawcoastlines(linewidth=1.25)
#m.fillcontinents(color='0.8')
m.drawparallels(np.arange(-80,81,20),labels=[1,0,0,0], fontsize=40)
m.drawmeridians(np.arange(0,360,60),labels=[0,0,0,1], fontsize=40)

cbar = m.colorbar(pad="15%",location='bottom')
cbar.set_label('contribution', rotation=0, fontsize=30)
tick_locator = ticker.MaxNLocator(nbins=8)
cbar.locator = tick_locator
cbar.ax.tick_params(labelsize=15)
cbar.update_ticks()

plt.title('z70_contribution', fontsize=55, y=1.02)
plt.savefig('z70_contribution.png')
plt.close()

## SST
fig=plt.figure(figsize=(18,10))
ax = fig.add_axes([0.09,0.10,0.80,0.80]) #0.5, 0.5, 0.9, 0.85
clevs = np.linspace(0,0.006,25)
orig_cmap = plt.cm.coolwarm
cs = m.contourf(x,y,results_plot_sst,clevs,cmap=orig_cmap, extend='max')

m.drawcoastlines(linewidth=1.25)
m.fillcontinents(color='0.8')
m.drawparallels(np.arange(-80,81,20),labels=[1,0,0,0], fontsize=40)
m.drawmeridians(np.arange(0,360,60),labels=[0,0,0,1], fontsize=40)

cbar = m.colorbar(pad="15%",location='bottom')
cbar.set_label('contribution', rotation=0, fontsize=30)
tick_locator = ticker.MaxNLocator(nbins=8)
cbar.locator = tick_locator
cbar.ax.tick_params(labelsize=15)
cbar.update_ticks()

plt.title('sst_contribution', fontsize=55, y=1.02)
plt.savefig('sst_contribution.png')
plt.close()
