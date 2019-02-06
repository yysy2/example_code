#-----------------BEGIN HEADERS-----------------
from netCDF4 import Dataset as NetCDFFile
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from mpl_toolkits.basemap import Basemap, addcyclic, cm
from scipy.ndimage.filters import minimum_filter, maximum_filter
from netCDF4 import Dataset
from scipy.stats.stats import pearsonr
from scipy import stats
import sys
import matplotlib.pyplot as plt
import pandas as pd
#-----------------END HEADERS-----------------

## Load the dataset
data = pd.read_csv("summary.csv",header=None)

## NAO index
nao = data.ix[:,0]
## ML index
ml = data.ix[:,1]
## Scaife index
scaife = data.ix[:,2]
## svm_rbf index
svm_rbf = data.ix[:,3]
## svm_linear index
svm_linear = data.ix[:,4]
## lasso index
lasso = data.ix[:,5]
## years
x = np.linspace(1980,2018,39)

fig=plt.figure(figsize=(28,12))
ax = fig.add_axes([0.08,0.10,0.9,0.80]) #0.5, 0.5, 0.9, 0.85

plt.xlim(1980,2018)
plt.ylim(-3,3)

plt.plot(x,nao,label="NAO",lw=5,marker='x',color='dimgrey',ls='-')
plt.plot(x,ml,label="ML",lw=3,marker='x',color='r',ls='-')
plt.plot(x,scaife,label="LR",lw=3,marker='x',color='b',ls='-')
#plt.plot(x,svm_rbf,label="SVM rbf",lw=3,marker='x',color='r',ls='--')
#plt.plot(x,svm_linear,label="SVM linear",lw=3,marker='x',color='b',ls='--')
#plt.plot(x,lasso,label="Lasso",lw=3,marker='x',color='y',ls='--')
plt.axhline(0, color='black', lw=2, alpha=0.8, ls='-')

plt.xlabel('Year', fontsize=40)
plt.ylabel('NAO index', fontsize=40)
plt.title("NAO index predictions", fontsize=50, y=1.03)
plt.xticks(x[::5], y=-0.02)

plt.tick_params(axis='both', direction='in', length=18, width=3, which='major', labelsize=30)
plt.tick_params(axis='both', direction='in', length=18, width=3, which='minor', labelsize=30)

plt.legend(fontsize=35)

#plt.show()
plt.savefig("figure_1.png")
