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
from matplotlib import ticker
import sys
#-----------------END HEADERS-----------------




#----------------------------------BEGIN FUNCTIONS----------------------------------
#-----------------BEGIN FUNCTION 1-----------------
def seasonname():
	if season == 0:
		season_name = 'djf'
	elif season == 1:
		season_name = 'mam'
	elif season == 2:
		season_name = 'jja'
	elif season == 3:
		season_name = 'son'
	else:
		print('Error, fifth season type')
		exit()

	return season_name
#-----------------END FUNCTION 1-----------------



#-----------------BEGIN FUNCTION 2-----------------
def maindataread():
	data_hold = []
	clean_data = []

	data_hold.append(data[a].variables[variable_type][0,0,:,:])

	clean_data_fn = np.array(data_hold)/100.0 #hpa
	del(data_hold)

	clean_data = clean_data_fn[:,:]
	del(clean_data_fn)

	return clean_data
#-----------------END FUNCTION 2-----------------



#-----------------BEGIN FUNCTION 3-----------------
def finalstat():
	results_mean = []
	results_stdev = []
	results_sterr = []

	for i in range(0,9): #Because we have 9 jobs
		results_mean.append(np.mean(results[i]))
		results_stdev.append(np.std(results[i]))
		#results_sterr.append(2*(np.std(results[i])/np.sqrt(float(no_of_years))))
		results_sterr.append(2*(stats.sem(results[i])))

	return (results_mean, results_stdev, results_sterr)
#-----------------END FUNCTION 3-----------------




#-----------------BEGIN FUNCTION 3-----------------
def preplot():
	lats = data[0].variables['latitude'][:]
	lons1 = data[0].variables['longitude'][:]
	nlats = len(lats)
	nlons = len(lons1)
	#cbarunits = 'Mean sea pressure hPa'
	cbarunits = 'hPa'

	# create Basemap instance.
	#m =\
	#Basemap(projection='spstere',boundinglat=blat,lon_0=165,resolution='l')
	m = Basemap(projection='mill', resolution='c', llcrnrlon=0, llcrnrlat=-90,urcrnrlon=360.01, urcrnrlat=20)


	# add wrap-around point in longitude.
	results_plot, lons = addcyclic(mydiff, lons1)

	# find x,y of map projection grid.
	lons, lats = np.meshgrid(lons, lats)
	x, y = m(lons, lats)

	# create figure.
	fig=plt.figure(figsize=(18,10))
	ax = fig.add_axes([0.09,0.10,0.80,0.80]) #0.5, 0.5, 0.9, 0.85
	#levels = [-10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5, -5, -4, -3, -2, -1, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25]
	clevs = np.linspace(-8.0,8.0,25)
	orig_cmap = plt.cm.coolwarm
	cs = m.contourf(x,y,results_plot,clevs,cmap=orig_cmap, extend='both')

	m.drawcoastlines(linewidth=1.25)
	#m.fillcontinents(color='0.8')
	m.drawparallels(np.arange(-80,81,20),labels=[1,0,0,0], fontsize=40)
	m.drawmeridians(np.arange(0,360,60),labels=[0,0,0,1], fontsize=40)

	# add colorbar.
	#plt.colorbar(orientation='horizontal')
	#cbar = m.colorbar(pad="15%",location='bottom')
	#cbar.set_label('hPa', rotation=0, fontsize=30)
	#tick_locator = ticker.MaxNLocator(nbins=8)
	#cbar.locator = tick_locator
	##cbar.ax.xaxis.set_major_locator(ticker.AutoLocator())
	#cbar.ax.tick_params(labelsize=30)
	#cbar.update_ticks()

	plt.title(plottitle + ' ' + str(float(myjob)*0.75-3) + 'K', fontsize=55, y=1.02)
	#plt.show()
	plt.savefig(filename)
#-----------------END FUNCTION 3-----------------




#----------------------------------BEGIN BODY----------------------------------
#-----------------INPUT VARIABLES-----------------
no_of_years = 55
season = int(sys.argv[1])
myjob = int(sys.argv[2])
#Get season name
#-----------------CALLING FUNCTION 1-----------------
season_name = seasonname();
#-----------------END CALLING FUNCTION 1-----------------
variable_name = 'psl'
datadir = '/home/scottyiu/Desktop/work/data/model/50yrs/seasonal/2D/psl/'
data = [] 
results = []
mydiff = np.zeros((145,192))
el_nino_strength = [-3.0, -2.25, -1.5, -0.75, 0.0, 0.75, 1.50, 2.25, 3.0]
#We go by order of El Nino strength
data.append((NetCDFFile(datadir + variable_name + '_sea_xlnjj_' + season_name + '_timmean.nc')))
data.append((NetCDFFile(datadir + variable_name + '_sea_xlnji_' + season_name + '_timmean.nc')))
data.append((NetCDFFile(datadir + variable_name + '_sea_xlnjh_' + season_name + '_timmean.nc')))
data.append((NetCDFFile(datadir + variable_name + '_sea_xlnjg_' + season_name + '_timmean.nc')))
data.append((NetCDFFile(datadir + variable_name + '_sea_xlnja_' + season_name + '_timmean.nc')))
data.append((NetCDFFile(datadir + variable_name + '_sea_xlnjc_' + season_name + '_timmean.nc')))
data.append((NetCDFFile(datadir + variable_name + '_sea_xlnjd_' + season_name + '_timmean.nc')))
data.append((NetCDFFile(datadir + variable_name + '_sea_xlnje_' + season_name + '_timmean.nc')))
data.append((NetCDFFile(datadir + variable_name + '_sea_xlnjf_' + season_name + '_timmean.nc')))

variable_type = 'p'
#plottitle = 'Model_' + str.upper(season_name) +  '_' + variable_name
plottitle = 'SLP anomaly: ' + str.upper(season_name)
filename = 'model_' + season_name +  '_' + variable_name + '_' + str(myjob) + '.png'
#-----------------END INPUT VARIABLES-----------------


#-----------------CALLING FUNCTION 2-----------------
for a in range(0,9): #9 because we have 9 jobs
	results.append(maindataread());
#-----------------END CALLING FUNCTION 2-----------------


#-----------------BEGIN ANALYSIS-----------------
mydiff = results[myjob][0,:,:] - results[4][0,:,:]
#-----------------END ANALYSIS-----------------
print(np.min(mydiff[:71,:]))
print(np.max(mydiff[12:24,90:155]))
exit()

#-----------------BEGIN PLOT-----------------
preplot();
#-----------------END PLOT-----------------

