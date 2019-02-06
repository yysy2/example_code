# To run this code: python2.7 -c 'execfile("code.py")' n1 jobid n2
# n1 is season(DJF=0,SON=3)
# Jobid(e.g.xlnja)
# n2 = heightlevel(e.g. 7 = 200hPa)

#-----------------BEGIN HEADERS-----------------
import numpy as np
import matplotlib as mpl
mpl.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic
from netCDF4 import Dataset
import scipy
from scipy.ndimage.filters import gaussian_filter

from windspharm.standard import VectorWind
from windspharm.tools import prep_data, recover_data, order_latdim
from windspharm.examples import example_data_path
from netCDF4 import Dataset as NetCDFFile
from matplotlib.patches import Polygon
import sys
#-----------------END HEADERS-----------------


#----------------------------------BEGIN FUNCTIONS----------------------------------------------
#-----------------BEGIN FUNCTION 1-----------------
#Reading in data
def readin():
  ncu = (NetCDFFile('/home/scottyiu/Desktop/work/data/model/50yrs/seasonal/3D/uwind/uwind_sea_' + myjob + '_yseasmean.nc'))
  uwnd = ncu.variables['u'][:,myheight,:,:] #7 for model, 14 for era: 200hPa
  lons = ncu.variables['longitude'][:]
  lats = ncu.variables['latitude'][:]
  ncu.close()
  ncv = (NetCDFFile('/home/scottyiu/Desktop/work/data/model/50yrs/seasonal/3D/vwind/vwind_sea_' + myjob + '_yseasmean.nc'))
  vwnd = ncv.variables['v'][:,myheight,:,:] #7 for model, 14 for era: 200hPa
  ncv.close()

  uwnd, uwnd_info = prep_data(uwnd, 'tyx')
  vwnd, vwnd_info = prep_data(vwnd, 'tyx')
  lats, uwnd, vwnd = order_latdim(lats, uwnd, vwnd)

  return (ncu,uwnd,lons,lats,ncv,vwnd,uwnd_info,vwnd_info)
#-----------------END FUNCTION 1-----------------


#-----------------BEGIN FUNCTION 2-----------------
#Windspharm
def windsfarm():
  #Prepare wund for vector
  w = VectorWind(uwnd, vwnd)

  #Calculating basic variables
  eta = w.absolutevorticity()
  div = w.divergence()
  uchi, vchi = w.irrotationalcomponent()
  etax, etay = w.gradient(eta)
  planvor = w.planetaryvorticity

  S = -eta * div - uchi * etax - vchi * etay
  S = recover_data(S, uwnd_info)

  return(w, eta, div, uchi, vchi, etax, etay, planvor, S)
#-----------------END FUNCTION 2-----------------


#-----------------BEGIN FUNCTION 2b-----------------
def gaussfilter():
  #scipy.ndimage.filters.gaussian_filter(uwnd[:,:,months],)
  uwnd_filtered_lat = np.zeros(((len(uwnd),len(uwnd[0]),len(uwnd[0,0]))))
  etay_filtered_lat = np.zeros(((len(uwnd),len(uwnd[0]),len(uwnd[0,0]))))
  uwnd_filtered = np.zeros(((len(uwnd),len(uwnd[0]),len(uwnd[0,0]))))
  etay_filtered = np.zeros(((len(uwnd),len(uwnd[0]),len(uwnd[0,0]))))


  for i in range(len(uwnd)):
    for j in range(len(uwnd[0,0])):
      uwnd_filtered_lat[i,:,j] = scipy.ndimage.filters.gaussian_filter(uwnd[i,:,j],0.5*8.0*np.cos((lats[i]/360.0)*2*np.pi),0,mode='wrap') #The 0.5 is for 2 sigma
      etay_filtered_lat[i,:,j] = scipy.ndimage.filters.gaussian_filter(etay[i,:,j],0.5*8.0*np.cos((lats[i]/360.0)*2*np.pi),0,mode='wrap')
  
  for k in range(len(uwnd[0])):
    for j in range(len(uwnd[0,0])):
      uwnd_filtered[:,k,j] = scipy.ndimage.filters.gaussian_filter(uwnd_filtered_lat[:,k,j],0.5*8.0,0,mode='wrap') #The 0.5 is for 2 sigma
      etay_filtered[:,k,j] = scipy.ndimage.filters.gaussian_filter(etay_filtered_lat[:,k,j],0.5*8.0,0,mode='wrap')

  del(uwnd_filtered_lat)
  del(etay_filtered_lat)

  return (uwnd_filtered,etay_filtered)

#-----------------END FUNCTION 2b-----------------


#-----------------BEGIN FUNCTION 3-----------------
def raytracing():
  store_lat = []
  store_lon = []
  index_lat = []
  index_lon = []
  xpt_array = []
  ypt_array = []
  store_lat.append(start_lat)
  store_lon.append(start_lon)
  index_lat.append(np.abs(lats - store_lat[-1]).argmin())
  index_lon.append(np.abs(lons - store_lon[-1]).argmin())
  #k = 0
  #n = 1
  A = 1.0 #This controls the +/- inside the root
  B = 1.0 #Reflecting the y if -ve, polewards if +ve
  D = False #This stops reflection when we are still inside reflection zone.
  E_in = 55 #30 #60 #This is the better version of F_kill. We only allow reflection is E >= E_in. This depends on the step size. If step_size is 5000, the default is 180. This stops local structures from causing to reflect twice within a small area/inside the reflection zone. However, This should only be a supplement to another version: No reflection until it leaves the reflection zone.
  E = E_in #This just sets E to E_in as a initial value, this allows reflection from the start.
  F = 0 #This is the inital value of F. If F >= F_kill, we kill that ray.
  F_kill = 35 #70 #This controls a emergency kill if the ray has been `stuck' in the reflection zone for too long. It is not ideal as we don't want any rays to even be stuck in the reflection zone in the first place. The default is 800 which effectively switches this off. I don't want to use this unless I have to.
  myiteration = 300 #1800 #This controls the number of steps to take before the program ends.
  step_size = 25000.0 #Controls how long the step size is. The units are in terms of the x, y coord that basemap uses. Default is 5000 but too small steps may cause it to see the kinks in the reflective surface. If insist on using small step_size then we need to regrid to a finer scale. Note that the actual size changes as this only controls movement in x and thus depends on angle.

  for i in range(0,myiteration):
    xpt, ypt = m(store_lon[-1],store_lat[-1])
    #print('hi ' + str(store_lat[-1]) + ' ' + str(np.absolute(store_lat[-1]*(np.pi/180.0))) + ' ' + str(np.cos(np.absolute(store_lat[-1]*(np.pi/180.0)))))
    #print(uwnd[index_lat[-1],index_lon[-1],month])
    if uwnd[index_lat[-1],index_lon[-1],month] <= 0.0: #0.01
      print('u=0, ' + str(start_lat) + ', ' + str(start_lon))
      #print(uwnd[index_lat[-1],index_lon[-1],month])
      break
    else:
      pass
    if (float(etay[index_lat[-1],index_lon[-1],month])/float(uwnd[index_lat[-1],index_lon[-1],month]))-((3/(6371000.0*2.0*np.pi*np.cos(np.absolute(store_lat[-1]*(np.pi/180.0)))))*(3/(6371000.0*2.0*np.pi*np.cos(np.absolute(store_lat[-1]*(np.pi/180.0)))))) <= 0 and D == False and E > E_in:
      #print(i)
      A = -A
      B = -B
      D = True
      E = 1
    elif (float(etay[index_lat[-1],index_lon[-1],month])/float(uwnd[index_lat[-1],index_lon[-1],month]))-((3/(6371000.0*2.0*np.pi*np.cos(np.absolute(store_lat[-1]*(np.pi/180.0)))))*(3/(6371000.0*2.0*np.pi*np.cos(np.absolute(store_lat[-1]*(np.pi/180.0)))))) <= 0 and D == False and E <= E_in:
      A = -A
      D = True
    elif (float(etay[index_lat[-1],index_lon[-1],month])/float(uwnd[index_lat[-1],index_lon[-1],month]))-((3/(6371000.0*2.0*np.pi*np.cos(np.absolute(store_lat[-1]*(np.pi/180.0)))))*(3/(6371000.0*2.0*np.pi*np.cos(np.absolute(store_lat[-1]*(np.pi/180.0)))))) <= 0 and D == True:
      A = -A
    else:
      D = False
    #print((float(etay[index_lat[-1],index_lon[-1],month])/float(uwnd[index_lat[-1],index_lon[-1],month]))-((3/(6371000.0*2.0*np.pi*np.cos(np.absolute(store_lat[-1]*(np.pi/180.0)))))*(3/(6371000.0*2.0*np.pi*np.cos(np.absolute(store_lat[-1]*(np.pi/180.0)))))))
    C = np.sqrt(A*((float(etay[index_lat[-1],index_lon[-1],month])/float(uwnd[index_lat[-1],index_lon[-1],month]))-((3/(6371000.0*2.0*np.pi*np.cos(np.absolute(store_lat[-1]*(np.pi/180.0)))))*(3/(6371000.0*2.0*np.pi*np.cos(np.absolute(store_lat[-1]*(np.pi/180.0))))))))/(3/(6371000.0*2.0*np.pi*np.cos(np.absolute(store_lat[-1]*(np.pi/180.0)))))
    if float(etay[index_lat[-1],index_lon[-1],month])/float(uwnd[index_lat[-1],index_lon[-1],month])-((3/(6371000.0*2.0*np.pi*np.cos(np.absolute(store_lat[-1]*(np.pi/180.0)))))*(3/(6371000.0*2.0*np.pi*np.cos(np.absolute(store_lat[-1]*(np.pi/180.0)))))) < 0.0 and float(uwnd[index_lat[-1],index_lon[-1],month]) > 0.0:
      F += 1
    elif float(etay[index_lat[-1],index_lon[-1],month])/float(uwnd[index_lat[-1],index_lon[-1],month])-((3/(6371000.0*2.0*np.pi*np.cos(np.absolute(store_lat[-1]*(np.pi/180.0)))))*(3/(6371000.0*2.0*np.pi*np.cos(np.absolute(store_lat[-1]*(np.pi/180.0)))))) >= 0.0 and float(uwnd[index_lat[-1],index_lon[-1],month]) > 0.0:
      F = 0
    if F >= F_kill:
      print('bad reflection, ' + str(start_lat) + ', ' + str(start_lon))
      #print(uwnd[index_lat[-1],index_lon[-1],month])
      break
    xpt = xpt + step_size
    ypt = ypt - B*C*step_size
    lonpt, latpt = m(xpt,ypt,inverse=True)
    A = 1.0
    E += 1
    if lonpt >= 360.0:
      lonpt = lonpt - 360.0
    elif latpt <= -89.9:
      print('lat=-90, ' + str(start_lat) + ', ' + str(start_lon))
      #print(uwnd[index_lat[-1],index_lon[-1],month])
      break
    elif latpt > 0.1:
      print('lat=+0.0, now in the NH ' + str(start_lat) + ', ' + str(start_lon))
      break
    else:
      pass
    #print(C)
    latidx = np.abs(lats - latpt).argmin()
    lonidx = np.abs(lons - lonpt).argmin()
    index_lat.append(latidx)
    index_lon.append(lonidx)
    store_lat.append(latpt)
    store_lon.append(lonpt)
    xpt_array.append(xpt)
    ypt_array.append(ypt)
    #k += 1

  return (store_lon, store_lat)
#-----------------END FUNCTION 3-----------------


#-----------------BEGIN FUNCTION 4-----------------
def checkc():
  see = np.zeros((len(uwnd),len(uwnd[0])))

  for i in range(len(uwnd)):
    for j in range(len(uwnd[0])):
      #see[i,j] = (float(etay[i,j,month])/float(uwnd[i,j,month]))-((3/(6371000.0*2.0*np.pi*np.cos(0)))*(3/(6371000.0*2.0*np.pi*np.cos(0))))
      #see[i,j] = float(etay[i,j,month])
      #'''
      if float(uwnd[i,j,month]) <= 0.0:
        see[i,j] = 1.0
      elif float(etay[i,j,month])/float(uwnd[i,j,month])-((3/(6371000.0*2.0*np.pi*np.cos((np.pi/180.0)*np.absolute(lats[i]))))*(3/(6371000.0*2.0*np.pi*np.cos((np.pi/180.0)*np.absolute(lats[i]))))) >= 0.0 and float(uwnd[i,j,month]) > 0.0:
        see[i,j] = np.sqrt(float(etay[i,j,month])/float(uwnd[i,j,month]))
      elif float(etay[i,j,month])/float(uwnd[i,j,month])-((3/(6371000.0*2.0*np.pi*np.cos((np.pi/180.0)*np.absolute(lats[i]))))*(3/(6371000.0*2.0*np.pi*np.cos((np.pi/180.0)*np.absolute(lats[i]))))) < 0.0 and float(uwnd[i,j,month]) > 0.0:
        see[i,j] = -1.0
      else:
        print('Extra condition')
        exit()
      #'''

  #see = see/np.mean(see)

  '''
  for i in range(len(see)):
    for j in range(len(see[0])):
      if see[i,j] > 0.0:
        see[i,j] = 1.0
      else:
        see[i,j] = -1.0
  '''

  return see
#-----------------END FUNCTION 4-----------------




#-----------------BEGIN FUNCTION 1-----------------
def seasonname():
  if month == 0:
    season_name = 'djf'
  elif month == 1:
    season_name = 'mam'
  elif month == 2:
    season_name = 'jja'
  elif month == 3:
    season_name = 'son'
  else:
    print('Error, fifth season type')
    exit()

  return season_name
#-----------------END FUNCTION 1-----------------
#----------------------------------END FUNCTIONS------------------------------------------------




#----------------------------------BEGIN BODY---------------------------------------------------
#-----------------initial conditions-----------------
myjob = sys.argv[2]
myheight = int(sys.argv[3])
fig=plt.figure(figsize=(18,10))
ax = fig.add_axes([0.10,0.10,0.8,0.8]) #Cannot see title with 0.07, 0.05, 0.9, 0.95
m = Basemap(projection='mill', resolution='c', llcrnrlon=0, llcrnrlat=-90,urcrnrlon=360.01, urcrnrlat=45)
start_lat_base = -35.0
#start_lon_base = 220.0
start_lon_base = float(sys.argv[4])
month = int(sys.argv[1])
#How many lines do we draw (pxq)
p = 5
q = 10
lon_long = []
lat_long = []
#-----------------CALLING FUNCTION 1-----------------
ncu,uwnd,lons,lats,ncv,vwnd,uwnd_info,vwnd_info = readin();
#-----------------END CALLING FUNCTION 1-----------------


#-----------------CALLING FUNCTION 2-----------------
w, eta, div, uchi, vchi, etax, etay, planvor, S = windsfarm();
#-----------------END CALLING FUNCTION 2-----------------


#-----------------CALLING FUNCTION 2b-----------------
uwnd_filtered,etay_filtered = gaussfilter();
del(uwnd)
del(etay)
uwnd = uwnd_filtered
etay = etay_filtered
#-----------------END CALLING FUNCTION 2b-----------------


#-----------------CALLING FUNCTION 3-----------------
for i in range(0,p):
  for j in range(0,q):
    start_lat = start_lat_base + 2*i
    start_lon = start_lon_base + 5*j
    store_lon, store_lat = raytracing();
    lon_long.append(store_lon)
    lat_long.append(store_lat)
    del(store_lon)
    del(store_lat)
    del(start_lat)
    del(start_lon)
print(np.shape(lon_long[4]))
#-----------------END CALLING FUNCTION 3-----------------


see = checkc();

see, lons_c = addcyclic(see, lons)
x, y = m(*np.meshgrid(lons_c, lats))
#clevs = [-2E-11, -1E-11, 0.0, 1E-11, 2E-11, 3E-11, 4E-11, 5E-11, 6E-11, 7E-11]
clevs = [-1.0,-0.1,0.0,0.25E-6,0.5E-6,0.75E-6,1E-6,1.25E-6,1.5E-6,1.75E-6,2E-6,2.25E-6,2.5E-6,2.75E-6,3E-6,3.25E-6,3.5E-6,3.75E-6,4E-6,0.1,1.0]

#orig_cmap = plt.cm.spring
orig_cmap = plt.cm.seismic 
cs = m.contourf(x, y, see, clevs, cmap=orig_cmap) #,extend='none')
m.drawcoastlines()

m.drawparallels((-90, -60, -30, 0, 30, 60, 90), labels=[1,0,0,0], fontsize=35)
m.drawmeridians((0, 60, 120, 180, 240, 300, 360), labels=[0,0,0,1], fontsize=35)
ax.tick_params(axis='both', labelsize=35)

#plt.colorbar(orientation='horizontal')
#plt.show()
#exit()
#cbar = m.colorbar(cs,pad="5%",location='bottom')
#cbar.set_label(cbarunits, rotation=0)

S_dec, lons_c = addcyclic(S[month], lons)

#m = Basemap(projection='cyl', resolution='c', llcrnrlon=180, llcrnrlat=-90,urcrnrlon=330.01, urcrnrlat=90)
#x, y = m(*np.meshgrid(lons_c, lats))
#clevs = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35]
#m.contourf(x, y, S_dec*1e11, clevs, cmap=plt.cm.RdBu_r,extend='both')
#m.drawcoastlines()
#m.drawparallels((-90, -60, -30, 0, 30, 60, 90), labels=[1,0,0,0])
#m.drawmeridians((0, 60, 120, 180, 240, 300, 360), labels=[0,0,0,1])
#plt.colorbar(orientation='horizontal')
#plt.title('Rossby Wave Source ($10^{-11}$s$^{-1}$)', fontsize=35)

for i in range(0,p):
  for j in range(0,q):
    print(len(lon_long[i*q+j]))
    x,y = m(lon_long[i*q+j], lat_long[i*q+j])
    #m.plot(x, y, 'bo', markersize=1)
    lon_long_2 = np.array(lon_long[i*q+j])
    if len(lon_long) <= 1.0:
      pass
    else:
      index = np.abs(lon_long_2 - 360.0).argmin()
      #print(index)
      if np.abs(lon_long_2[index] - 360.0) < 1.0:
        m.plot(x[0:index+1], y[0:index+1], linewidth=0.5,color='r')
        m.plot(x[index+1:], y[index+1:], linewidth=0.5,color='r')
      elif np.abs(lon_long_2[index] - 360.0) >= 1.0:
        m.plot(x, y, linewidth=0.5,color='r')
      else:
        print('Error, extra condition')
        exit()
    del(lon_long_2)
    del(x)
    del(y)
    del(index)


def draw_screen_poly( lats, lons, m):
    x, y = m( lons, lats )
    xy = zip(x,y)
    poly = Polygon( xy, facecolor='k', alpha=0.4 )
    plt.gca().add_patch(poly)

lats = [ -60, -60, -75, -75 ]
lons = [ 170, 290, 290, 170 ]

draw_screen_poly( lats, lons, m )

def draw_screen_poly( lats, lons, m):
    x, y = m( lons, lats )
    xy = zip(x,y)
    poly = Polygon( xy, facecolor='k', alpha=0.4 )
    plt.gca().add_patch(poly)

lats = [ 5, 5, -5, -5 ]
lons = [ 190, 240, 240, 190 ]

draw_screen_poly( lats, lons, m )

#-----------------CALLING FUNCTION 5-----------------
season_name = seasonname();
#-----------------END CALLING FUNCTION 5-----------------
if myheight == 7:
  height_name = '200hPa'
elif myheight == 8:
  height_name = '170hPa'
elif myheight == 6:
  height_name = '250hPa'
elif myheight == 0:
  height_name = '1000hPa'
else:
  height_name = 'Error'

if myjob == 'xlnjj':
  job_name = '-3.0K'
elif myjob == 'xlnji':
  job_name = '-2.25K'
elif myjob == 'xlnjh':
  job_name = '-1.5K'
elif myjob == 'xlnjg':
  job_name = '-0.75K'
elif myjob == 'xlnja':
  job_name = '0.0K'
elif myjob == 'xlnjc':
  job_name = '0.75K'
elif myjob == 'xlnjd':
  job_name = '1.5K'
elif myjob == 'xlnje':
  job_name = '2.25K'
elif myjob == 'xlnjf':
  job_name = '3.0K'
else:
  job_name = 'Error'

plottitle = height_name + ' ray tracing: ' + str.upper(season_name) +  ' ' + job_name
plt.title(plottitle, fontsize=50, y=1.03)
filename = 'ray_' + season_name +  '_' + myjob + '_height' + str(myheight) + '.png'
#plt.show()
plt.savefig(filename)
exit()

