"""
Antoine Grouazel
Nov 2019
"""
import netCDF4
import numpy as np
import logging
from src.config import RASTER_NASA_COASTLINE
nc = netCDF4.Dataset(RASTER_NASA_COASTLINE)
DISTANCE_COASTs = nc.variables['distance_to_coast'][:]
LON_COASTs = nc.variables['lon'][:]
LAT_COASTs = nc.variables['lat'][:]
nc.close()


def latlon2ij ( lat,lon,shape2D,llbox ) :
    """
    convert lat,lon into i,j index
    args:
        lat (float or 1D nd.array):
        lon (float or 1D nd.array):
        shape2D (tuple): (10,20) for instance
        llbox (tuple): latmin, lonmin, latmax,lonmax
    """
    logging.debug('input lat latlon2ij | %s',lat)
    latmin,lonmin,latmax,lonmax = llbox
    if isinstance(lat,float) or isinstance(lat,int) :
        lat = np.array([lat])
    if isinstance(lon,float) or isinstance(lon,int) :
        lon = np.array([lon])
    dlon = lonmax - lonmin
    dlat = latmax - latmin
    logging.debug('dlon = %s',dlon)
    logging.debug('dlat = %s',dlat)
    logging.debug('shape2D = %s',shape2D)
    logging.debug('lat type %s %s',type(lat),lat)
    logging.debug('lat range %s %s',lat.min(),lat.max())
    logging.debug('dlat %s shapz %s',dlat,shape2D)
    logging.debug('itest %s',np.floor((lat - latmin) * shape2D[0] / dlat))
    i = np.floor((lat - latmin) * shape2D[0] / dlat).astype(
        int)  # changed May 2019 after founding a bug with B. Coatanea where indices can reach the maximum value of the shape... (agrouaze)
    j = np.floor((lon - lonmin) * shape2D[1] / dlon).astype(int)

    return i,j

def get_distance_to_coast_vecto(lons,lats):
    llbox=(LAT_COASTs[0],LON_COASTs[0],LAT_COASTs[-1],LON_COASTs[-1])
    indlat,indlon= latlon2ij(lats,lons,np.shape(DISTANCE_COASTs),llbox)
    indlat[(indlat>=DISTANCE_COASTs.shape[0])]  = DISTANCE_COASTs.shape[0]-1
    indlon[(indlon>=DISTANCE_COASTs.shape[1])] = DISTANCE_COASTs.shape[1]-1
    dsts = DISTANCE_COASTs[indlat,indlon]
    diff_lon = lons-LON_COASTs[indlon]
    diff_lat = lats-LAT_COASTs[indlat]
    return dsts