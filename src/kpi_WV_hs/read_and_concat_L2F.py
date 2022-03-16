# encoding -utf8-
"""
read and concatenate daily files variables into a xarray Datatset
"""
import time
import xarray
import glob
import logging
import datetime
import numpy as np
import os
import resource
from src.config import DIR_L2F_WV_DAILY


def preprocess_L2F(ds, variables=None, add_ecmwf_wind=True):
    """
    
    :param ds:
    :param variables:
    :return:
    """
    filee = ds.encoding['source']
    # logging.info(os.path.basename(filee))
    ds = ds.sortby('fdatedt')
    if variables is not None:
        ds = ds[variables]
        for kk in ds.keys():
            logging.debug('test kk : %s', kk)
            if 'dataset' in ds[kk].dims:
                logging.debug('split %s in S1 + WW3 vars')
                ds[kk + '_ww3'] = xarray.DataArray(ds[kk][:, 1].values,
                                                   dims=['fdatedt'])  # checked in partition_wv_xassignement.py
                ds[kk + '_s1'] = xarray.DataArray(ds[kk][:, 0].values, dims=['fdatedt'])
    if 'ecmwf_wind_speed' not in ds and add_ecmwf_wind:
        tmpval = np.sqrt(ds['ecmwf0125_uwind'].values ** 2. + ds['ecmwf0125_vwind'].values ** 2.)
        ds['ecmwf_wind_speed'] = xarray.DataArray(tmpval, dims=['fdatedt'])
    if 'pol' in ds:
        ds['pol'] = ds['pol'].astype(str)
    #     #logging.info('pol dtype %s',ds['pol'].dtype)
    # # if 'class_1' in ds:
    # #     logging.info('class_1 dtype %s',ds['class_1'].dtype)
    # #     logging.info('class_1 val %s',ds['class_1'].values)
    # #     ds['class_1'] = ds['class_1'].astype(str)
    # for vv in ds:
    #     if ds[vv].dtype!='float32':
    #         logging.info('%s %s is %s',os.path.basename(filee),vv,ds[vv].dtype)
    # #     if ds[vv].dtype=='byte':
    # #         logging.info('%s is byte')
    # #     elif ds[vv].dtype=='str':
    # #         logging.info('%s is str')
    # #     logging.info('vv : %s dtype %s',vv,ds[vv].dtype)
    # #for dd in ds.dims:
#    if 'dataset' in ds.dims:
    ds = ds.assign_coords({'dataset': ['sar', 'ww3']})  # to fix a S1B 20210223 file...
    # logging.info('ds.dims  %s ',ds.dims)
    # logging.info('ds %s',ds)
    return ds


def read_L2F_with_xarray(start, stop, satellites=['S1A', 'S1B'], variables=None, alternative_L2F_path=None,
                         add_ecmwf_wind=True):
    """

    :param start:
    :param stop:
    :param satellites:
    :param variables:
    :param alternative_L2F_path:
    :param add_ecmwf_wind :bool
    :return:
    """
    if isinstance(start, datetime.date):
        start = datetime.datetime(start.year, start.month, start.day)
    if isinstance(stop, datetime.date):
        stop = datetime.datetime(stop.year, stop.month, stop.day)
    logging.info('Sentinel-1 L2F lecture between %s and %s', start, stop)
    ds_dict_sat = {}
    for satr in satellites:
        ds_dict_sat[satr] = {}
    for sensor in satellites:  # pas de S1B pour le moment car pas de fichier avec les varaibles cross assigned 28 janvier 2020
        if alternative_L2F_path is None:
            datadir = DIR_L2F_WV_DAILY
        else:
            datadir = alternative_L2F_path
        if start.year == stop.year:
            pat = os.path.join(datadir, start.strftime('%Y'), '*', sensor + '*nc')
        else:
            pat = os.path.join(datadir, '*', '*', sensor + '*nc')
        logging.info('pattern to search for L2F =%s', pat)
        listnc0 = sorted(glob.glob(pat))[::-1]
        listnc = []
        dates = []
        for ff in listnc0:
            datdt = datetime.datetime.strptime(os.path.basename(ff).split('_')[5], '%Y%m%d')
            if datdt >= start and datdt <= stop and datdt not in dates:
                listnc.append(ff)
                dates.append(datdt)
        logging.info("nb files found without date filter = %s", len(listnc0))
        logging.info("nb files found with date filter = %s", len(listnc))
        for kk in listnc:
            logging.debug(kk)
        if len(listnc) > 0:
            t0 = time.time()
            #logging.info('nested')
            tmpds = xarray.open_mfdataset(listnc, preprocess=lambda ds: preprocess_L2F(ds, variables, add_ecmwf_wind),
                                          combine='by_coords')#, concat_dim='fdatedt')
            ds_dict_sat[sensor] = tmpds

    return ds_dict_sat


if __name__ == '__main__':
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    import argparse

    parser = argparse.ArgumentParser(description='read and concat')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    else:
        logging.basicConfig(level=logging.INFO, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    start = datetime.datetime(2021, 7, 1)
    stop = datetime.datetime(2021, 8, 2)
    start = datetime.datetime(2021, 7, 24)
    stop = datetime.datetime(2021, 8, 9)
    print('start', start)
    logging.info('%s %s', start, stop)
    test_vars = ['oswWindSpeed', 'oswLon', 'oswLat', 'fdatedt', 'oswHeading', 'wind_speed_model', 'azimuth_angle',
                 'wind_speed_model_u',
                 'wind_speed_model_v', 'ecmwf0125_uwind', 'ecmwf0125_vwind', 'rvlNrcsGridmean', 'class_1']
    test_vars = None
    sat = ['S1A', 'S1B']
    ds_sat = read_L2F_with_xarray(start, stop, satellites=['S1A', 'S1B'], variables=None, alternative_L2F_path=None,
                         add_ecmwf_wind=True)
    logging.info(ds_sat.keys())
    logging.info('var = %s', ds_sat[sat[0]].keys())
    logging.info('nb dates : %s', len(ds_sat[sat[0]]['fdatedt']))
    logging.info('over')
    print('over')
