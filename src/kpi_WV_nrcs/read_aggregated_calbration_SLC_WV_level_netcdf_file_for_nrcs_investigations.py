import os
import xarray
import sys
import time
import numpy as np
import netCDF4
import logging
from datetime import datetime
import reader_nasa_gsfc_distance_to_coast_super_light
import compute_wind_azimuth
import glob
t0 = time.time()
from gmf_cmod5n import GMFCmod5n
#first test to look at the content of the daily files SAFE containing the noise and denoised sigma0
DIR_INPUT = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/L1_v16/WV'
INPUT_files = os.path.join(DIR_INPUT,'%s_wv_ocean_calibration_CMOD5n_ecmwf0125_windspeed_weighted_slc_level1_20150101_today_runv16.nc')

def read_fat_calib_nc(satellite_list=['S1A','S1B']):
    """
    read the fat netcdf files for NRCS investigations build from ocean_wv_calibration_huimin_method.py
    :return:
    """
    df_slc_sat = {}
    for satellite0 in satellite_list :
        pattern = INPUT_files%satellite0
        logging.info('pattern %s',pattern)
        output_file_calibration = glob.glob(pattern)[0]
        logging.info('found %s',output_file_calibration)
        # logging.debug('is same file %s',output_file_calibration == testf)
        if os.path.exists(output_file_calibration) :
            logging.info('read file %s',output_file_calibration)
            nctest = netCDF4.Dataset(output_file_calibration)
            logging.debug('var kurt_sar %s','kurt_sar' in nctest.variables.keys())
            logging.debug('var skew_sar %s','skew_sar' in nctest.variables.keys())
            nctest.close()
            xd = xarray.open_dataset(output_file_calibration)
            df_slc = xd.to_dataframe()
            df_slc['time'] = xd['time'].values
            df_slc.index = df_slc['time']
            df_slc.drop_duplicates(subset=['time'],inplace=True)
            #drop NaN latitudes
            logging.info('before lat filter %s',len(df_slc))
            df_slc = df_slc.dropna(subset=['_lat_sar'])
            df_slc = df_slc[(df_slc['_lat_sar']>-90 ) & (df_slc['_lat_sar']<90 )]
            logging.info('after lat filter %s finite %s %s',len(df_slc),np.isfinite(df_slc['_lat_sar']).sum(),np.isfinite(df_slc['_lon_sar']).sum())
            # print(df_slc)
            df_slc.dropna(subset=['time'],inplace=True)
            df_slc['_sig_sar_db'] = 10. * np.log10(df_slc['_sig_sar'])
            df_slc['sigma0_denoised_db'] = 10. * np.log10(df_slc['sigma0_denoised'])

            t0 = time.time()
            dst = reader_nasa_gsfc_distance_to_coast_super_light.get_distance_to_coast_vecto(df_slc['_lon_sar'].values,
                                                                                             df_slc['_lat_sar'].values)
            logging.info('elapsed time to have the data reade %1.1f seconds' % (time.time() - t0))
            df_slc['distance2coast'] = dst
            # add the windazi
            test_trackangle = (df_slc['_tra_sar']) % 360
            azi_sar_test = compute_wind_azimuth.method_wind_azi_range(df_slc['_zon_coloc_ecmwf'],df_slc['_mer_coloc_ecmwf'],
                                                                      test_trackangle)
            df_slc['_wind_azimuth_ecmwf'] = azi_sar_test


            gmf = GMFCmod5n()
            sigma0_cmod5n = gmf._getNRCS(df_slc['_inc_sar'].values,df_slc['_spd_coloc_ecmwf'].values,azi_sar_test)
            df_slc['tmp_gmf_cmod5n_nrcs'] = sigma0_cmod5n
            df_slc['tmp_gmf_cmod5n_nrcs_db'] = 10. * np.log10(sigma0_cmod5n)

            # fix ecmwf wind directions
            _dirtest2 = np.mod(np.degrees(np.arctan2(df_slc['_zon_coloc_ecmwf'],df_slc['_mer_coloc_ecmwf'])),360.)
            df_slc['_dir_coloc_ecmwf'] = _dirtest2
            # define nosie in db
            df_slc['noise_db'] = 10.0 * np.log10(df_slc['noise'])
            # save the dataframe enriched in dict
            logging.info('%s df enriched ok',satellite0)
            df_slc_sat[satellite0] = df_slc

        else :
            print('no %s' % output_file_calibration)
    return df_slc_sat


if __name__ =='__main__':
    logging.basicConfig(level=logging.INFO)
    start_date = datetime(2015,1,1)
    stop_date = datetime(2019,12,1)
    stop_date = datetime(2020,1,9)

    # stop_date= datetime(2019,12,31)
    # start_date = datetime(2019,2,1)
    # stop_date= datetime(2019,3,20)
    # start_date = datetime(2016,1,1)
    # start_date = datetime(2019,8,27)
    # stop_date= datetime(2018,2,28)
    # start_date = datetime(2019,1,1)
    # stop_date= datetime(2019,10,10)
    level = 'L1'
    satellite = 'S1A'
    subdir = 'v11'  # ecmwf0125 with corrcted wind direction computation
    subdir = 'v12'  # correction noise sur IPF2.91 + dlon et dlon coloc ecmwf
    subdir = 'v14'  # add roughness classification
    # subdir = 'v15' # kurt + skew encmwf horaire
    subdir = 'v16'  # new noise vectors cst correction from Pauline
    if subdir == 'v15' :
        start_date = datetime(2019,8,21)  # v15
        stop_date = datetime(2020,2,20)
    if subdir == 'v16' :
        stop_date = datetime(2020,3,27)

    # sta_str = start_date.strftime('%Y%m%d')
    # sto_str = stop_date.strftime('%Y%m%d')
    df_slc_sat = read_fat_calib_nc(['S1A'])