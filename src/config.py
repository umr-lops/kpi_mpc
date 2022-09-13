import os
RASTER_NASA_COASTLINE = '/home/datawork-lops-siam/project/cfosat_calval_wwec/data/colocations/NASA_tiff_distance_to_coast_converted_v2.nc'
DIR_INPUT = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/L1_v16/WV'
INPUT_files = os.path.join(DIR_INPUT,'%s_wv_ocean_calibration_CMOD5n_ecmwf0125_windspeed_weighted_slc_level1_20150101_today_runv16.nc')
EXTRACTION_SERIES_L2F_FOR_LONGTERM_MONITORING_L2 = '/home/datawork-cersat-public/project/mpc-sentinel1/analysis/s1_data_analysis/longterm_extraction_L2F/'
OUTPUTDIR_KPI_1D = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/kpi/kpi_1d/v2'
OUTPUTDIR_KPI_1B = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/kpi/kpi_1b/v2'
VERSION_L2F = '0.8' #add cci hs NN Quach 2020 in v3.2 format + add 10 classificaitons Chen + probabilities + wind speed SAR + fix missing obs at begining of each days
PROJECT_DIR_DATARMOR = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/'
DIR_ANALYSIS = os.path.join(PROJECT_DIR_DATARMOR,"analysis","s1_data_analysis")
DIR_L2F_WV_DAILY = os.path.join(DIR_ANALYSIS,'L2_full_daily',VERSION_L2F)
