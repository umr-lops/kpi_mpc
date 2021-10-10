import os
RASTER_NASA_COASTLINE = '/home/datawork-lops-siam/project/cfosat_calval_wwec/data/colocations/NASA_tiff_distance_to_coast_converted_v2.nc'
DIR_INPUT = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/L1_v16/WV'
INPUT_files = os.path.join(DIR_INPUT,'%s_wv_ocean_calibration_CMOD5n_ecmwf0125_windspeed_weighted_slc_level1_20150101_today_runv16.nc')
EXTRACTION_SERIES_L2F_FOR_LONGTERM_MONITORING_L2 = '/home/datawork-cersat-public/project/mpc-sentinel1/analysis/s1_data_analysis/longterm_extraction_L2F/'