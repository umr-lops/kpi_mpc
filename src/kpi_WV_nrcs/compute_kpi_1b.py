# -*- coding: utf-8 -*-
"""
IFREMER
Oct 2021
KPI-1b SLA Vol3 document MPC contract 2021 - 2016
"""
import os
import logging
import numpy as np
import datetime
import time
from read_aggregated_calbration_SLC_WV_level_netcdf_file_for_nrcs_investigations import read_fat_calib_nc

POLARIZATION = 'VV'
MODE = 'WV'
#ENVELOP = 2 #sigma
ENVELOP = 95 # %
PRIOR_PERIOD = 3 #months
LAT_MAX = 55
MIN_DIST_2_COAST = 100 #km
def compute_kpi_1b(sat,wv,stop_analysis_period=None,df_slc_sat=None):
    """
    NRCS (denoised) observed compared to predicted GMF CMOD5n
    :param sat: str S1A or ..
    :param wv: str wv1 or wv2
    :param stop_analysis_period: datetime (-> period considered date-1 month : date)
    :return:
        kpi_value (float): between 0 and 100 %
        start_current_month (datetime):
        stop_current_month (datetime):
        envelop_value : (float) 2-sigma dB threshold based on 3 months prior period
    """
    if df_slc_sat is None:
        df_slc_sat = read_fat_calib_nc(satellite_list=[sat])
    if stop_analysis_period is None:
        stop_current_month = datetime.datetime.today()
    else:
        stop_current_month = stop_analysis_period
    start_current_month = stop_current_month -  datetime.timedelta(days=30)
    logging.debug('start_current_month : %s',start_current_month)
    logging.debug('stop_current_month : %s',stop_current_month)
    #compute the 2 sigma envelopp on the last 3 months prior to current month
    start_prior_period = start_current_month -  datetime.timedelta(days=30*PRIOR_PERIOD)
    stop_prior_period = start_current_month
    df_slc = df_slc_sat[sat]
    df_slc['direct_diff_calib_cst_db'] = df_slc['sigma0_denoised_db'] - df_slc['tmp_gmf_cmod5n_nrcs_db']
    if wv=='wv1':
        cond_inc = (df_slc['_inc_sar']<30)
    elif wv=='wv2':
        cond_inc = (df_slc['_inc_sar'] > 30)
    else:
        raise Exception('wv value un expected : %s'%wv)
    ocean_acqui_filters = (abs(df_slc['_lat_sar']) < LAT_MAX) & (df_slc['_land'] == False) & (df_slc['distance2coast'] > MIN_DIST_2_COAST)
    mask_prior_period = ocean_acqui_filters & cond_inc & (df_slc['time']>=start_prior_period) & (df_slc['time']<=stop_prior_period) & (np.isfinite(df_slc['direct_diff_calib_cst_db']))
    subset_df = df_slc[mask_prior_period]
    nb_nan = np.isnan(subset_df['direct_diff_calib_cst_db']).sum()
    logging.debug('some values: %s',subset_df['direct_diff_calib_cst_db'].values)
    logging.info('nb_nan : %s',nb_nan)
    logging.info('nb finite %s/%s',np.isfinite(subset_df['direct_diff_calib_cst_db']).sum(),len(subset_df['direct_diff_calib_cst_db']))
    #envelop_value = ENVELOP*np.nanstd(subset_df['direct_diff_calib_cst_db'])
    std = np.nanstd(subset_df['direct_diff_calib_cst_db'])
    envelop_value = np.percentile(abs(subset_df['direct_diff_calib_cst_db']),ENVELOP)
    logging.debug('envelop_value : %s',envelop_value)


    #compute the number of product within the envelop for current month
    nb_measu_total = 0
    nb_measu_inside_envelop = 0
    mask_current_month = ocean_acqui_filters & cond_inc & (df_slc['time']>=start_current_month) & (df_slc['time']<=stop_current_month)
    subset_current_period = df_slc[mask_current_month]
    logging.info('nb pts current month : %s',len(subset_current_period['time']))
    nb_measu_total = len(subset_current_period['time'])
    #nb_measu_outside_envelop = (abs(subset_current_period['direct_diff_calib_cst_db'])<envelop_value).sum()

    # definition proposee par Hajduch le 10dec2021 screenshot a lappuit (je ne suis pas convaincu pas l introduction du biais dans le calcul de levenveloppe car le KPI sera dautant plus elever que le biais sera fort (cest linverse qui est cherche)
    bias_minus_2sigma = abs(subset_current_period['direct_diff_calib_cst_db'].mean() - envelop_value)
    bias_plus_2sigma = abs(subset_current_period['direct_diff_calib_cst_db'].mean() + envelop_value)
    logging.info('bias_plus_2sigma %s', bias_plus_2sigma)
    T = np.max([bias_minus_2sigma, bias_plus_2sigma])
    logging.info('T : %s %s', T.shape, T)
    nb_measu_inside_envelop = (abs(subset_current_period['direct_diff_calib_cst_db']) < T).sum()
    mean_bias = np.mean(subset_current_period['direct_diff_calib_cst_db'])
    logging.debug('nb_measu_inside_envelop : %s', nb_measu_inside_envelop)
    kpi_value = 100. * nb_measu_inside_envelop / nb_measu_total

    logging.debug('kpi_value : %s',kpi_value)
    return kpi_value,start_current_month,stop_current_month,envelop_value,nb_measu_total,nb_measu_inside_envelop,mean_bias,std

if __name__ == '__main__':
    root = logging.getLogger ()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler (handler)
    import argparse
    import resource
    time.sleep(np.random.rand(1,1)[0][0]) #to avoid issue with mkdir
    parser = argparse.ArgumentParser (description='kpi-1b')
    parser.add_argument ('--verbose',action='store_true',default=False)
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='overwrite the existing outputs [default=False]', required=False)
    parser.add_argument('--satellite',choices=['S1A','S1B'],required=True,help='S-1 unit choice')
    parser.add_argument('--wv',choices=['wv1','wv2'],required=True,help='WV incidence angle choice')
    parser.add_argument('--enddate',help='end of the 1 month period analysed',required=False,action='store',default=None)
    args = parser.parse_args ()

    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'

    if args.verbose :
        logging.basicConfig(level=logging.DEBUG,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    else :
        logging.basicConfig(level=logging.INFO,format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    t0 = time.time ()
    sat = args.satellite
    wv = args.wv
    if args.enddate is not None:
        end_date = datetime.datetime.strptime(args.enddate,'%Y%m%d')
    else:
        end_date = args.enddate # None case
    # output_file = '/home1/scratch/agrouaze/kpi_1b/%s/kpi_output_%s_%s_%s.txt' % ('v2percentile95',
    #                                                                                 sat, wv,
    #                                                                                 end_date.strftime('%Y%m%d'))
    output_file = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/kpi/kpi_1b/%s/kpi_output_%s_%s_%s.txt' % ('v2percentile95',
                                                                                    sat, wv,
                                                                                    end_date.strftime('%Y%m%d'))
    if os.path.exists(output_file) and args.overwrite is False:
        logging.info('output %s already exists', output_file)
    else:


        kpi_v,start_cur_month,stop_cur_month,envelop_val,nb_measu_total,nb_measu_inside_envelop,mean_bias,std =\
            compute_kpi_1b(sat,wv=wv,stop_analysis_period=end_date)
        logging.info('##########')
        logging.info('kpi_v %s %s :  %s (envelop %s-sigma value: %s dB)',sat,wv,kpi_v,ENVELOP,envelop_val)
        logging.info('##########')
        logging.info('start_cur_month : %s stop_cur_month : %s',start_cur_month,stop_cur_month)

        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file), 0o0775)
        fid = open(output_file, 'w')
        fid.write('%s %s %s %s %s %s %s\n' % (kpi_v, start_cur_month, stop_cur_month, envelop_val, nb_measu_total, mean_bias,std))
        fid.close()
        logging.info('output: %s', output_file)


        logging.info('done in %1.3f min',(time.time()-t0)/60.)
        logging.info('peak memory usage: %s Mbytes',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.)
    logging.info("over")