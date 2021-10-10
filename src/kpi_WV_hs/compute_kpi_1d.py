# -*- coding: utf-8 -*-
"""
IFREMER
Oct 2021
KPI-1d SLA Vol3 document MPC contract 2021 - 2016
"""
import os
import logging
import numpy as np
import datetime
import time
import xarray
from src.config import EXTRACTION_SERIES_L2F_FOR_LONGTERM_MONITORING_L2
POLARIZATION = 'VV'
MODE = 'WV'
ENVELOP = 2 #sigma
PRIOR_PERIOD = 3 #months
LAT_MAX = 55
MIN_DIST_2_COAST = 100 #km
MS = {'wv1':-6.5,'wv2':-15.} #NRCS central values dB
DS = {'wv1':3,'wv2':5} #delta
def load_Level2_series(satellite):
    """

    :param satellite: str S1A or ...
    :return:
    """
    component = 'osw'
    fpathagg = os.path.join(EXTRACTION_SERIES_L2F_FOR_LONGTERM_MONITORING_L2,
                            '%s_%s_longterm_inputdata_from_L2F_%s.nc' % (
                                component,satellite,'last365days'))  # for dev/test
    logging.info('path aggregation = %s exists %s',fpathagg,os.path.exists(fpathagg))
    df = xarray.open_dataset(fpathagg).to_dataframe()
    logging.info("%s %s ok",component,satellite)
    logging.debug('keys : %s',df.keys())
    return df

def compute_kpi_1d(sat,wv,dev=False,stop_analysis_period=None):
    """
    osw VV WV S-1 effective Hs (2D-cutoff) compared to WWIII Hs computed on same grid/same mask
    note that low freq mask is applied both on S-1 spectrum and WWIII spectrum
    :param sat: str S1A or ..
    :param wv: str wv1 or wv2
    :param stop_analysis_period: datetime (-> period considered date-1 month : date)
    :return:
        kpi_value (float): between 0 and 100 %
        start_current_month (datetime):
        stop_current_month (datetime):
        envelop_value : (float) 2-sigma m threshold based on 3 months prior period
    """
    df = load_Level2_series(satellite=sat)
    if stop_analysis_period is None :
        stop_current_month = datetime.datetime.today()
    else :
        stop_current_month = stop_analysis_period
    start_current_month = stop_current_month -  datetime.timedelta(days=30)
    logging.debug('start_current_month : %s',start_current_month)
    logging.debug('stop_current_month : %s',stop_current_month)
    #compute the 2 sigma envelopp on the last 3 months prior to current month
    start_prior_period = start_current_month -  datetime.timedelta(days=30*PRIOR_PERIOD)
    stop_prior_period = start_current_month
    logging.info('prior period ; %s to %s',start_prior_period,stop_prior_period)
    _swh_azc_mod = df['ww3_effective_2Dcutoff_hs'].values
    logging.info('nb value WW3 eff Hs above 25 m : %s',(_swh_azc_mod>25).sum())
    if (_swh_azc_mod>25).sum()>0:
        ind_bad_ww3 = np.where(_swh_azc_mod>25)[0][0]
        logging.info('a date SAR for which ww3 is extremely too high: %s -> Hs:%1.1fm',
                     df['fdatedt'][ind_bad_ww3],_swh_azc_mod[ind_bad_ww3])
    logging.info('max Hs WW3 : %s',np.nanmax(_swh_azc_mod))
    _swh_azc_s1 = df['s1_effective_hs_2Dcutoff'].values
    df['bias_swh_azc_' + wv] = _swh_azc_s1 - _swh_azc_mod
    if wv=='wv1':
        cond_inc = (df['oswIncidenceAngle']<30)
    elif wv=='wv2':
        cond_inc = (df['oswIncidenceAngle'] > 30)
    else:
        raise Exception('wv value un expected : %s'%wv)

    cond_nrcs = (np.abs(df['oswNrcs'] - MS[wv]) < DS[wv])
    #moderate wind speed conditions
    delta_ws = 2.5
    central_ws = 7
    cond_outlier_ww3_hs = (abs(df['ww3_effective_2Dcutoff_hs']) < 50) # & (abs(df['ww3_effective_2Dcutoff_hs']) > 1)
    cond_wind = (np.abs(df['ecmwf_wind_speed']-central_ws)<delta_ws)
    logging.info('nb pts after wind filter %s %1.1f%%',np.sum(cond_wind),100.0*np.sum(cond_wind)/cond_wind.size)
    logging.info('nb pts after NRCS filter %s %1.1f%%',np.sum(cond_nrcs),100.0*np.sum(cond_nrcs)/cond_nrcs.size)

    polarizationcond = (df.pol == POLARIZATION.lower())

    ocean_acqui_filters = cond_outlier_ww3_hs & cond_wind & polarizationcond & cond_nrcs & (abs(df['oswLat']) < LAT_MAX) \
                          & (df['oswLandFlag'] == 0) & (df['dist2coastKM'] > MIN_DIST_2_COAST) & cond_inc &(np.isfinite(df['bias_swh_azc_' + wv]))
    mask_prior_period =  (df['fdatedt']>=start_prior_period) \
                        & (df['fdatedt']<=stop_prior_period)
    final_mask_prior = ocean_acqui_filters
    logging.info('nb pts in prior period (without extra filters) : %s',mask_prior_period.sum())
    subset_df = df[final_mask_prior]
    nb_nan = np.isnan(subset_df['bias_swh_azc_' + wv]).sum()
    logging.debug('some values: %s',subset_df['bias_swh_azc_' + wv].values)
    logging.info('nb_nan : %s',nb_nan)
    logging.info('nb finite %s/%s',np.isfinite(subset_df['bias_swh_azc_' + wv]).sum(),len(subset_df['bias_swh_azc_' + wv]))
    envelop_value = ENVELOP*np.nanstd(subset_df['bias_swh_azc_' + wv])
    logging.debug('envelop_value : %s',envelop_value)

    #compute the number of product within the envelop for current month
    mask_current_month = ocean_acqui_filters \
                         & (df['fdatedt']>=start_current_month) & (df['fdatedt']<=stop_current_month)
    subset_current_period = df[mask_current_month]
    logging.info('max Hs WW3 in subset : %s',np.nanmax(subset_current_period['ww3_effective_2Dcutoff_hs']))
    logging.info('max Hs SAR in subset : %s',np.nanmax(subset_current_period['s1_effective_hs_2Dcutoff']))
    logging.info('min Hs WW3 in subset : %s',np.nanmin(subset_current_period['ww3_effective_2Dcutoff_hs']))
    logging.info('min Hs SAR in subset : %s',np.nanmin(subset_current_period['s1_effective_hs_2Dcutoff']))
    logging.info('nb pts current month : %s',len(subset_current_period['fdatedt']))
    nb_measu_total = len(subset_current_period['fdatedt'])
    nb_measu_outside_envelop = (abs(subset_current_period['bias_swh_azc_' + wv])<envelop_value).sum()

    kpi_value = 100.*nb_measu_outside_envelop/nb_measu_total
    logging.debug('kpi_value : %s',kpi_value)
    if dev:
        from matplotlib import pyplot as plt
        plt.figure()
        binz = np.arange(0,10,0.1)
        hh,_ = np.histogram(subset_current_period['ww3_effective_2Dcutoff_hs'],binz)
        plt.plot(binz[0:-1],hh,label='WWIII %s'%len(subset_current_period['ww3_effective_2Dcutoff_hs']))
        hh,_ = np.histogram(subset_current_period['s1_effective_hs_2Dcutoff'],binz)
        plt.plot(binz[0 :-1],hh,label='SAR %s'%len(subset_current_period['ww3_effective_2Dcutoff_hs']))
        plt.grid(True)
        plt.legend()
        plt.xlabel('Hs (m)')
        output = '/home1/scratch/agrouaze/test_histo_kpi_1d.png'
        plt.savefig(output)
        logging.info('png test : %s',output)
        #plt.show()
    return kpi_value,start_current_month,stop_current_month,envelop_value

if __name__ == '__main__':
    root = logging.getLogger ()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler (handler)
    import argparse
    import resource
    time.sleep(np.random.rand(1,1)[0][0]) #to avoid issue with mkdir
    parser = argparse.ArgumentParser (description='kpi-1d')
    parser.add_argument ('--verbose',action='store_true',default=False)
    parser.add_argument('--satellite',choices=['S1A','S1B'],required=True,help='S-1 unit choice')
    parser.add_argument('--wv',choices=['wv1','wv2'],required=True,help='WV incidence angle choice')
    parser.add_argument('--enddate',help='end of the 1 month period analysed',required=False,action='store',
                        default=None)
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
    if args.enddate is not None :
        end_date = datetime.datetime.strptime(args.enddate,'%Y%m%d')
    else :
        end_date = args.enddate  # None case
    kpi_v,start_cur_month,stop_cur_month,envelop_val = compute_kpi_1d(sat,wv=wv,dev=True,stop_analysis_period=end_date)
    logging.info('##########')
    logging.info('kpi: %s %s :  %1.3f %% (envelop %s-sigma value: %1.3f m)',sat,wv,kpi_v,ENVELOP,envelop_val)
    logging.info('##########')
    logging.info('start_cur_month : %s stop_cur_month : %s',start_cur_month,stop_cur_month)
    logging.info('done in %1.3f min',(time.time()-t0)/60.)
    logging.info('peak memory usage: %s Mbytes',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.)