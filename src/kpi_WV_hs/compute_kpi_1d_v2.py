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
from read_and_concat_L2F import read_L2F_with_xarray
from src.config import EXTRACTION_SERIES_L2F_FOR_LONGTERM_MONITORING_L2
POLARIZATION = 'VV'
MODE = 'WV'
ENVELOP = 2 #sigma
PRIOR_PERIOD = 3 #months
LAT_MAX = 55
MIN_DIST_2_COAST = 100 #km
MS = {'wv1':-6.5,'wv2':-15.} #NRCS central values dB
DS = {'wv1':3,'wv2':5} #delta
def load_Level2_series(satellite,start,stop,alternative_L2F=None):
    """

    :param satellite: str S1A or ...
    :return:
    """
    # component = 'osw'
    # fpathagg = os.path.join(EXTRACTION_SERIES_L2F_FOR_LONGTERM_MONITORING_L2,
    #                         '%s_%s_longterm_inputdata_from_L2F_%s.nc' % (
    #                             component,satellite,'fullmission'))  # for dev/test last365days
    # logging.info('path aggregation = %s exists %s',fpathagg,os.path.exists(fpathagg))
    # df = xarray.open_dataset(fpathagg).to_dataframe()
    # logging.info("%s %s ok",component,satellite)
    # logging.debug('keys : %s',df.keys())

    logging.info('load L2F data')
    #start = datetime.datetime(2019,1,1)
    #stop = datetime.datetime(2020,1,1)  # pour test
    vv = ['oswQualityFlagPartition1','fdatedt','oswLon','oswLat',
          's1_effective_hs_2Dcutoff','ecmwf0125_uwind','ecmwf0125_vwind',
          'oswIncidenceAngle','oswLandFlag','dist2coastKM', 'pol','class_1',
          'ww3_effective_2Dcutoff_hs','oswNv','oswNrcs','oswAzCutoff','oswEcmwfWindSpeed',
          'oswQualityFlagPartition1','oswQualityFlagPartition2','oswQualityFlagPartition3',
          'oswQualityFlagPartition4','oswQualityFlagPartition5','s1_hs_emp_tot_v3p2',
          'oswXA_hs_ww3spec_firstSARpartition','oswXA_hs_ww3spec_secondSARpartition','oswXA_hs_ww3spec_thirdSARpartition',
          'oswXA_hs_ww3spec_fourthSARpartition','oswXA_hs_ww3spec_fifthSARpartition',
          "oswXA_wl_ww3spec_firstSARpartition",'oswXA_wl_ww3spec_secondSARpartition','oswXA_wl_ww3spec_thirdSARpartition',
          'oswXA_wl_ww3spec_fourthSARpartition','oswXA_wl_ww3spec_fifthSARpartition'
            ]
    ds_dict_sat = read_L2F_with_xarray(start,stop,satellites=[satellite],variables=vv,
                                       alternative_L2F_path=alternative_L2F,
                                       add_ecmwf_wind=True)
    dswv = ds_dict_sat[satellite]
    if dswv != {}:
        # drop Nan
        dswv = dswv.where(np.isfinite(dswv['s1_effective_hs_2Dcutoff']) & np.isfinite(dswv['ww3_effective_2Dcutoff_hs']),drop=True)
    return dswv

def compute_kpi_1d(sat,wv,version,dev=False,stop_analysis_period=None,period_analysed_width=30,df=None,alternative_L2F=None):
    """
    osw VV WV S-1 effective Hs (2D-cutoff) compared to WWIII Hs computed on same grid/same mask
    note that low freq mask is applied both on S-1 spectrum and WWIII spectrum
    :param sat: str S1A or ..
    :param wv: str wv1 or wv2
    :param version: str test1 test2 ..
    :param stop_analysis_period: datetime (-> period considered date-1 month : date)
    :param period_analysed_width : int 30 days by default
    :return:
        kpi_value (float): between 0 and 100 %
        start_current_month (datetime):
        stop_current_month (datetime):
        envelop_value : (float) 2-sigma m threshold based on 3 months prior period
    """
    #compute the 2 sigma envelopp on the last 3 months prior to current month
    if stop_analysis_period is None :
        stop_current_month = datetime.datetime.today()
    else :
        stop_current_month = stop_analysis_period
    start_current_month = stop_current_month - datetime.timedelta(days=period_analysed_width)
    start_prior_period = start_current_month -  datetime.timedelta(days=30*PRIOR_PERIOD)
    stop_prior_period = start_current_month
    nb_measu_total = 0
    if df is None:
        df = load_Level2_series(satellite=sat,start=start_prior_period,stop=stop_current_month,alternative_L2F=alternative_L2F)


    logging.debug('start_current_month : %s',start_current_month)
    logging.debug('stop_current_month : %s',stop_current_month)

    logging.debug('prior period ; %s to %s',start_prior_period,stop_prior_period)
    _swh_azc_mod = df['ww3_effective_2Dcutoff_hs'].values
    logging.debug('nb value WW3 eff Hs above 25 m : %s',(_swh_azc_mod>25).sum())
    if (_swh_azc_mod>25).sum()>0:
        ind_bad_ww3 = np.where(_swh_azc_mod>25)[0][0]
        logging.debug('a date SAR for which ww3 is extremely too high: %s -> Hs:%1.1fm',
                     df['fdatedt'][ind_bad_ww3],_swh_azc_mod[ind_bad_ww3])
    logging.debug('max Hs WW3 : %s',np.nanmax(_swh_azc_mod))
    _swh_azc_s1 = df['s1_effective_hs_2Dcutoff'].values
    #df['bias_swh_azc_' + wv] = _swh_azc_s1 - _swh_azc_mod
    if version=='test1' or version=='test4' or version=='test5':
        df['bias_swh_azc_' + wv] = xarray.DataArray(_swh_azc_s1 - _swh_azc_mod,dims=['fdatedt'],coords={'fdatedt':df['fdatedt']})
    elif version=='test2' or version=='test9':
        df['bias_swh_azc_' + wv] = \
            xarray.DataArray(df['oswXA_hs_ww3spec_firstSARpartition'].values[:,0] - df['oswXA_hs_ww3spec_firstSARpartition'].values[:,1],
                             dims=['fdatedt'],coords={'fdatedt':df['fdatedt']})
    elif version == 'test3' or version=='test7' or version=='test8':
        df['bias_swh_azc_' + wv] = xarray.DataArray(
            df['oswXA_hs_ww3spec_firstSARpartition'].values[:,0] - df['oswXA_hs_ww3spec_firstSARpartition'].values[:,1],
            dims=['fdatedt'],coords={'fdatedt':df['fdatedt']})
    elif version == 'test6': #wavelength bias
        df['bias_swh_azc_' + wv] = xarray.DataArray(
            df['oswXA_wl_ww3spec_firstSARpartition'].values[:,0] - df['oswXA_wl_ww3spec_firstSARpartition'].values[:,1],
            dims=['fdatedt'],coords={'fdatedt':df['fdatedt']})
    else:
        raise Exception('version unknown')
    if wv=='wv1':
        cond_inc = (df['oswIncidenceAngle']<30)
    elif wv=='wv2':
        cond_inc = (df['oswIncidenceAngle'] > 30)
    else:
        raise Exception('wv value un expected : %s'%wv)
    polarizationcond = (df.pol == POLARIZATION.lower())
    logging.debug('df.pol %s',df.pol.values)
    logging.debug('polarizationcond %s',polarizationcond.values.sum())
    if version=='test1':
        cond_nrcs = (np.abs(df['oswNrcs'] - MS[wv]) < DS[wv])

        #moderate wind speed conditions
        delta_ws = 2.5
        central_ws = 7
        cond_outlier_ww3_hs = (abs(df['ww3_effective_2Dcutoff_hs']) < 50) # & (abs(df['ww3_effective_2Dcutoff_hs']) > 1)
        cond_wind = (np.abs(df['ecmwf_wind_speed']-central_ws)<delta_ws)
        logging.debug('nb pts after wind filter %s %1.1f%%',np.sum(cond_wind.values),100.0*np.sum(cond_wind.values)/cond_wind.size)
        logging.debug('nb pts after NRCS filter %s %1.1f%%',np.sum(cond_nrcs.values),100.0*np.sum(cond_nrcs.values)/cond_nrcs.size)

        ocean_acqui_filters = cond_outlier_ww3_hs & cond_wind & polarizationcond & cond_nrcs & (abs(df['oswLat']) < LAT_MAX) \
                              & (df['oswLandFlag'] == 0) & (df['dist2coastKM'] > MIN_DIST_2_COAST) & cond_inc &(np.isfinite(df['bias_swh_azc_' + wv]))
    elif version == 'test2':
        cond_quality = (df['oswQualityFlagPartition1']==0) | (df['oswQualityFlagPartition1']==1) #very good
        cond_outlier_ww3_hs = (abs(df['ww3_effective_2Dcutoff_hs']) < 50)
        logging.debug('cond_quality %s',cond_quality.values.sum())
        fini_bias =  np.isfinite(df['bias_swh_azc_' + wv])
        logging.debug('fini_bias %s',fini_bias.values.sum())
        latmax_cond = (abs(df['oswLat']) < LAT_MAX)
        logging.debug('latmax_cond %s',latmax_cond.values.sum())
        dstmax_cond = (df['dist2coastKM'] > MIN_DIST_2_COAST)
        logging.debug('dstmax_cond %s',dstmax_cond.values.sum())
        ocean_acqui_filters = cond_quality & polarizationcond & latmax_cond & cond_outlier_ww3_hs \
                              & (df['oswLandFlag'] == 0) & dstmax_cond & cond_inc & fini_bias
        logging.debug('ocean_acqui_filters %s',ocean_acqui_filters.values.sum())
    elif version == 'test9':
        cond_quality = (df['oswQualityFlagPartition1']==0) | (df['oswQualityFlagPartition1']==1) | (df['oswQualityFlagPartition1']==2) #very good
        cond_outlier_ww3_hs = (abs(df['ww3_effective_2Dcutoff_hs']) < 50)
        logging.debug('cond_quality %s',cond_quality.values.sum())
        fini_bias =  np.isfinite(df['bias_swh_azc_' + wv])  & (abs(df['bias_swh_azc_' + wv])<50)
        logging.debug('fini_bias %s',fini_bias.values.sum())
        latmax_cond = (abs(df['oswLat']) < LAT_MAX)
        logging.debug('latmax_cond %s',latmax_cond.values.sum())
        dstmax_cond = (df['dist2coastKM'] > MIN_DIST_2_COAST)
        logging.debug('dstmax_cond %s',dstmax_cond.values.sum())
        ocean_acqui_filters = cond_quality & polarizationcond & latmax_cond & cond_outlier_ww3_hs \
                              & (df['oswLandFlag'] == 0) & dstmax_cond & cond_inc & fini_bias
        logging.debug('ocean_acqui_filters %s',ocean_acqui_filters.values.sum())

    elif version=='test3' or version=='test4' or version=='test6' or version=='test7' or version=='test8':
        cond_outlier_ww3_hs = (abs(df['ww3_effective_2Dcutoff_hs']) < 50)
        logging.debug('cond_outlier_ww3_hs %s',cond_outlier_ww3_hs.values.sum())
        fini_bias = np.isfinite(df['bias_swh_azc_' + wv]) & (abs(df['bias_swh_azc_' + wv])<50)
        logging.debug('fini_bias %s',fini_bias.values.sum())
        latmax_cond = (abs(df['oswLat']) < LAT_MAX)
        logging.debug('latmax_cond %s',latmax_cond.values.sum())
        dstmax_cond = (df['dist2coastKM'] > MIN_DIST_2_COAST)
        logging.debug('dstmax_cond %s',dstmax_cond.values.sum())
        ocean_acqui_filters = polarizationcond & latmax_cond & cond_outlier_ww3_hs \
                              & (df['oswLandFlag'] == 0) & dstmax_cond & cond_inc & fini_bias
        logging.debug('ocean_acqui_filters %s',ocean_acqui_filters.values.sum())
    elif version=='test5':
        cond_outlier_ww3_hs = (abs(df['ww3_effective_2Dcutoff_hs']) < 50) & (df['ww3_effective_2Dcutoff_hs']>0.3)
        logging.debug('cond_outlier_ww3_hs %s',cond_outlier_ww3_hs.values.sum())
        fini_bias = np.isfinite(df['bias_swh_azc_' + wv])
        logging.debug('fini_bias %s',fini_bias.values.sum())
        latmax_cond = (abs(df['oswLat']) < LAT_MAX)
        logging.debug('latmax_cond %s',latmax_cond.values.sum())
        dstmax_cond = (df['dist2coastKM'] > MIN_DIST_2_COAST)
        logging.debug('dstmax_cond %s',dstmax_cond.values.sum())
        ocean_acqui_filters = polarizationcond & latmax_cond & cond_outlier_ww3_hs \
                              & (df['oswLandFlag'] == 0) & dstmax_cond & cond_inc & fini_bias
        logging.debug('ocean_acqui_filters %s',ocean_acqui_filters.values.sum())
    else:
        raise Exception('version unknown')
    logging.debug('start_prior_period : %s -> stop_prior_period %s',start_prior_period,stop_prior_period)
    #logging.debug('fdatetd : %s',df['fdatedt'])
    start_prior_period64 = np.datetime64(start_prior_period)
    stop_prior_period64 = np.datetime64(stop_prior_period)
    mask_prior_period = (df['fdatedt']>=start_prior_period64) & (df['fdatedt']<=stop_prior_period64)
    final_mask_prior = ocean_acqui_filters & mask_prior_period
    logging.debug('final_mask_prior %s',final_mask_prior.values.sum())
    logging.debug('nb pts in prior period (without extra filters) : %s',mask_prior_period.values.sum())
    #subset_df = df[final_mask_prior]
    subset_df = df.where(final_mask_prior,drop=True)
    nb_nan = np.isnan(subset_df['bias_swh_azc_' + wv].values).sum()
    #logging.debug('some values: %s',subset_df['bias_swh_azc_' + wv].values)
    logging.debug('nb_nan : %s',nb_nan)
    logging.debug('nb finite %s/%s',np.isfinite(subset_df['bias_swh_azc_' + wv].values).sum(),len(subset_df['bias_swh_azc_' + wv]))
    envelop_value = ENVELOP*np.nanstd(subset_df['bias_swh_azc_' + wv].values)
    logging.debug('envelop_value : %s',envelop_value)

    #compute the number of product within the envelop for current month
    start_current_month64 = np.datetime64(start_current_month)
    stop_current_month64 = np.datetime64(stop_current_month)
    current_date_cond = (df['fdatedt'] >= start_current_month64) & (df['fdatedt'] <= stop_current_month64)
    logging.debug('current_date_cond %s',current_date_cond.values.sum())
    logging.debug('ocean_acqui_filters %s',ocean_acqui_filters.values.sum())
    mask_current_month = ocean_acqui_filters & current_date_cond
    logging.debug('mask_current_month %s',mask_current_month.values.sum())
    #subset_current_period = df[mask_current_month]
    subset_current_period = df.where(mask_current_month,drop=True)
    if 'ww3_effective_2Dcutoff_hs' in subset_current_period and len(subset_current_period['ww3_effective_2Dcutoff_hs'])>0:
        logging.debug('max Hs WW3 in subset : %s',np.nanmax(subset_current_period['ww3_effective_2Dcutoff_hs']))
        logging.debug('max Hs SAR in subset : %s',np.nanmax(subset_current_period['s1_effective_hs_2Dcutoff']))
        logging.debug('min Hs WW3 in subset : %s',np.nanmin(subset_current_period['ww3_effective_2Dcutoff_hs']))
        logging.debug('min Hs SAR in subset : %s',np.nanmin(subset_current_period['s1_effective_hs_2Dcutoff']))
        logging.debug('nb pts current month : %s',len(subset_current_period['fdatedt']))
        nb_measu_total = len(subset_current_period['fdatedt'])
        logging.debug('bias : %s',subset_current_period['bias_swh_azc_' + wv].values)
        if version=='test7':
            current_mean_bias = np.mean(subset_current_period['bias_swh_azc_' + wv]).values
            cond_sup = subset_current_period['bias_swh_azc_' + wv]<=current_mean_bias+envelop_value
            cond_inf = subset_current_period['bias_swh_azc_' + wv]>=current_mean_bias-envelop_value
            nb_measu_inside_envelop = (cond_sup & cond_inf).sum().values
        elif version=='test8':#definition proposee par Hajduch le 10dec2021 screenshot a lappuit (je ne suis pas convaincu pas l introduction du biais dans le calcul de levenveloppe car le KPI sera dautant plus elever que le biais sera fort (cest linverse qui est cherche)
            bias_minus_2sigma = abs(subset_current_period['bias_swh_azc_' + wv].mean().values-envelop_value)
            bias_plus_2sigma = abs(subset_current_period['bias_swh_azc_' + wv].mean().values+envelop_value)
            logging.info('bias_plus_2sigma %s',bias_plus_2sigma)
            T = np.max([bias_minus_2sigma,bias_plus_2sigma])
            logging.info('T : %s %s',T.shape,T)
            nb_measu_inside_envelop = (abs(subset_current_period['bias_swh_azc_' + wv])<T).sum().values
        else:
            nb_measu_inside_envelop = (abs(subset_current_period['bias_swh_azc_' + wv])<envelop_value).sum().values
        mean_bias = np.mean(subset_current_period['bias_swh_azc_' + wv]).values
        logging.debug('nb_measu_inside_envelop : %s',nb_measu_inside_envelop)
        kpi_value = 100.*nb_measu_inside_envelop/nb_measu_total
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
            logging.debug('png test : %s',output)
            #plt.show()
    else:
        mean_bias = np.nan
        kpi_value = np.nan
        logging.debug('no data for period %s to %s',start_current_month,stop_current_month)
        logging.debug('subset_current_period %s',subset_current_period)
    return kpi_value,start_current_month,stop_current_month,envelop_value,nb_measu_total,mean_bias


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
    parser.add_argument('--overwrite',action='store_true',default=False,
                        help='overwrite the existing outputs [default=False]',required=False)
    parser.add_argument('--satellite',choices=['S1A','S1B'],required=True,help='S-1 unit choice')
    parser.add_argument('--wv',choices=['wv1','wv2'],required=True,help='WV incidence angle choice')
    parser.add_argument('--enddate',help='end of the 1 month period analysed',required=False,action='store',
                        default=None)
    parser.add_argument('--version',choices=['test1','test2','test3','test4','test5','test6','test7','test8','test9'],help='version of the KPI-1d',required=False,default='test1')
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
    output_file = '/home1/scratch/agrouaze/kpi_1d_v2/%s/kpi_output_%s_%s_%s.txt' % (args.version,
    sat,wv,end_date.strftime('%Y%m%d'))
    if os.path.exists(output_file) and args.overwrite is False:
        logging.info('output %s already exists',output_file)
    else:
        alternative_L2F = '/home/datawork-cersat-public/project/mpc-sentinel1/analysis/s1_data_analysis/L2_full_daily/0.8/'
        kpi_v,start_cur_month,stop_cur_month,envelop_val,nbvalused,meanbias = compute_kpi_1d(sat,wv=wv,version=args.version,
                                                                        dev=False,stop_analysis_period=end_date,df=None,
                                                                        alternative_L2F=alternative_L2F)
        logging.info('##########')
        logging.info('kpi: %s %s :  %1.3f %% (envelop %s-sigma value: %1.3f m)',sat,wv,kpi_v,ENVELOP,envelop_val)
        logging.info('nb pts used: %s',nbvalused)
        logging.info('##########')
        logging.info('start_cur_month : %s stop_cur_month : %s',start_cur_month,stop_cur_month)


        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file),0o0775)
        fid = open(output_file,'w')
        fid.write('%s %s %s %s %s %s\n'%(kpi_v,start_cur_month,stop_cur_month,envelop_val,nbvalused,meanbias))
        fid.close()
        logging.info('output: %s',output_file)
        logging.info('done in %1.3f min',(time.time() - t0) / 60.)
        logging.info('peak memory usage: %s Mbytes',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.)
    logging.info('end')
