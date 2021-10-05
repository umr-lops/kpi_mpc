"""
IFREMER
Oct 2021
KPI-1b SLA Vol3 document MPC contract 2021 - 2016
"""
import os
import logging
import numpy as np
import datetime
from read_aggregated_calbration_SLC_WV_level_netcdf_file_for_nrcs_investigations import read_fat_calib_nc

POLARIZATION = 'VV'
MODE = 'WV'
ENVELOP = 2 #sigma
PRIOR_PERIOD = 3 #months
def compute_kpi_1b(sat):
    """
    NRCS (denoised) observed compared to predicted GMF CMOD5n
    :param sat: str S1A or ..
    :return:
        kpi_value (float): between 0 and 100 %
        start_current_month (datetime):
        stop_current_month (datetime):
        envelop_value : (float) 2-sigma dB threshold based on 3 months prior period
    """
    df_slc_sat = read_fat_calib_nc(sta_str=None,sto_str=None,subdir=None)
    stop_current_month = datetime.datetime.today()
    start_current_month = stop_current_month -  datetime.timdelta(days=30)
    logging.debug('start_current_month : %s',start_current_month)
    logging.debug('stop_current_month : %s',stop_current_month)
    #compute the 2 sigma envelopp on the last 3 months prior to current month
    start_prior_period = start_current_month -  datetime.timdelta(days=30*3)
    stop_prior_period = start_current_month
    df_slc = df_slc_sat[sat]
    df_slc['direct_diff_calib_cst_db'] = df_slc['sigma0_denoised_db'] - df_slc['tmp_gmf_cmod5n_nrcs_db']
    mask_prior_period = (df_slc['time']>=start_prior_period) & (df_slc['time']<=stop_prior_period)
    subset_df = df_slc[mask_prior_period]
    envelop_value = ENVELOP*np.std(subset_df['direct_diff_calib_cst_db'])
    logging.debug('envelop_value : %s',envelop_value)

    #compute the number of product within the envelop for current month
    nb_measu_total = 0
    nb_measu_outside_envelop = 0
    mask_current_month = (df_slc['time']>=start_current_month) & (df_slc['time']<=stop_current_month)
    subset_current_period = df_slc[mask_current_month]
    nb_measu_total = len(subset_current_period['time'])
    nb_measu_outside_envelop = (abs(subset_current_period['direct_diff_calib_cst_db'])<envelop_value).sum()

    kpi_value = 100.*nb_measu_outside_envelop/nb_measu_total
    logging.debug('kpi_value : %s',kpi_value)
    return kpi_value,start_current_month,stop_current_month,envelop_value
