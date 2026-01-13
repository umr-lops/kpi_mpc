#!/bin/bash
set -e
set -u

# script posted by CLS to push input data used by KPI MPC routines
# July 2022
# tuned by A Grouazel for KPI-1b and KPI-1d

# to test connection in command line:
#         sftp -i $HOME/.ssh/id_sar_mpc_ece_upload_cls upload@upload.sar-mpc.eu


# the public SSH key has been upload by P. Vincent on the server
# usage :
#      bash push_input_files_to_MPC_ECE.bash /home/datawork-cersat-public/project/mpc-sentinel1/analysis/s1_data_analysis/L2_full_daily/0.8/2022/242/S1A_WV_L2F_enriched_LOPS_20220830_daily_IPF_03.52.nc
#      bash push_input_files_to_MPC_ECE.bash /home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/L1_v16/WV/S1A_wv_ocean_calibration_CMOD5n_ecmwf0125_windspeed_weighted_slc_level1_20150101_today_runv16.nc
# input files for KPI-1b (NRCS): /home/datawork-cersat-public/cache/project/mpc-sentinel1/analysis/s1_data_analysis/L1_v16/WV/S1A_wv_ocean_calibration_CMOD5n_ecmwf0125_windspeed_weighted_slc_level1_20150101_today_runv16.nc
# output for ECE (WVRCS-ECMWF): SAR_MPC_<mission>_<TYPE>_<subtype>_V<START-DATE>_<END-DATE>_G<GENERATION-DATE>.<Extension> eg SAR_MPC_S1A_WVRCS-ECMWF_V20150101T000000_20220702T000000_G20220702T000000.nc
# documentation https://groupcls.sharepoint.com/:x:/r/sites/MPC-S1/_layouts/15/doc2.aspx?sourcedoc=%7BF5AB8AEA-43E4-45F3-9C93-2F454A79D635%7D&file=KPI_dependencies_20220322.xlsx&action=default&mobileredirect=true&cid=386408fb-2c38-4bc7-bc0b-0ded9b820e0a
# output for ECE (WVHS-WW3):


usage() {
  echo "usage: $(basename "$0") source_file|source_dir"
  exit 1
}

if [ $# -ne 1 ] || [ -z "$1" ] ; then
  usage
fi
filename=`basename $1`
arrIN=(${filename//_/ })
generation_date=`date "+%Y%m%dT%H%M%S"`
echo $arrIN
if [[  "$1" == *$'L2F'* ]]; then
  echo 'Hs KPI case (WW3 L2F etc...)'
  start_date=${arrIN[5]}
  echo "start_date",$start_date
  #nextday=$(date  '+%Y%m%d + 1 day' -d $start_date)
  start_day=`date "+%Y%m%dT%H%M%S" -d "$start_date"  `
  nextday=`date "+%Y%m%dT%H%M%S" -d "$start_date +1 day"`
  sat=${arrIN[0]}
  echo $sat,$start_day,$nextday,$generation_date
  filename_dest=SAR_MPC_${sat}_WVHS-WW3_V${start_day}_${nextday}_G${generation_date}.nc
elif [[ "$1" == *$'L1_v16'*  ]]; then

  echo 'NRCS WV KPI case (ECMWF etc...)'
  #SAR_MPC_S1A_WVRCS-ECMWF_V20150101T000000_20220702T000000_G20220702T000000.nc
  start_day='20150101T000000'
  sat=${arrIN[0]}
  filename_dest=SAR_MPC_${sat}_WVRCS-ECMWF_V${start_day}_${generation_date}_G${generation_date}.nc
else
  filename_dest="none"
  echo 'case not handle'
fi
echo 'filename: '$filename_dest

myvariable=$(whoami)
if [[ "${myvariable}" == "agrouaze" ]]; then
  filessh=/home3/homedir11/perso/agrouaze/.ssh/id_sar_mpc_ece_upload_cls
  filessh=/home1/datahome/agrouaze/.ssh/id_sar_mpc_ece_upload_cls
else
  filessh=/home1/datahome/satwave/.ssh/id_sar_mpc_ece_upload_cls
fi

if [ "$filename_dest" = "none" ]; then
  echo 'nothing to do'
else
  echo -e "put $1 upload/.$filename\nrename upload/.$filename upload/$filename_dest" | sftp -q -r -i $filessh  upload@upload.sar-mpc.eu
fi
echo 'job finished'
