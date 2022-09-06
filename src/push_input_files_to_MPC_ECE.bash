#!/bin/bash
set -e
set -u

# script posted by CLS to push input data used by KPI MPC routines
# July 2022
# tuned by A Grouazel for KPI-1b and KPI-1d
# to test connection in command line: sftp -i $HOME/.ssh/id_sar_mpc_ece_upload_cls upload@upload.sar-mpc.eu
# the public SSH key has been upload by P. Vincent on the server
# usage :
#      bash push_input_files_to_MPC_ECE.bash /home/datawork-cersat-public/project/mpc-sentinel1/analysis/s1_data_analysis/L2_full_daily/0.8/2022/242/S1A_WV_L2F_enriched_LOPS_20220830_daily_IPF_03.52.nc

usage() {
  echo "usage: $(basename "$0") source_file|source_dir"
  exit 1
}

if [ $# -ne 1 ] || [ -z "$1" ] ; then
  usage
fi

filename=`basename $1`
arrIN=(${filename//_/ })
echo $arrIN
start_date=${arrIN[5]}
echo "start_date",$start_date
#nextday=$(date  '+%Y%m%d + 1 day' -d $start_date)
start_day=`date "+%Y%m%dT%H%M%S" -d "$start_date"  `
nextday=`date "+%Y%m%dT%H%M%S" -d "$start_date +1 day"`
generation_date=`date "+G%Y%m%dT%H%M%S"`
sat=${arrIN[0]}
echo $sat,$start_day,$nextday,$generation_date
filename_dest=SAR_MPC_${sat}_WVHS-WW3_V${start_day}_${nextday}_${generation_date}.nc
echo $filename_dest
echo -e "put $1 upload/.$filename\nrename upload/.$filename upload/$filename_dest" | sftp -q -r -i $HOME/.ssh/id_sar_mpc_ece_upload_cls upload@upload.sar-mpc.eu
echo 'job finished'