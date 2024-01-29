#!/bin/bash
set -e
# A Grouazel
# Oct 2022
echo start
kpiname=$1
datesta=$2
datesto=$3

if [ -z "$datesta" ]
then
  datesta=`date -d ' -5 day' +%Y%m%d`
  datesto=`date -d ' -0 day' +%Y%m%d`
fi


echo $datesta $datesto
#module load singularity/3.6.4

pybin=/home/datawork-cersat-public/project/mpc-sentinel1/workspace/mamba/envs/xsar_oct22/bin/python

myvariable=$(whoami)
if [[ "${myvariable}" == "agrouaze" ]]; then
    echo 'je suis agrouaze'
    exepy=/home1/datahome/agrouaze/git/kpi_mpc/src/push_many_input_files_to_MPC_ECE.py
else
   echo 'je suis '$(whoami)
   exepy=/home1/datahome/satwave/sources_en_exploitation2/kpi_mpc/src/push_many_input_files_to_MPC_ECE.py
fi;
sing_image=/home/datawork-cersat-public/project/mpc-sentinel1/workspace/singularity/push_input_KPI_to_ECE/image_ubuntu22.04.sif
echo 'pybin'$pybin
echo 'exepy'$exepy
echo 'sing_image'$sing_image
#-B /home3/homedir7 # not possible on datarmor nodes

hostna=`hostname`
if [ "${hostna}" == "datarmor" ] || [ "${hostna}" == "datarmor1" ] || [ "${hostna}" == "datarmor2" ] || [ "${hostna}" == "datarmor3" ] ; then
  echo 'hostname is datarmor'
  singubin=/appli/singularity/3.6.4/bin/singularity
else
  echo 'shotname is '$hostna
  singubin=singularity
fi
#$singubin version
echo 'ok for singularity call'
$singubin exec -B /home1/datahome -B /home/datawork-cersat-public -B /home1/datawork -B /home1/scratch  $sing_image $pybin $exepy --kpi $kpiname --startdate $datesta --stopdate $datesto
echo done
