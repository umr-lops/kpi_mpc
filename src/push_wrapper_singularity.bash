#!/bin/bash
set -e
# A Grouazel
# Oct 2022
echo start
datesta=$1
datesto=$2
kpiname=$3
echo $datesta $datesto
#module load singularity/3.6.4

pybin=/home/datawork-cersat-public/project/mpc-sentinel1/workspace/mamba/envs/xsar_oct22/bin/python
exepy=/home1/datahome/agrouaze/git/kpi_mpc/src/push_many_input_files_to_MPC_ECE.py
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
$singubin version
echo 'ok for singularity call'
$singubin exec -B /home1/datahome -B /home/datawork-cersat-public -B /home1/datawork -B /home1/scratch  $sing_image $pybin $exepy --kpi $kpiname --startdate $datesta --stopdate $datesto
echo done