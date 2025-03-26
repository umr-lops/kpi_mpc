#!/bin/bash
#PBS -l walltime=01:30:00
#PBS -l mem=1gb
#PBS -m n
#PBS -N push-inp-KPI-2-ECE
#PBS -q ftp
set -e
# A Grouazel
# Oct 2022
echo start
kpiname=$1
datesta=$2
datesto=$3
echo "kpiname = $kpiname"
echo "datesta = $datesta"
echo "datesto = $datesto"
if [ -z "$datesta" ]
then
  datesta=`date -d ' -5 day' +%Y%m%d`
  datesto=`date -d ' -0 day' +%Y%m%d`
fi

ipadress=`ip addr | grep 134`
echo "IP adress host $ipadress"
echo $datesta $datesto
#source /usr/share/Modules/3.2.10/init/bash
source /usr/share/modules/init/bash # found on DGX 101-9
module load singularity/3.6.4
echo 'singularity module loaded'

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
#if [ "${hostna}" == "datarmor" ] || [ "${hostna}" == "datarmor1" ] || [ "${hostna}" == "datarmor2" ] || [ "${hostna}" == "datarmor3" ] ; then
if [[ "${hostna}" =~ ^(datarmor[1-3]?|compute-101-(9|15|23))$ ]] ; then
  echo 'hostname is datarmor'
  module load singularity/3.6.4
  singubin=/appli/singularity/3.6.4/bin/singularity
else
  echo 'shotname is '$hostna
  singubin=singularity
fi
#$singubin version
echo 'ok for singularity call'
# previously singularity exec
$singubin exec --bind /home1/datahome --bind /home/datawork-cersat-public --bind /home1/datawork --bind /home1/scratch  $sing_image $pybin $exepy --kpi $kpiname --startdate $datesta --stopdate $datesto
echo done
