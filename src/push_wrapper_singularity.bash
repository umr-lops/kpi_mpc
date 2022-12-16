#!/bin/bash
# A Grouazel
# Oct 2022
echo start
datesta=$1
datesto=$2
kpiname=$3
echo $datesta $datesto
module load singularity/3.6.4
singularity version
pybin=/home/datawork-cersat-public/project/mpc-sentinel1/workspace/mamba/envs/xsar_oct22/bin/python
exepy=/home1/datahome/agrouaze/git/kpi_mpc/src/push_many_input_files_to_MPC_ECE.py
sing_image=/home/datawork-cersat-public/project/mpc-sentinel1/workspace/singularity/push_input_KPI_to_ECE/image_ubuntu22.04.sif
echo 'pybin'$pybin
echo 'exepy'$exepy
echo 'sing_image'$sing_image
#-B /home3/homedir7 # not possible on datarmor nodes
singularity exec  -B /home1/datahome -B /home/datawork-cersat-public -B /home1/datawork -B /home1/scratch  $sing_image $pybin $exepy --kpi $kpiname --startdate $datesta --stopdate $datesto
echo done