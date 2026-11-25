# Kpi Mpc

Documentation: https://kpi-mpc.readthedocs.io (sources in `docs/`).

libs to compute 2 KPIs for MPC Sentinel-1 project:

- a) WV VV effective Hs bias (wrt WW3) called KPI-1d
- b) WV VV NRCS (denoised) bias (wrt CMOD-5n) called KPI-1b

# author:

    IFREMER LOPS
    antoine.grouazel@ifremer.fr

# creation:

    2021

# usage in command-line

    kpihs -h      # KPI-1d
    kpinrcs -h    # KPI-1b
    kpipush -h    # push of the KPI input files to the MPC ECE (sftp)

The data paths are built from `KPIMPC_DATA_ROOT` (default `/home/datawork-cersat-public`,
Datarmor); on the Ifremer HPC (Slurm):

    export KPIMPC_DATA_ROOT=/legacy/project/cersat/public

The ECE settings can be overridden with `KPIMPC_ECE_DESTINATION`, `KPIMPC_ECE_REMOTE_DIR`,
`KPIMPC_ECE_SSH_KEY` and `KPIMPC_ECE_PROXY_COMMAND` (see `kpimpc/config.py`).

# Apptainer image (production on the Ifremer HPC / Airflow lops)

```bash
V=$(git describe --tags)
apptainer build --fakeroot --build-arg KPIMPC_VERSION=$V kpimpc_$V.sif apptainer/kpimpc.def
apptainer exec --bind /legacy --env KPIMPC_DATA_ROOT=/legacy/project/cersat/public kpimpc_$V.sif \
    kpinrcs --satellite S1C --wv wv1 --enddate 20261120 --outputdir /tmp/kpi_1b
apptainer exec --bind /legacy --env KPIMPC_DATA_ROOT=/legacy/project/cersat/public kpimpc_$V.sif \
    kpipush --kpi kpi1d --startdate 20261120 --stopdate 20261125 --dry-run
```

The image is used by the DAG `sentinel1_WV_kpi_ece` of
https://gitlab.ifremer.fr/lops-sys/airflow-aas-dags-siam

# usage in python interactive mode

> > > import kpimpc

# installation

```bash
conda create -n kpi_conda_env python=3.9
conda activate kpi_conda_env
conda install numpy scipy matplotlib xarray netCDF4 ipykernel
git clone https://github.com/umr-lops/kpi_mpc.git
cd kpi_mpc
# python setup.py install
pip install .
```

#data dependencies:

## KPI-1b (NRCS SLC)

KPI-1b needs a netCDF file containing the differences of NRCS denoised per SAFE for all the WV since 2015 ( S1%\_wv_ocean_calibration_CMOD5n_ecmwf0125_windspeed_weighted_slc_level1_20150101_today_runv16.nc)

## KPI-1d (Hs OCN)

KPI-1d is designed to run using Sentinel-1 WV IFREMER L2F daily aggregated nc files.
