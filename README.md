# Kpi Mpc

libs to compute 2 KPIs  for MPC Sentinel-1 project:
  - a) WV VV effective Hs bias (wrt WW3)
  - b) WV VV NRCS (denoised) bias (wrt CMOD-5n)
#author:
    IFREMER LOPS
    antoine.grouazel@ifremer.fr
#creation:
    2021
    
#usage
    python ./src/kpi_WV_nrcs/compute_kpi_1b.py --verbose


#installation

```bash
conda create -n kpi_conda_env python=3.9
conda activate kpi_conda_env
conda install numpy scipy matplotlib xarray netCDF4 ipykernel
git clone https://gitlab.ifremer.fr/lops-wave/kpi_mpc.git
cd kpi_mpc
python setup.py install 
```
#data dependencies:

## KPI-1b (NRCS SLC)
KPI-1b needs a netCDF file containing the differences of NRCS denoised per SAFE for all the WV since 2015 ( S1%_wv_ocean_calibration_CMOD5n_ecmwf0125_windspeed_weighted_slc_level1_20150101_today_runv16.nc)

## KPI-1d (Hs OCN)
KPI-1d is designed to run using Sentinel-1 WV IFREMER L2F daily aggregated nc files.
