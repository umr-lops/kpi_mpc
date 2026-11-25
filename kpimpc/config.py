"""Paths and settings of the KPI-1b / KPI-1d processing and of the push to ECE.

Every path is built from ``KPIMPC_DATA_ROOT`` (default: the Datarmor mount
``/home/datawork-cersat-public``). On the Ifremer HPC (Slurm / Airflow lops)
the same tree is mounted at ``/legacy/project/cersat/public``::

    export KPIMPC_DATA_ROOT=/legacy/project/cersat/public

The settings of the push of the KPI input files to the MPC ECE server can be
overridden the same way (``KPIMPC_ECE_*`` variables, see below).
"""

import os

DATA_ROOT = os.environ.get("KPIMPC_DATA_ROOT", "/home/datawork-cersat-public")
PROJECT_DIR_DATARMOR = os.path.join(DATA_ROOT, "cache/project/mpc-sentinel1")
DIR_ANALYSIS = os.path.join(
    PROJECT_DIR_DATARMOR, "analysis", "s1_data_analysis"
)

RASTER_NASA_COASTLINE = os.path.join(
    PROJECT_DIR_DATARMOR,
    "data/ancillary/NASA_tiff_distance_to_coast_converted_v3.nc",
)
DIR_INPUT = os.path.join(DIR_ANALYSIS, "L1_v16", "WV")
INPUT_files = os.path.join(
    DIR_INPUT,
    "%s_wv_ocean_calibration_CMOD5n_ecmwf0125_windspeed_weighted_slc_level1_20150101_today_runv16.nc",
)
EXTRACTION_SERIES_L2F_FOR_LONGTERM_MONITORING_L2 = os.path.join(
    DIR_ANALYSIS, "longterm_extraction_L2F"
)
OUTPUTDIR_KPI_1D = os.path.join(DIR_ANALYSIS, "kpi", "kpi_1d", "v2")
OUTPUTDIR_KPI_1B = os.path.join(DIR_ANALYSIS, "kpi", "kpi_1b", "v2")
VERSION_L2F = "0.8"  # add cci hs NN Quach 2020 in v3.2 format + add 10 classificaitons Chen + probabilities + wind speed SAR + fix missing obs at begining of each days
DIR_L2F_WV_DAILY = os.path.join(DIR_ANALYSIS, "L2_full_daily", VERSION_L2F)
SAR_UNITS = ["S1A", "S1B", "S1C", "S1D"]
# units still flying (S1A decommissioned 29 June 2026, S1B in 2022)
ACTIVE_SAR_UNITS = ["S1C", "S1D"]

# --- push of the KPI input files to the MPC ECE (CLS) sftp server ---
ECE_SFTP_DESTINATION = os.environ.get(
    "KPIMPC_ECE_DESTINATION", "upload@upload.sar-mpc.eu"
)
ECE_REMOTE_DIR = os.environ.get("KPIMPC_ECE_REMOTE_DIR", "upload")
ECE_SSH_KEY = os.environ.get(
    "KPIMPC_ECE_SSH_KEY",
    os.path.join(os.path.expanduser("~"), ".ssh", "id_sar_mpc_ece_upload_cls"),
)
# e.g. "ncat --proxy proxy.local:8080 --proxy-type http %h %p" if the network
# path to the ECE goes through an HTTP CONNECT proxy (empty: direct connection)
ECE_PROXY_COMMAND = os.environ.get("KPIMPC_ECE_PROXY_COMMAND", "")
