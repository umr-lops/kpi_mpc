"""Paths and settings of the KPI-1b / KPI-1d processing and of the push to ECE.

Every path is built from ``KPIMPC_DATA_ROOT`` (default: the Datarmor mount
``/home/datawork-cersat-public``). On the Ifremer HPC (Slurm / Airflow lops)
the same tree is mounted at ``/legacy/project/cersat/public``::

    export KPIMPC_DATA_ROOT=/legacy/project/cersat/public

The settings of the push of the KPI input files to the MPC ECE server can be
overridden the same way (``KPIMPC_ECE_*`` variables, see below).
"""

import os

#: Root of all the data paths (``KPIMPC_DATA_ROOT``).
DATA_ROOT = os.environ.get("KPIMPC_DATA_ROOT", "/home/datawork-cersat-public")
#: MPC Sentinel-1 project directory.
PROJECT_DIR_DATARMOR = os.path.join(DATA_ROOT, "cache/project/mpc-sentinel1")
#: Directory of the MPC analyses (inputs and outputs of the KPIs).
DIR_ANALYSIS = os.path.join(
    PROJECT_DIR_DATARMOR, "analysis", "s1_data_analysis"
)

#: NASA GSFC distance-to-coast raster (netCDF).
RASTER_NASA_COASTLINE = os.path.join(
    PROJECT_DIR_DATARMOR,
    "data/ancillary/NASA_tiff_distance_to_coast_converted_v3.nc",
)
#: Directory of the aggregated WV SLC calibration files (KPI-1b input).
DIR_INPUT = os.path.join(DIR_ANALYSIS, "L1_v16", "WV")
#: Aggregated WV SLC calibration file, ``%s`` being the unit (KPI-1b input).
INPUT_files = os.path.join(
    DIR_INPUT,
    "%s_wv_ocean_calibration_CMOD5n_ecmwf0125_windspeed_weighted_slc_level1_20150101_today_runv16.nc",
)
#: Directory of the long-term L2F extractions.
EXTRACTION_SERIES_L2F_FOR_LONGTERM_MONITORING_L2 = os.path.join(
    DIR_ANALYSIS, "longterm_extraction_L2F"
)
#: Output directory of KPI-1d (``kpihs``).
OUTPUTDIR_KPI_1D = os.path.join(DIR_ANALYSIS, "kpi", "kpi_1d", "v2")
#: Output directory of KPI-1b (``kpinrcs``).
OUTPUTDIR_KPI_1B = os.path.join(DIR_ANALYSIS, "kpi", "kpi_1b", "v2")
#: Version of the WV L2F daily files.
VERSION_L2F = "0.8"  # add cci hs NN Quach 2020 in v3.2 format + add 10 classificaitons Chen + probabilities + wind speed SAR + fix missing obs at begining of each days
#: Directory of the WV L2F daily files (KPI-1d input).
DIR_L2F_WV_DAILY = os.path.join(DIR_ANALYSIS, "L2_full_daily", VERSION_L2F)
#: Sentinel-1 units accepted by the KPI commands.
SAR_UNITS = ["S1A", "S1B", "S1C", "S1D"]
# units still flying (S1A decommissioned 29 June 2026, S1B in 2022)
#: Units still flying, default of ``kpipush``.
ACTIVE_SAR_UNITS = ["S1C", "S1D"]

# --- push of the KPI input files to the MPC ECE (CLS) sftp server ---
#: sftp user and host of the MPC ECE (``KPIMPC_ECE_DESTINATION``).
ECE_SFTP_DESTINATION = os.environ.get(
    "KPIMPC_ECE_DESTINATION", "upload@upload.sar-mpc.eu"
)
#: Remote directory on the ECE (``KPIMPC_ECE_REMOTE_DIR``).
ECE_REMOTE_DIR = os.environ.get("KPIMPC_ECE_REMOTE_DIR", "upload")
#: Private ssh key accepted by the ECE (``KPIMPC_ECE_SSH_KEY``).
ECE_SSH_KEY = os.environ.get(
    "KPIMPC_ECE_SSH_KEY",
    os.path.join(os.path.expanduser("~"), ".ssh", "id_sar_mpc_ece_upload_cls"),
)
# e.g. "ncat --proxy proxy.local:8080 --proxy-type http %h %p" if the network
# path to the ECE goes through an HTTP CONNECT proxy (empty: direct connection)
#: ssh ProxyCommand to reach the ECE, empty for a direct connection
#: (``KPIMPC_ECE_PROXY_COMMAND``).
ECE_PROXY_COMMAND = os.environ.get("KPIMPC_ECE_PROXY_COMMAND", "")
