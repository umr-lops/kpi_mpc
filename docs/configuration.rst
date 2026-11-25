Configuration
=============

The settings live in :mod:`kpimpc.config`. They are read once, at import time,
from environment variables.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Variable
     - Default
     - Use
   * - ``KPIMPC_DATA_ROOT``
     - ``/home/datawork-cersat-public``
     - root of all input and output paths. Use
       ``/legacy/project/cersat/public`` on the Ifremer HPC (Slurm).
   * - ``KPIMPC_ECE_DESTINATION``
     - ``upload@upload.sar-mpc.eu``
     - sftp user and host of the MPC ECE
   * - ``KPIMPC_ECE_REMOTE_DIR``
     - ``upload``
     - remote directory of the uploaded files
   * - ``KPIMPC_ECE_SSH_KEY``
     - ``~/.ssh/id_sar_mpc_ece_upload_cls``
     - private ssh key accepted by the ECE
   * - ``KPIMPC_ECE_PROXY_COMMAND``
     - empty (direct connection)
     - ssh ``ProxyCommand``, e.g.
       ``ncat --proxy proxy.local:8080 --proxy-type http %h %p``

With Apptainer, pass the variables with ``--env`` (or as
``APPTAINERENV_<NAME>``) and bind the data tree::

    apptainer exec --bind /legacy \
        --env KPIMPC_DATA_ROOT=/legacy/project/cersat/public \
        kpimpc.sif kpinrcs --satellite S1C --wv wv1 --enddate 20261120

Paths derived from the data root
--------------------------------

.. list-table::
   :header-rows: 1

   * - Setting
     - Path under ``KPIMPC_DATA_ROOT``
   * - :data:`~kpimpc.config.INPUT_files` (KPI-1b input)
     - ``cache/project/mpc-sentinel1/analysis/s1_data_analysis/L1_v16/WV/<sat>_wv_ocean_calibration_..._runv16.nc``
   * - :data:`~kpimpc.config.DIR_L2F_WV_DAILY` (KPI-1d input)
     - ``.../s1_data_analysis/L2_full_daily/0.8/YYYY/DDD/``
   * - :data:`~kpimpc.config.OUTPUTDIR_KPI_1B`
     - ``.../s1_data_analysis/kpi/kpi_1b/v2``
   * - :data:`~kpimpc.config.OUTPUTDIR_KPI_1D`
     - ``.../s1_data_analysis/kpi/kpi_1d/v2``
   * - :data:`~kpimpc.config.RASTER_NASA_COASTLINE`
     - ``cache/project/mpc-sentinel1/data/ancillary/NASA_tiff_distance_to_coast_converted_v3.nc``
