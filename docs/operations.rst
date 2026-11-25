Operations
==========

Production on the Ifremer HPC
-----------------------------

The KPIs are computed and their input files pushed every day by the Airflow
DAG ``sentinel1_WV_kpi_ece`` of
https://gitlab.ifremer.fr/lops-sys/airflow-aas-dags-siam (Airflow lops,
Slurm). The DAG runs the Apptainer image built from ``apptainer/kpimpc.def``
with ``KPIMPC_DATA_ROOT=/legacy/project/cersat/public``. It has four
independent tasks:

``kpi1b_nrcs_wv`` and ``kpi1d_hs_wv``
   ``kpinrcs`` and ``kpihs`` for each active unit and WV angle, with an end
   date every 15 days over the last 20 days. Existing outputs are kept.

``kpi_1b_input_file_ECE_upload`` and ``kpi_1d_input_file_ECE_upload``
   ``kpipush`` over the last 5 days, so that L2F files that arrive late are
   still pushed. When the push is disabled (DAG parameter ``push_enabled``),
   the tasks run ``kpipush --dry-run`` and ``kpipush --check-connection``.

Push to the MPC ECE
-------------------

``kpipush`` uploads the input files with ``sftp`` in batch mode, one session
per file. Each file is first uploaded under a hidden name and then renamed,
so that the ECE never reads a partial file. A failed transfer is retried
(``--retries``), and the command fails if a file could not be pushed.

.. list-table::
   :header-rows: 1

   * - KPI
     - Local file
     - Name on the ECE
   * - 1b
     - ``L1_v16/WV/<sat>_wv_ocean_calibration_..._20150101_today_runv16.nc``
     - ``SAR_MPC_<sat>_WVRCS-ECMWF_V20150101T000000_<G>_G<G>.nc``
   * - 1d
     - ``L2_full_daily/0.8/YYYY/DDD/*.nc``, one per day
     - ``SAR_MPC_<sat>_WVHS-WW3_V<day>T000000_<day+1>T000000_G<G>.nc``

``<G>`` is the generation date, i.e. the push time (UTC), as
``%Y%m%dT%H%M%S``.

Network requirements
~~~~~~~~~~~~~~~~~~~~

The ECE (``upload.sar-mpc.eu``, port 22) only accepts whitelisted source
addresses. Two conditions are needed before enabling the push from the HPC:

* the HPC compute nodes must be able to reach the ECE on port 22, either
  through NAT or through an HTTP CONNECT proxy (then set
  ``KPIMPC_ECE_PROXY_COMMAND``);
* the public address used by that path must be whitelisted by CLS.

Check the connection with::

    kpipush --check-connection
