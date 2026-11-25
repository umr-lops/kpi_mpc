Command-line usage
==================

Each KPI command processes one unit, one WV incidence angle and one end date,
and writes one text file
``<outputdir>/<version>/kpi_output_<sat>_<wv>_<YYYYMMDD>.txt``. An existing
output is kept unless ``--overwrite`` is given.

The output line holds the KPI (%), the start and stop of the analysed period,
the envelope, the number of measurements, the mean bias and the standard
deviation of the difference.

Example on the Ifremer HPC::

    export KPIMPC_DATA_ROOT=/legacy/project/cersat/public
    kpinrcs --satellite S1C --wv wv1 --enddate 20261120 --outputdir /tmp/kpi_1b
    kpihs --satellite S1D --wv wv2 --enddate 20261120 --outputdir /tmp/kpi_1d
    kpipush --kpi kpi1d --startdate 20261120 --stopdate 20261125 --dry-run

kpinrcs (KPI-1b)
----------------

Reads the aggregated WV SLC calibration file of the unit
(:data:`kpimpc.config.INPUT_files`). Output version directory:
``v2percentile95``.

.. argparse::
   :module: kpimpc.kpi_WV_nrcs.compute_kpi_1b
   :func: get_parser
   :prog: kpinrcs

kpihs (KPI-1d)
--------------

Reads the WV L2F daily files (:data:`kpimpc.config.DIR_L2F_WV_DAILY`) of the
analysed month and of the 3 months before. Output version directory:
``v8percentile95``.

.. argparse::
   :module: kpimpc.kpi_WV_hs.compute_kpi_1d_v8
   :func: get_parser
   :prog: kpihs

kpipush (input files to the ECE)
--------------------------------

See :doc:`operations` for the file naming and the network requirements.

.. argparse::
   :module: kpimpc.ece_push
   :func: get_parser
   :prog: kpipush
