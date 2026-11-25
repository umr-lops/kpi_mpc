kpimpc
======

Key Performance Indicators (KPI) of the Sentinel-1 Mission Performance Centre
(MPC) for the Wave Mode (WV), computed at Ifremer LOPS:

KPI-1b
   WV VV NRCS (denoised) bias with respect to the CMOD5n prediction
   (ECMWF wind), command ``kpinrcs``.

KPI-1d
   WV VV effective significant wave height (Hs) bias with respect to WAVEWATCH
   III, command ``kpihs``.

For each unit (S1A, S1B, S1C, S1D) and WV incidence angle (``wv1``,
``wv2``), the KPI is the percentage of the measurements of the analysed month
whose difference to the reference lies inside an envelope. The envelope is the
95th percentile of the absolute difference over the 3 months before the
analysed month.

The input files of both KPIs are also pushed to the MPC ECE server (command
``kpipush``), so that the MPC can recompute the KPIs.

.. toctree::
   :maxdepth: 2

   installation
   usage
   configuration
   operations
   api
   changelog

Source code: https://github.com/umr-lops/kpi_mpc
