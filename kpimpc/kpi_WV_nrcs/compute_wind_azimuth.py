import numpy as np

from kpimpc.kpi_WV_nrcs.format_angle import format_angle


def method_wind_azi_range(wind_u, wind_v, trackangle):
    """
    agrouaze: January 2020
    code adapted from IPF owi_lop_application

    :Parameters:
        param wind_u: np.ndarray or float zonal wind component (from East) in m/s
        param wind_v: np.ndarray or float meridional wind component (from North) in m/s
        param trackangle: np.ndarray or float SAR track angle in degrees (from North)

    :return:
        param ancillary_wind_dir_range: in meteorological convention from range
    """
    # the wind direction in mathematical ref
    ext_ancillary_wind_direction = 90.0 - np.rad2deg(
        np.arctan2(wind_v, wind_u)
    )
    # the the wind direction in meteorological convention (not from North, but where the wind blows from), starting from range direction
    ext_ancillary_wind_dir_range = -format_angle(
        90.0 + trackangle - ext_ancillary_wind_direction + 180.0
    )
    return ext_ancillary_wind_dir_range
