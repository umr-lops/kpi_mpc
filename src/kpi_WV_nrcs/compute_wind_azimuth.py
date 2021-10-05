import numpy as np
from format_angle import format_angle
def method_wind_azi_range(wind_u,wind_v,trackangle):
    """
    agrouaze: January 2020
    code adapted from IPF owi_lop_application
    :return:
        param ancillary_wind_dir_range: in meteorological convention from range
    """
    # the wind direction in mathematical ref
    ext_ancillary_wind_direction = 90. - np.rad2deg(np.arctan2(wind_v, wind_u))
    # the the wind direction in meteorological convention (not from North, but where the wind blows from), starting from range direction
    ext_ancillary_wind_dir_range = -format_angle(90. + trackangle -ext_ancillary_wind_direction + 180.)
    return ext_ancillary_wind_dir_range