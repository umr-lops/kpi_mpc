import numpy as np
from numpy import *


def format_angle(angle, compass=False):
    """

    :Parameters:
    - angle:
    - compass:
    """
    theta = mod(angle, 360.0)
    try:
        # on test un array
        idx = where(theta < -180)
        if len(idx) > 0:
            theta[idx] += 360
        idx = where(theta > 180)
        if len(idx) > 0:
            theta[idx] -= 360

        if compass:
            idx = where(theta < 0)
            if len(idx) > 0:
                theta[idx] += 360
    except:
        if theta < -180:
            theta += 360

        if theta > 180:
            theta -= 360

        if compass:
            if theta < 0:
                theta += 360
    finally:
        return theta
