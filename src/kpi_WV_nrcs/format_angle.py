from numpy import mod, where


def format_angle(angle, compass=False):
    """

    method to format angle in [-180,180] or [0,360] if compass=True

    :Parameters:
    - angle: np.ndarray or float angle in degrees to be formatted
    - compass: bool, if True the angle is in [0,360]

    :Returns:
    - theta: np.ndarray or float, formatted angle in degrees
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
    except IndexError:
        if theta < -180:
            theta += 360

        if theta > 180:
            theta -= 360

        if compass:
            if theta < 0:
                theta += 360
    finally:
        return theta
