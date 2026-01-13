# tests/test_compute_kpi_1d.py
import datetime
from unittest.mock import patch

import numpy as np
import xarray as xr
import pytest

from kpimpc.kpi_WV_hs.compute_kpi_1d_v8 import compute_kpi_1d


def make_dataset():
    times = np.array(
        [
            np.datetime64("2021-01-20T00:00:00"),  # prior
            np.datetime64("2021-03-20T00:00:00"),  # prior
            np.datetime64("2021-04-20T00:00:00"),  # current
            np.datetime64("2021-04-25T00:00:00"),  # current
        ],
        dtype="datetime64[s]",
    )
    ds = xr.Dataset(
        {
            "fdatedt": (("fdatedt",), times),
            "ww3_effective_2Dcutoff_hs": (("fdatedt",), np.array([2.0, 3.0, 4.0, 5.0])),
            # two columns: first - second = bias; prior biases = 0.5, 0.6 -> envelope large
            # current biases = 0.2, 0.3 -> both should be inside envelope -> KPI 100%
            "oswXA_hs_ww3spec_firstSARpartition": (
                ("fdatedt", "two"),
                np.array([[1.4, 0.9], [1.7, 1.1], [1.2, 1.0], [1.3, 1.0]]),
            ),
            "oswIncidenceAngle": (("fdatedt",), np.array([20, 25, 20, 25])),
            "pol": (("fdatedt",), np.array(["vv", "vv", "vv", "vv"])),
            "oswLandFlag": (("fdatedt",), np.zeros(4, dtype=int)),
            "dist2coastKM": (("fdatedt",), np.array([200, 200, 200, 200])),
            "oswLat": (("fdatedt",), np.array([10, 10, 10, 10])),
            "s1_effective_hs_2Dcutoff": (("fdatedt",), np.array([1.0, 1.1, 1.2, 1.3])),
        }
    )
    return ds


def test_compute_kpi_with_mocked_reader():
    ds = make_dataset()
    stop_date = datetime.datetime(2021, 5, 15)

    # patch the reader imported in the module; return a dict mapping satellite -> dataset
    with patch("kpimpc.kpi_WV_hs.compute_kpi_1d_v8.read_L2F_with_xarray") as mock_read:
        mock_read.return_value = {"S1A": ds}
        kpi, start, stop, envelop, nb, mean_bias, std = compute_kpi_1d(
            "S1A", "wv1", stop_analysis_period=stop_date, ds=None
        )

    assert kpi == pytest.approx(100.0)
    assert nb == 2
    assert mean_bias == pytest.approx(0.25)  # mean of 0.2 and 0.3
    assert std == pytest.approx(np.std([0.2, 0.3]))