"""Command-line parsers of kpinrcs and kpihs (also used by the docs)."""

import pytest

from kpimpc.kpi_WV_hs.compute_kpi_1d_v8 import get_parser as kpihs_parser
from kpimpc.kpi_WV_nrcs.compute_kpi_1b import get_parser as kpinrcs_parser


@pytest.mark.parametrize(
    "get_parser,prog", [(kpinrcs_parser, "kpinrcs"), (kpihs_parser, "kpihs")]
)
def test_parser(get_parser, prog):
    parser = get_parser()
    assert parser.prog == prog
    args = parser.parse_args(
        ["--satellite", "S1C", "--wv", "wv2", "--enddate", "20261120"]
    )
    assert (args.satellite, args.wv, args.enddate) == (
        "S1C",
        "wv2",
        "20261120",
    )
    assert not args.overwrite
    with pytest.raises(SystemExit):
        parser.parse_args(["--satellite", "S1X", "--wv", "wv1"])
