import datetime
import importlib
import os
import pathlib
import subprocess

import pytest

from kpimpc import config, ece_push

GEN = datetime.datetime(2026, 11, 25, 6, 30, 0)


def test_ece_filename_l2f():
    path = "/x/2026/326/S1C_WV_L2F_enriched_LOPS_20261122_daily_IPF_004.03.nc"
    assert ece_push.ece_filename(path, GEN) == (
        "SAR_MPC_S1C_WVHS-WW3_V20261122T000000_20261123T000000"
        "_G20261125T063000.nc"
    )


def test_ece_filename_l2f_end_of_year():
    path = "/x/S1D_WV_L2F_enriched_LOPS_20261231_daily_IPF_004.03.nc"
    assert "_V20261231T000000_20270101T000000_" in ece_push.ece_filename(
        path, GEN
    )


def test_ece_filename_calibration():
    path = config.INPUT_files % "S1D"
    assert ece_push.ece_filename(path, GEN) == (
        "SAR_MPC_S1D_WVRCS-ECMWF_V20150101T000000_20261125T063000"
        "_G20261125T063000.nc"
    )


def test_ece_filename_unknown():
    with pytest.raises(ValueError):
        ece_push.ece_filename("/x/S1C_something_else.nc", GEN)


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """Reload the config on a fake data tree."""
    monkeypatch.setenv("KPIMPC_DATA_ROOT", str(tmp_path))
    importlib.reload(config)
    yield tmp_path
    monkeypatch.delenv("KPIMPC_DATA_ROOT")
    importlib.reload(config)


def test_data_root_env(data_root):
    expected = data_root.joinpath(
        "cache", "project", "mpc-sentinel1", "analysis", "s1_data_analysis"
    ).joinpath("L2_full_daily", "0.8")
    assert pathlib.Path(config.DIR_L2F_WV_DAILY) == expected


def test_list_input_files_kpi1d(data_root):
    for day, doy in (("20261121", "325"), ("20261122", "326")):
        daydir = pathlib.Path(config.DIR_L2F_WV_DAILY, "2026", doy)
        daydir.mkdir(parents=True)
        for sat in ("S1A", "S1C", "S1D"):
            (
                daydir
                / f"{sat}_WV_L2F_enriched_LOPS_{day}_daily_IPF_004.03.nc"
            ).touch()
    files = ece_push.list_input_files(
        "kpi1d",
        datetime.datetime(2026, 11, 20),
        datetime.datetime(2026, 11, 22),
    )
    names = [os.path.basename(f)[:12] for f in files]
    # S1A is not an active unit any more; day 20261120 is missing
    assert names == ["S1C_WV_L2F_e", "S1D_WV_L2F_e"] * 2


def test_list_input_files_kpi1b(data_root):
    os.makedirs(config.DIR_INPUT)
    open(config.INPUT_files % "S1C", "w").close()
    files = ece_push.list_input_files("kpi1b", None, None)
    assert files == [config.INPUT_files % "S1C"]  # S1D missing -> warning


def test_sftp_batch_hidden_then_rename():
    path = "/x/S1C_WV_L2F_enriched_LOPS_20261122_daily_IPF_004.03.nc"
    batch, transfers = ece_push.sftp_batch([path], GEN, remote_dir="upload")
    lines = batch.splitlines()
    assert lines[0] == (
        f'put "{path}" "upload/.S1C_WV_L2F_enriched_LOPS_20261122_daily_IPF_004.03.nc"'
    )
    assert lines[1].startswith(
        'rename "upload/.S1C_WV_L2F_enriched_LOPS_20261122_daily_IPF_004.03.nc"'
        ' "upload/SAR_MPC_S1C_WVHS-WW3_'
    )
    assert transfers[0][0] == path


def test_sftp_command_proxy():
    cmd = ece_push.sftp_command(
        destination="u@h", key="/k", proxy_command="ncat --proxy p:1 %h %p"
    )
    assert cmd[-1] == "u@h"
    assert "ProxyCommand=ncat --proxy p:1 %h %p" in cmd
    assert cmd[cmd.index("-i") + 1] == "/k"
    assert "BatchMode=yes" in cmd


def test_push_files_dry_run(tmp_path, monkeypatch):
    path = tmp_path / "S1C_WV_L2F_enriched_LOPS_20261122_daily_IPF_004.03.nc"
    path.write_bytes(b"0" * 10)

    def boom(*a, **k):
        raise AssertionError("sftp must not run in dry-run")

    monkeypatch.setattr(subprocess, "run", boom)
    assert ece_push.push_files([str(path)], dry_run=True) == 1


def test_push_files_failure_raises(tmp_path, monkeypatch):
    path = tmp_path / "S1C_WV_L2F_enriched_LOPS_20261122_daily_IPF_004.03.nc"
    path.write_bytes(b"0")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(kw["input"])
        return subprocess.CompletedProcess(
            cmd, 255, "", "Connection timed out"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="1/1 file"):
        ece_push.push_files([str(path)], retries=1, retry_delay=0)
    assert len(calls) == 2


def test_entrypoint_dry_run(data_root):
    assert (
        ece_push.entrypoint(
            [
                "--kpi",
                "kpi1d",
                "--startdate",
                "20261120",
                "--stopdate",
                "20261122",
                "--dry-run",
            ]
        )
        == 0
    )
