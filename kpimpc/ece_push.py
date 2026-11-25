"""Push the KPI-1b / KPI-1d input files to the MPC ECE (CLS) sftp server.

Replaces push_wrapper_singularity.bash + push_many_input_files_to_MPC_ECE.py +
push_input_files_to_MPC_ECE.bash (script given by CLS, July 2022).

* KPI-1b (NRCS): the aggregated WV SLC calibration file of each unit
  ``L1_v16/WV/<sat>_wv_ocean_calibration_..._20150101_today_runv16.nc``,
  pushed as ``SAR_MPC_<sat>_WVRCS-ECMWF_V20150101T000000_<G>_G<G>.nc``
* KPI-1d (Hs): the WV L2F daily files ``L2_full_daily/<version>/YYYY/DDD/*.nc``,
  pushed as ``SAR_MPC_<sat>_WVHS-WW3_V<day>T000000_<day+1>T000000_G<G>.nc``

(<G> = generation date, i.e. the push time, ``%Y%m%dT%H%M%S``.)

Each file is uploaded under a hidden name then renamed, so that the ECE never
reads a partial file. sftp runs in batch mode: any failed transfer makes the
command fail (the 2022 bash script ignored sftp errors).

usage::

    kpipush --kpi kpi1b --startdate 20261120 --stopdate 20261125 --dry-run
    kpipush --kpi kpi1d --startdate 20261120 --stopdate 20261125
    kpipush --check-connection
"""

import argparse
import datetime
import glob
import logging
import os
import shlex
import subprocess
import sys
import time

from kpimpc import config

KPI_CHOICES = ("kpi1b", "kpi1d")
GENERATION_FORMAT = "%Y%m%dT%H%M%S"


def list_input_files(kpi, start, stop, satellites=None):
    """List the local files to push.

    :param kpi: str kpi1b or kpi1d
    :param start: datetime.datetime first day (kpi1d only)
    :param stop: datetime.datetime last day, included (kpi1d only)
    :param satellites: list of units (default: config.ACTIVE_SAR_UNITS)
    :return: list of absolute paths, sorted
    """
    if satellites is None:
        satellites = config.ACTIVE_SAR_UNITS
    files = []
    if kpi == "kpi1b":
        # one aggregated file per unit, re-pushed at each run (it grows)
        for sat in satellites:
            path = config.INPUT_files % sat
            if os.path.exists(path):
                files.append(path)
            else:
                logging.warning("missing KPI-1b input file %s", path)
    elif kpi == "kpi1d":
        day = start
        while day <= stop:
            daydir = os.path.join(
                config.DIR_L2F_WV_DAILY, day.strftime("%Y"), day.strftime("%j")
            )
            for sat in satellites:
                files += sorted(glob.glob(os.path.join(daydir, sat + "_*.nc")))
            day += datetime.timedelta(days=1)
    else:
        raise ValueError(f"no such kpi: {kpi}")
    logging.info("%s: %d file(s) to push", kpi, len(files))
    return files


def ece_filename(path, generation_date):
    """Name of the file on the ECE server (convention of the CLS script).

    :param path: str local path of a KPI input file
    :param generation_date: datetime.datetime
    :return: str
    """
    filename = os.path.basename(path)
    parts = filename.split("_")
    sat = parts[0]
    gen = generation_date.strftime(GENERATION_FORMAT)
    if "_L2F_" in filename:
        # S1C_WV_L2F_enriched_LOPS_20261122_daily_IPF_004.03.nc
        day = datetime.datetime.strptime(parts[5], "%Y%m%d")
        nextday = day + datetime.timedelta(days=1)
        return f"SAR_MPC_{sat}_WVHS-WW3_V{day.strftime(GENERATION_FORMAT)}_{nextday.strftime(GENERATION_FORMAT)}_G{gen}.nc"
    if "_wv_ocean_calibration_" in filename:
        return f"SAR_MPC_{sat}_WVRCS-ECMWF_V20150101T000000_{gen}_G{gen}.nc"
    raise ValueError(f"not a KPI input file: {path}")


def sftp_command(
    destination=None, key=None, proxy_command=None, connect_timeout=30
):
    """sftp command line (list), reading its commands from stdin."""
    destination = destination or config.ECE_SFTP_DESTINATION
    key = key or config.ECE_SSH_KEY
    if proxy_command is None:
        proxy_command = config.ECE_PROXY_COMMAND
    cmd = [
        "sftp",
        "-q",
        "-b",
        "-",
        "-i",
        key,
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"ConnectTimeout={connect_timeout}",
    ]
    if proxy_command:
        cmd += ["-o", "ProxyCommand=" + proxy_command]
    return cmd + [destination]


def sftp_batch(files, generation_date, remote_dir=None):
    """sftp batch script: put each file under a hidden name, then rename it.

    :return: (str batch script, list of (local path, remote name))
    """
    remote_dir = remote_dir or config.ECE_REMOTE_DIR
    lines, transfers = [], []
    for path in files:
        dest = ece_filename(path, generation_date)
        tmp = remote_dir + "/." + os.path.basename(path)
        lines.append(f"put {_quote(path)} {_quote(tmp)}")
        lines.append(
            "rename {} {}".format(_quote(tmp), _quote(remote_dir + "/" + dest))
        )
        transfers.append((path, dest))
    return "\n".join(lines) + "\n", transfers


def _quote(path):
    # sftp batch files accept double-quoted arguments
    return '"' + path.replace('"', '\\"') + '"'


def push_files(files, dry_run=False, retries=2, retry_delay=60, **sftp_kw):
    """Upload the files to the ECE, one sftp session per file.

    :return: int number of files pushed (or that would be pushed)
    :raises RuntimeError: when at least one file could not be pushed
    """
    failed = []
    for ii, path in enumerate(files):
        batch, transfers = sftp_batch(
            [path], datetime.datetime.now(datetime.timezone.utc)
        )
        size_mb = os.path.getsize(path) / 1e6
        logging.info(
            "%d/%d %s (%.1f MB) -> %s",
            ii + 1,
            len(files),
            path,
            size_mb,
            transfers[0][1],
        )
        if dry_run:
            continue
        for attempt in range(retries + 1):
            t0 = time.time()
            res = subprocess.run(
                sftp_command(**sftp_kw),
                input=batch,
                text=True,
                capture_output=True,
            )
            if res.returncode == 0:
                logging.info(
                    "  pushed in %.1f s (%.1f MB/s)",
                    time.time() - t0,
                    size_mb / max(time.time() - t0, 1e-3),
                )
                break
            logging.warning(
                "  sftp failed (attempt %d/%d, code %d): %s",
                attempt + 1,
                retries + 1,
                res.returncode,
                res.stderr.strip(),
            )
            if attempt < retries:
                time.sleep(retry_delay)
        else:
            failed.append(path)
    if failed:
        raise RuntimeError(
            f"{len(failed)}/{len(files)} file(s) not pushed: {failed}"
        )
    return len(files)


def check_connection(**sftp_kw):
    """Open an sftp session and list the remote directory.

    :return: bool
    """
    cmd = sftp_command(**sftp_kw)
    logging.info("check connection: %s", shlex.join(cmd))
    res = subprocess.run(
        cmd,
        input=f"ls {config.ECE_REMOTE_DIR}\n",
        text=True,
        capture_output=True,
    )
    if res.returncode == 0:
        logging.info("connection to the ECE OK")
        return True
    logging.error(
        "connection to the ECE failed (code %d): %s",
        res.returncode,
        res.stderr.strip(),
    )
    return False


def _parse_day(value):
    return datetime.datetime.strptime(value.replace("-", ""), "%Y%m%d")


def get_parser():
    parser = argparse.ArgumentParser(
        prog="kpipush",
        description="push the KPI-1b / KPI-1d input files to the MPC ECE",
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--kpi", choices=KPI_CHOICES)
    parser.add_argument("--startdate", type=_parse_day, help="YYYYMMDD")
    parser.add_argument(
        "--stopdate", type=_parse_day, help="YYYYMMDD (included)"
    )
    parser.add_argument(
        "--satellites",
        default=",".join(config.ACTIVE_SAR_UNITS),
        help="comma-separated units [default: %(default)s]",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list the files and their ECE names, push nothing",
    )
    parser.add_argument(
        "--check-connection",
        action="store_true",
        help="only test the sftp connection to the ECE",
    )
    parser.add_argument("--retries", type=int, default=2)
    return parser


def entrypoint(argv=None):
    args = get_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
    )
    if args.check_connection:
        return 0 if check_connection() else 1
    if args.kpi is None:
        get_parser().error("--kpi is required (or --check-connection)")
    stop = args.stopdate or datetime.datetime.now().replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    start = args.startdate or stop - datetime.timedelta(days=5)
    if stop < start:
        get_parser().error("stopdate is before startdate")
    sats = [s.strip().upper() for s in args.satellites.split(",") if s.strip()]
    t0 = time.time()
    files = list_input_files(args.kpi, start, stop, satellites=sats)
    nb = push_files(files, dry_run=args.dry_run, retries=args.retries)
    logging.info(
        "%s %d file(s) in %.1f s",
        "would push" if args.dry_run else "pushed",
        nb,
        time.time() - t0,
    )
    return 0


if __name__ == "__main__":
    sys.exit(entrypoint())
