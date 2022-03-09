#!/home1/datawork/agrouaze/conda_envs2/envs/py2.7_cwave/bin/python
# coding: utf-8
"""
"""
import sys
print(sys.executable)
import subprocess
import logging
import getpass
import calendar
from dateutil import rrule
import datetime
def write_listing_range_dates(args):
    # sta = datetime.datetime(2015,1,1)
    # sta = datetime.datetime(2019,6,1)

    # sta = datetime.datetime(2020,6,1) # pour test 2 qui utilisent les cross assignments de partitions

    #sto = datetime.datetime.today()
    #sta = sto - datetime.timedelta(days=20)
    logging.info('start year: %s %s', args.start,args.stop)
    # sto = datetime.datetime(2020,9,1)
    fid = open(listing, 'w')
    cpt = 0
    for unit in ['S1A', 'S1B']:
        for wv in ['wv1', 'wv2']:
            logging.info('%s', unit)
            for dd in rrule.rrule(rrule.DAILY, dtstart=args.start, until=args.stop):
                fid.write('%s %s %s\n' % (unit, wv, dd.strftime('%Y%m%d')))
                cpt += 1
    fid.close()
    return cpt

def write_listing_monthly():
    fid = open(listing, 'w')
    cpt = 0
    sto = datetime.datetime.today()
    sta = sto - datetime.timedelta(days=366)
    for unit in ['S1A', 'S1B']:
        for wv in ['wv1', 'wv2']:
            logging.info('%s', unit)
            for dd in rrule.rrule(rrule.MONTHLY, dtstart=sta, until=sto):
                start, stop = calendar.monthrange(dd.year, dd.month)
                dd2 = dd.replace(day=stop)
                fid.write('%s %s %s\n' % (unit, wv, dd2.strftime('%Y%m%d')))
                cpt += 1
    fid.close()
    return cpt

def fct_parse_date(s):
    return datetime.datetime.strptime(s, '%Y%m%d')

if __name__ == '__main__':
    root = logging.getLogger ()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler (handler)
    import argparse

    parser = argparse.ArgumentParser(description='start prun')
    parser.add_argument('--verbose', action='store_true', default=False)
    # parser.add_argument('--mode', choices=['range_dates','monthly'], default='range_dates',
    #                     help='mode : range_dates (last 20 days) or monthly (last 12 months)')
    subparsers = parser.add_subparsers()
    parser_range = subparsers.add_parser('range_dates')
    parser_range.set_defaults(func=write_listing_range_dates)
    parser_range.add_argument('--start', help='start date to analyze', required=True, type=fct_parse_date)
    parser_range.add_argument('--stop', help='stop date to analyze', required=True, type=fct_parse_date)
    parser_monthly = subparsers.add_parser('monthly')
    parser_monthly.set_defaults(func=write_listing_monthly)
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig (level=logging.DEBUG,format='%(asctime)s %(levelname)-5s %(message)s',
                             datefmt='%d/%m/%Y %H:%M:%S')
    else:
        logging.basicConfig (level=logging.INFO,format='%(asctime)s %(levelname)-5s %(message)s',
                             datefmt='%d/%m/%Y %H:%M:%S')
    prunexe = '/appli/prun/bin/prun'
    if getpass.getuser() == 'agrouaze':
        listing = '/home1/scratch/agrouaze/list_kpi_1b_prun.txt' # written below
        pbs = '/home1/datahome/agrouaze/git/kpi_mpc/src/kpi_WV_nrcs/compute_kpi_1b.pbs'
    else:
        listing = '/home1/scratch/satwave/list_kpi_1b_prun.txt'  # written below
        pbs = '/home1/datahome/satwave/sources_en_exploitation2/kpi_mpc/src/kpi_WV_nrcs/compute_kpi_1b.pbs'
    # call prun
    opts = ' --split-max-lines=3 --background -e '

    cpt = args.func(args)

    logging.info('listing written ; %s nb lines: %s',listing,cpt)

    cmd = prunexe+opts+pbs+' '+listing
    logging.info('cmd to cast = %s',cmd)
    st = subprocess.check_call(cmd,shell=True)
    logging.info('status cmd = %s',st)
