#!/home1/datawork/agrouaze/conda_envs2/envs/py2.7_cwave/bin/python
# coding: utf-8
"""
"""
import sys
print(sys.executable)
import subprocess
import logging
import calendar
from dateutil import rrule
import datetime
import getpass
if __name__ == '__main__':
    root = logging.getLogger ()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler (handler)
    import argparse
    parser = argparse.ArgumentParser (description='start prun')
    parser.add_argument ('--verbose',action='store_true',default=False)
    parser.add_argument('--mode', choices=['range_dates','monthly'], default='range_dates',
                        help='mode : range_dates (last 20 days) or monthly (last 12 months)')
    args = parser.parse_args ()
    if args.verbose:
        logging.basicConfig (level=logging.DEBUG,format='%(asctime)s %(levelname)-5s %(message)s',
                             datefmt='%d/%m/%Y %H:%M:%S')
    else:
        logging.basicConfig (level=logging.INFO,format='%(asctime)s %(levelname)-5s %(message)s',
                             datefmt='%d/%m/%Y %H:%M:%S')
    prunexe = '/appli/prun/bin/prun'
    if getpass.getuser() == 'agrouaze':
        listing = '/home1/scratch/agrouaze/list_kpi_1d_v2_prun_test.txt' # written below
        pbs = '/home1/datahome/agrouaze/git/kpi_mpc/src/kpi_WV_hs/compute_kpi_1d_v2.pbs'
    else:
        listing = '/home1/scratch/satwave/list_kpi_1d_v2_prun.txt'  # written below
        pbs = '/home1/datahome/satwave/sources_en_exploitation2/kpi_mpc/src/kpi_WV_hs/compute_kpi_1d_v2.pbs'
    # call prun
    opts = ' --split-max-lines=3 --background -e '
    listing_content = []
    if args.mode == 'range_dates':
        #sta = datetime.datetime(2015,1,1)
        #sta = datetime.datetime(2019,6,1)

        #sta = datetime.datetime(2020,6,1) # pour test 2 qui utilisent les cross assignments de partitions

        sto = datetime.datetime.today()
        sta = sto - datetime.timedelta(days=20)
        logging.info('start year: %s', sta)
        #sto = datetime.datetime(2020,9,1)
        fid = open(listing,'w')
        cpt = 0
        for unit in ['S1A','S1B']:
            for wv in ['wv1','wv2']:
                logging.info('%s',unit)
                for dd in rrule.rrule(rrule.DAILY,dtstart=sta,until=sto):
                    fid.write('%s %s %s\n'%(unit,wv,dd.strftime('%Y%m%d')))
                    cpt +=1
        fid.close()
    elif args.mode=='monthly': # last day of each month is used to compute the KPI on the last 12 months
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
    else:
        raise Exception("mode unknown")
    logging.info('listing written ; %s nb lines: %s',listing,cpt)

    cmd = prunexe+opts+pbs+' '+listing
    logging.info('cmd to cast = %s',cmd)
    st = subprocess.check_call(cmd,shell=True)
    logging.info('status cmd = %s',st)
