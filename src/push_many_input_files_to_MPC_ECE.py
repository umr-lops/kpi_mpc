"""
Author: Antoine Grouazel
purpose: push KPI input files on ECE
creation: Sept 2022
"""
import subprocess
import logging
import glob
import time
import os
import datetime
from dateutil import rrule
from config import DIR_L2F_WV_DAILY,INPUT_files
import getpass
user = getpass.getuser()
if user=='agrouaze':
    bash_exe = '/home1/datahome/agrouaze/sources/git/kpi_mpc/src/push_input_files_to_MPC_ECE.bash'
elif user=='satwave':
    bash_exe = '/home1/datahome/satwave/sources_en_exploitation2/kpi_mpc/src/push_input_files_to_MPC_ECE.bash'
else:
    raise Exception('user %s is not handle'%user)
def list_input_files(kpi,start,stop):
    """

    :param kpi: str kpi1d or kpi1b
    :param start: datetime.datetime
    :param stop: datetime.datetime
    :return:
    """
    lst_files = []
    if kpi == 'kpi1d':
        logging.info('the version of WV L2F daily data push to ECE is %s',DIR_L2F_WV_DAILY)
        for dd in rrule.rrule(rrule.DAILY,dtstart=start,until=stop):
            pattern = os.path.join(DIR_L2F_WV_DAILY,dd.strftime('%Y'),dd.strftime('%j'),'*nc')
            lst_files += glob.glob(pattern)
    elif kpi == 'kpi1b':
        pattern = INPUT_files.replace('%s','*')
        logging.info('pattern 1B: %s',pattern)
        lst_files = glob.glob(pattern)
    else:
        raise Exception('no such kpi: %s',kpi)
    logging.info('Nb files found: %s',len(lst_files))
    if len(lst_files)>6:
        for ff in lst_files[0:3]+lst_files[-3:-1]:
            logging.debug('%s',ff)
    else:
        logging.debug('%s',lst_files)
    return lst_files

def push_to_ftp(lst_files):
    """
    :param lst_files: list of full path
    :return:
    """
    for ffi,ff in enumerate(lst_files):
        if ffi%20==0:
            logging.info('%s/%s %s',ffi,len(lst_files),ff)
        cmd = 'bash '+bash_exe+' '+ff
        logging.debug('cmd: %s',cmd)
        st = subprocess.check_output(cmd,shell=True)
        #st = 0
        logging.debug('status %s \n',st)

if __name__ == '__main__':
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    import argparse
    import resource

    parser = argparse.ArgumentParser(description='push FTP ECE')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--kpi', choices=['kpi1b', 'kpi1d'], required=True, help='kpi1b kpi1d')
    parser.add_argument('--startdate', help='YYYYMMDD', required=True, action='store',
                        default=None)
    parser.add_argument('--stopdate', help='YYYYMMDD', required=True, action='store',
                        default=None)
    args = parser.parse_args()

    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    else:
        logging.basicConfig(level=logging.INFO, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    t0 = time.time()
    a_listing = list_input_files(kpi=args.kpi, start=datetime.datetime.strptime(args.startdate,'%Y%m%d'),
                                 stop=datetime.datetime.strptime(args.stopdate,'%Y%m%d'))
    push_to_ftp(lst_files=a_listing)
    logging.info('done in %1.1f seconds',time.time()-t0)