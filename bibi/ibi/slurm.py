import time
import socket
import logging
from subprocess import Popen, PIPE
import re
import sys
import os
from math import ceil
from glob import glob
from . import lammps


def submit_job(job_type, cmd, slurm):
    ''' Submit the jub through the slurm to run. '''
    # Make the script using the inputs
    input_file = make_slurm_input(job_type, cmd, slurm)
    p = Popen(['sbatch', f'{input_file}'], stdin=PIPE, stdout=PIPE, stderr=PIPE
                                                     , encoding = 'utf8')
    stdout, stderr = p.communicate(input=input_file)
    if stderr:
        logging.error(f'Slurm submission error! {stderr}')
        sys.exit(1)
    m = re.search('Submitted batch job (\d+)\s*', stdout)
    return m.group(1) if m else None


def make_slurm_input(job_type, cmd, slurm):
    ''' Submits a single job and returns the jod id. '''
    input_path = 'in_scripts/slurm.sh'
    if job_type == 'lmp':
        cgd = ''
    elif job_type == 'cgd':
        cgd = 'cd cg-dist\n'
        cgd += f'export OMP_NUM_THREADS={slurm["c"]}\n'
    else:
        logging.error('Unknown job type, please check.')
        sys.exit(1)
    # Implemented only for Agave and SOL clusters of ASU using Slurm
    if 'agave' in socket.gethostname():
        script = open(os.path.join(os.path.dirname(__file__),
                                        '../templates/agave.slurm.sh')).read()
    elif 'sol' in socket.gethostname():
        script = open(os.path.join(os.path.dirname(__file__),
                                        '../templates/sol.slurm.sh')).read()
    else:
        logging.info('---- Cluster hostname is not recognized!')
        sys.exit(1)

    input_path = f'in_scripts/{job_type}.slurm.sh'
    with open(input_path, 'w') as fid:
        fid.write(script.format(N=slurm['N'], c=slurm['c'], n=slurm['n'],
                                m=slurm['m'], p=slurm['p'], q=slurm['q'],
                                t=slurm['t'], o=slurm['o'], j=slurm['j'],
                                cgd=cgd, cmd=cmd))
    return input_path


def wait_on_jobs(job_list):
    ''' Sleeps all job ids in the list job_list are completed. '''
    not_done_states = ['PD', 'R', 'CF', 'CG']
    logging.info(f'---- Waiting on {len(job_list)} jobs to complete ... {sorted(job_list.keys())}')
    while True:
        out,_ = Popen(['squeue'], stdout=PIPE, encoding = 'utf8').communicate()
        jobs = [s.split() for s in out.split('\n')[1:] if s.strip()]
        running = [j[0] for j in jobs if j[4] in not_done_states]
        if all(j not in running for j in job_list):
            break
        time.sleep(15)
    for jobid, jobtype in job_list.items():
        if jobtype == 'lammps':
            if not lammps.check_lammps_completion(glob(f'slurms/{jobid}.*.slurm')[0]):
                logging.info(f'---- Lammps job {jobid} failed!')
                sys.exit(1)
        elif jobtype == 'cgd':
            if not check_cg_complete(glob(f'slurms/{jobid}.dist*.slurm')[0]):
                logging.info(f'---- cg-distributions job {jobid} failed!')
                sys.exit(1)
    logging.info(f'---- {len(job_list)} jobs completed.')


def check_cg_complete(logpath):
    if not os.path.exists(logpath): return False
    with open(logpath, 'r') as f:
        lines = f.readlines()
        if not lines:
            return False
        for line in lines:
            if line.startswith('Finished'):
                return True
        return False

