import os
import re
import numpy
import sys
from . import slurm
import subprocess
import logging


def run_lammps(phase, iteration, lmpdata, task, options):
    ''' Runs either a pressure correction or sampling LAMMPS run. '''
    bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../bin')
    lmp_exe = os.path.join(bin_dir, 'lmp_mpi')
    if not lmpdata:
        logging.error('No CG data files found.')
        sys.exit(1)
    jobs = {}
    pattern = options.lmpdata_path[phase].replace('*', '[^.]*(.)[^\d]*')
    for data in lmpdata:
        system_number = int(re.match(pattern, data).group(1))
        log_path = 'log/{}/{}-s{}-i{:02d}.log'.format(phase,task, system_number, iteration)
        if check_lammps_completion(log_path) and task == 'sample':
            # TODO - this case should be removed.
            logging.info(f'---- Already sampled {phase} {system_number}, skipping.')
            continue
        input_file = make_lammps_input(iteration, phase, system_number, task, data, options)
        lmp_cmd = f'{lmp_exe} -i {input_file} -log {log_path}'
        # If we are on a local machine, just run the jobs.
        if not options.cluster_mode:
            lmp_cmd = f'mpiexec -n {options.lmp_ntasks} ' + lmp_cmd
            out = subprocess.run(lmp_cmd.split(), capture_output=1, text=1)
            if out.returncode != 0:
                logging.error('LAMMPS failed to run!')
                sys.exit(1)
        else:
            logging.info('---- Cluster mode on - submitting lammps job.')
            options.slurm['lmp']['o'] = f'{task}.{phase}.{system_number}.{iteration}.slurm'
            options.slurm['lmp']['j'] = f'{task}-{system_number}'
            jobs[slurm.submit_job('lmp', lmp_cmd, options.slurm['lmp'])] = 'lammps'
    if jobs:
        slurm.wait_on_jobs(jobs)


def make_lammps_input(iteration, phase, system_number, task, data, options):
    ''' Make input file for lammps. '''
    # Read task's script from its template
    lmp_script = read_task_template(task)
    if task == 'pressure':
        logging.debug(f'Pressure correction for {phase} system {system_number}')
    else:
        logging.debug(f'Sampling {phase} system {system_number}')
    input_path = 'in_scripts/{}/in.{}.lammps'.format(phase, system_number)
    data_path = 'cg-data/{}'.format(phase)
    dump_path = 'dump/{}'.format(phase)
    # Forcefield settings for pair, bond, angle
    pair_style, pair_coeff = construct_pair_settings(iteration, options)
    bond_style, bond_coeff = construct_bond_settings(iteration, options)
    angle_style, angle_coeff = construct_angle_settings(iteration, options)

    # Write LAMMPS input script
    with open(input_path, 'w') as f:
        f.write(lmp_script.format(i=iteration, lmp_style=options.lmp_style,
                pair_style = pair_style, pair_coeff=pair_coeff,
                bond_style = bond_style, bond_coeff=bond_coeff,
                angle_style=angle_style, angle_coeff=angle_coeff,
                data=data, sys=system_number, sb=options.lmp_special_bond,
                ts=options.timestep, nve_ts=options.nve_step,
                dump_path=dump_path, data_path=data_path))
    return input_path


def read_task_template(task):
    ''' For the defined task read the corresponding template. '''
    if task == 'sample':
        lmp_script = open(os.path.join(
            os.path.dirname(__file__), '../templates/lammps.cg_sampling.in')).read()
    elif task == 'pressure':
        lmp_script = open(os.path.join(
            os.path.dirname(__file__), '../templates/lammps.pressure_correction.in')).read()
    else:
        logging.info(f'---- Unknown LAMMPS run type: {task}')
        sys.exit(1)
    return lmp_script


def construct_pair_settings(iteration, options):
    ''' construct pair style, type, and coeffs for table style. '''
    rdf_range = options.sample_cutoff['rdf']['end'] - options.sample_cutoff['rdf']['beg']
    rdf_N = int(rdf_range / options.sample_cutoff['rdf']['inv'])
    pair_style = 'pair_style table linear {}'.format(rdf_N)
    pair_table = 'pair_coeff {b1} {b2} potentials/pair.table.{pair_type}.{i} {pair_type}\n'
    pair_coeff = ''
    pair_type = None
    for b1 in options.beads.keys():
        for b2 in options.beads.keys():
            if b2 >= b1:
                pair_type = options.beads[b1]['type'] + options.beads[b2]['type']
                pair_coeff += pair_table.format(b1=b1, b2=b2, i=iteration,
                    pair_type=pair_type)
    return pair_style, pair_coeff


def construct_bond_settings(iteration, options):
    ''' construct bond style, type, and coeffs for table style. '''
    bdf_range = options.sample_cutoff['bdf']['end'] - options.sample_cutoff['bdf']['beg']
    bdf_N = int(bdf_range / options.sample_cutoff['bdf']['inv'])
    bond_style = 'bond_style table linear {}'.format(bdf_N)
    bond_table = 'bond_coeff {b_id} potentials/bond.table.{b_type}.{i} {b_type}\n'
    bond_coeff = ''
    for b in options.bonds.items():
        bond_coeff += bond_table.format(b_id=b[1]['id'],
                                        b_type=b[1]['type'], i=iteration)
    return bond_style, bond_coeff


def construct_angle_settings(iteration, options):
    ''' construct angle style, type, and coeffs for table style. '''
    adf_range = options.sample_cutoff['adf']['end'] - options.sample_cutoff['adf']['beg']
    adf_N = int(adf_range / options.sample_cutoff['adf']['inv'])
    angle_style = 'angle_style  table linear {}'.format(adf_N)
    angle_table = 'angle_coeff {a_id} potentials/angle.table.{a_type}.{i} {a_type}\n'
    angle_coeff = ''
    for a in options.angles.items():
        angle_coeff += angle_table.format(a_id=a[1]['id'],
            a_type=a[1]['type'], i=iteration)
    return angle_style, angle_coeff


def write_potential_table(phase, r, e, f, poten_type, pair_type, iteration):
    ''' Write the table potential into a file in the following path. '''
    if phase == 'avg':
        path = 'potentials/{}.table.{}.{}'.format(poten_type, pair_type, iteration)
    else:
        path = f'potentials/{phase}/{poten_type}.table.{pair_type}.{iteration}'
    fid = open(path,'w')
    fid.write('{}\n'.format(pair_type))
    if poten_type == 'pair':
        fid.write('N {} R {} {}\n\n'.format(len(r),min(r),max(r)))
    elif poten_type == 'bond':
        fid.write('N {}\n\n'.format(len(r)))
    elif poten_type == 'angle':
        if r[0] != 0.0:
            r[0] = 0.0
        if r[-1] != 180.0:
            r[-1] = 180.0
        fid.write('N {}\n\n'.format(len(r)))
    elif poten_type == 'dihedral':
        fid.write('N {} DEGREES\n\n'.format(len(r)))
    for i, (ri, ei, fi) in enumerate(zip(r,e,f), 1):
        fid.write(f'{i} {ri:.4f} {ei:.6g} {fi:.6g}\n')


def write_potential_updates(phase, r, dU, potential_type, pair_type, iteration):
    ''' Write the updates Delta U of each iteration into a file. '''
    # Path to the location to store the file
    path = 'cg-updates/{}/{}.update.{}.{}'.format(phase, potential_type, pair_type, iteration)
    fid = open(path, 'w')
    for i, (rr, du) in enumerate(zip(r, dU)):
        fid.write('{:4d}\t{:6.4f}\t{:6.4f}\n'.format(i, rr, du))


def read_tabular_potential(phase, poten_type, pair_type, iteration):
    if phase == 'avg':
        path = 'potentials/{}.table.{}.{}'.format(poten_type, pair_type, iteration)
    else:
        path = 'potentials/{}/{}.table.{}.{}'.format(phase, poten_type, pair_type, iteration)
    return numpy.genfromtxt(path, usecols=(1,2,3), skip_header=3).T


def write_initial_coeff_file(dis_type, options):
    '''write initial bond/angle potential coefficients to file for lammps to read'''
    if dis_type == 'bond':
        with open('in_scripts/bond_coeff', 'w') as f:
            bond_table = 'bond_coeff      {b_id} {k} {x0}\n'
            for b in options.bonds.iterkeys():
                f.write(bond_table.format(b_id=b,
                    k=options.bonds[b]['k'], x0=options.bonds[b]['x0']))
    if dis_type == 'angle':
        with open('in_scripts/angle_coeff', 'w') as f:
            angle_table = 'angle_coeff      {b_id}  {k} {x0}\n'
            for b in options.angles.iterkeys():
                f.write(angle_table.format(b_id=b,
                    k=options.angles[b]['k'], x0=options.angles[b]['x0']))


def check_lammps_completion(logpath):
    if not os.path.exists(logpath):
        return False
    with open(logpath, 'rb') as f:
        try:
            # If the LAMMPS run was successful, it should end with:
            # Total wall time: XX:XX:XX
            pos = f.seek(-30, 2)
        except:
            return False
        return 'Total wall time: ' in str(f.read())

