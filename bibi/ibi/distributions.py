import sys
import os
import re
from glob import glob
import numpy
import logging
from copy import deepcopy
from scipy.signal import savgol_filter
from . import lammps
from . import slurm
import subprocess


class distribution_function():
    ''' documentation '''
    def __init__(self):
        self.x = None
        self.p = None
        self.var = None


def average_all_targets(options, phase):
    ''' Given a dictionary of paths for each distribution type, compute the
    average of each target distribution for each distribution type and
    write to a file.  Returns file paths and the number of target samples. '''
    paths = collect_paths(options, phase)
    target_cache = {}
    num_files = {}
    for d in paths:
        target_cache[d], num_files[d] = average_single_target(options, phase, paths[d], d)
    # Check that # of target files is the same for all distributions and types.
    t0 = next(iter(num_files['rdf']))
    num_samples = num_files['rdf'][t0]
    for d in num_files:
        for t in num_files[d]:
            if num_files[d][t] != num_samples:
                logging.info(f'There are {num_samples} target rdf files of type {t0}, '\
                              'but {n} target {d} files of type {t}.')
                sys.exit(1)
    return target_cache, num_samples


def collect_paths(options, phase):
    ''' Collect paths of adf/bdf/rdf and other target distribution files. '''
    target = {t: sorted(glob(p)) for t, p in options.target_path[phase].items()}
    if (not target['adf'] or not target['bdf'] or not target['rdf']):
        print('Target rdf/bdf/adf files not found!')
        sys.exit(1)
    return target


def average_single_target(options, phase, files, dist_type):
    ''' Computes average target distributions of a specific phase and
    distribution type (e.g. RDF, ADF, or BDF). '''
    dist_files = {}
    for f in files:
        pair_type = re.search('_([A-Z]+)\.txt$', f).group(1)
        if pair_type not in dist_files:
            dist_files[pair_type] = []
        dist_files[pair_type].append(f)

    dist2pot = dict(rdf='pair', bdf='bond', adf='angle')
    df = dist2pot[dist_type]
    window  = options.smooth_factor[df]['window']
    polyorder = options.smooth_factor[df]['polyorder']

    target_cache = []
    num_target_files = {}
    logging.debug(f'---- Computing target {phase} {dist_type.upper()}s')
    for pair_type in dist_files:
        dists = [read_distribution_file(f) for f in dist_files[pair_type]]
        num_target_files[pair_type] = len(dists)
        d = distribution_function()
        d.x = dists[0].x
        d.p = numpy.average([x.p for x in dists], axis=0)
        d.var = numpy.var([x.p for x in dists], axis=0)
        d.p = savgol_filter(d.p, window, polyorder)
        d, regions = extrapolate_distribution(d, options.tolerance[df], return_regions=1)
        path = f'averaged_targets/{phase}/{dist_type}-{pair_type}.txt'
        write_distribution_file(path, d)
        target_cache.append(path)
    return target_cache, num_target_files


def extrapolate_distribution(p, p_min, return_regions=False):
    ''' Extrapolate the average target for values below the defined tolerance all
        the way to first encounter of zero using a quadratic function '''
    # Determine mask of values that are below the tolerance value.
    m = p.p < p_min
    regions = split_extrapolated_regions(m, min_sep=4)
    for r in regions:
        ii = r.interpolate_from
        f = numpy.polynomial.Polynomial.fit(p.x[ii], numpy.log(p.p[ii]), deg=2)
        if f.coef[-1] > 0.0:
            logging.warning('---- WARNING: distribution quadratic extrapolation diverges.')
            f = numpy.polynomial.Polynomial.fit(p.x[ii], numpy.log(p.p[ii]), deg=1)
        p.p[r.points] = numpy.exp(f(p.x[r.points]))
    if return_regions:
        return p, regions
    return p


class region:
    ''' Defines a range of consecutive points to be interpolated. '''
    def __init__(self):
        self.points = []
        self.interpolate_from = []


def split_extrapolated_regions(m, min_sep=1):
    ''' Given a boolean array, return indices of consecutive True values.
    min_sep>1 allows for gaps of more than 1 missing index. '''
    regions = []
    for i in numpy.flatnonzero(m):
        if not regions or i > regions[-1].points[-1] + min_sep:
            regions.append(region())
        regions[-1].points.append(i)
    new_regions = []
    for r in regions:
        # Fills in any gaps due to islands of size < min_sep.
        r.points = list(range(r.points[0], r.points[-1]+1))
        new_regions += set_region_interpolation(r, len(m))
    return new_regions


def set_region_interpolation(r, n):
    ''' Given a region, set the interpolation range, or if the region is
    internal to the domain, split it in two. '''
    if r.points[0] == 0:
        # Region is bounded at left boundary (must interpolate from right).
        i = r.points[-1]
        r.interpolate_from = list(range(i+1, i+4))
        return [r]
    elif r.points[-1] == n-1:
        # Region is bounded at right boundary (must interpolate from left).
        i = r.points[0]
        r.interpolate_from = list(range(i-3, i))
        return [r]
    else:
        # Region is entirely within domain, split in two parts.
        print('WARNING: distribution or potential has a middle invalid region.')
        left, right = [region() for _ in range(2)]
        left.points = r.points[:len(r.points)//2]
        i = r.points[0]
        left.interpolate_from = list(range(i+1, i+4))
        right.points = r.points[len(r.points)//2:]
        i = r.points[-1]
        right.interpolate_from = list(range(n-3, n))
        return [left, right]


def read_distribution_file(path):
    df = distribution_function()
    data = numpy.genfromtxt(path).T
    if len(data) == 3:
        df.x, df.p, df.var = numpy.genfromtxt(path, usecols=(0,1,2)).T
    elif len(data) == 2:
        df.x, df.p = numpy.genfromtxt(path, usecols=(0,1)).T
    return df


def write_distribution_file(path, df, var = True):
    f = open(path,'w')
    if var:
        for x , p, v in zip(df.x, df.p, df.var):
            f.write(f'{x} {p} {v}\n')
    else:
        for x, p in zip(df.x, df.p):
            f.write(f'{x} {p}\n')


def average_distributions(phase, target_files, iteration, options):
    ''' Computes averaged RDFs for all pair types at a given iteration. '''
    # Loops over the averaged target distribution files to get all CG types.
    for target in target_files:
        # Gets the distribution type (dt) and type of the interaction (tt).
        dt, tt = re.search('/(.df)-([A-Z]+)\.txt', target).groups()
        files = glob(f'cg-dist/{phase}/{dt}-sys*-{iteration}_{tt}.txt')
        pp = [read_distribution_file(p) for p in files]
        mean_p = distribution_function()
        mean_p.x = pp[0].x
        mean_p.p = numpy.average([x.p for x in pp], axis=0)
        mean_p.var = numpy.var([x.p for x in pp], axis=0)
        write_distribution_file(f'cg-dist/{phase}/{dt}-{iteration}_{tt}.avg', mean_p)


def compute_distribution(phase, target_cache, lmpdata, iteration, options):
    ''' Compute RDF of each systems and average RDF for next iteration'''
    bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../bin')
    cg_exe = os.path.join(bin_dir, 'cg-distributions')
    if not lmpdata:
        logging.error('No CG data files found.')
        sys.exit(1)
    jobs = {}
    pattern = options.lmpdata_path[phase].replace('*', '[^.]*(.)[^\d]*')
    for i, data in enumerate(lmpdata):
        logging.debug(f'Computing distributions of {phase} system {i+1}')
        system_number = int(re.match(pattern, data).group(1))
        skip_cg_dist = False
        if cg_job_complete(phase, system_number, iteration, options):
            logging.debug(f'Already computed distributions, skipping.')
            skip_cg_dist = True

        input_file = make_cg_distribution_input(iteration, phase, system_number, i, data, options)
        cg_cmd = f'{cg_exe} -i ../{input_file}'
        if not skip_cg_dist:
            if not options.cluster_mode:
                os.chdir('cg-dist')
                env = {'OMP_NUM_THREADS': str(options.cg_ntasks)}
                out = subprocess.run(cg_cmd.split(), capture_output=1, text=1, env=env)
                os.chdir('..')
                if out.returncode != 0:
                    logging.error('cg-distributions failed to run!')
                    sys.exit(1)

            else:
                options.slurm['cgd']['o'] = f'dist.{phase}.{system_number}.{iteration}.slurm'
                options.slurm['cgd']['j'] = f'dist-{system_number}'
                jobs[slurm.submit_job('cgd', cg_cmd, options.slurm['cgd'])] = 'cgd'
    if jobs:
        slurm.wait_on_jobs(jobs)

    average_distributions(phase, target_cache['rdf'], iteration, options)
    average_distributions(phase, target_cache['adf'], iteration, options)
    average_distributions(phase, target_cache['bdf'], iteration, options)


def make_cg_distribution_input(iteration, phase, system_number, i, data, options):
    ''' Make input script for cg-distributions code. '''
    cgini = open(os.path.join(os.path.dirname(__file__),
                                    '../templates/cg_distributions.in')).read()
    input_path = f'in_scripts/{phase}/cg-sys{system_number}.ini'
    with open(input_path, 'w') as f:
        def make_sample_string(s):
            if s in options.sample_cutoff:
                return '{} {beg} {end} {inv}'.format(
                        s, **options.sample_cutoff[s])
            return ''

        f.write(cgini.format(phase=phase, data = data,
                             i = iteration, sys = system_number,
                             rdf = make_sample_string('rdf'),
                             bdf = make_sample_string('bdf'),
                             adf = make_sample_string('adf'),
                             sp_bond = options.sample_special_bond))
        if options.dumpstep_range:
            f.write('timestep_range {} {} {}\n'.format(*options.dumpstep_range))
        for i in options.beads:
            f.write(f'\ntype {i} {options.beads[i]["data"]}')
        f.write('\n')
        for i in options.beads:
            bead = options.beads[i]
            f.write(f'\nbead {bead["type"]} {bead["data"]}')
            f.write(f'\nweights {bead["type"]} 1.0')
    return input_path


def cg_job_complete(phase, sys, iteration, options):
    name = f'sys{sys}-{iteration}_*.txt'
    num_beads = len(options.beads)
    num_pairs = num_beads*(num_beads-1)//2 + num_beads
    num_angles = len(options.angles)
    # Only valid for PE
    num_bonds = len(options.bonds)

    return len(glob(f'cg-dist/{phase}/rdf-{name}')) == num_pairs \
           and len(glob(f'cg-dist/{phase}/bdf-{name}')) == num_bonds \
           and len(glob(f'cg-dist/{phase}/adf-{name}')) == num_angles

