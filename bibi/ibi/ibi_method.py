import os
import sys
import random
import shutil
import logging
from scipy.signal import savgol_filter
from .ibi_config import IbiConfig
from .initial_potentials import *
from .update_potentials import *
from .pressure_correction import correct_pressure
from . import distributions
from . import lammps
from .convergence import convergence_criterion


class ibi_method:
    def __init__(self, ini_file, clean_start):
        self.options = IbiConfig(ini_file)
        if clean_start:
            clean_folders()
            self.iteration = 0
        else:
            self.iteration = find_start_point()
        # Path to each phase CG data files
        self.lmp_data = {}
        # Path to each phase target distributions
        self.target_cache = {}
        # Number of target samples
        self.options.num_target_sample = {}
        # Number of CG systems of each phase
        self.sys_num = {}
        # CG data file of each phase in current iteration
        self.current_lmpdata = {}
        # Pressure correction coefficient from previous step.
        self.pressure_coeff = 0.0
        # Read the CG data and the target distributions
        logging.info(f'Computing target distributions')
        for phase in self.options.phases:
            create_folders(phase, self.options.cluster_mode)
            self.lmp_data[phase] = sorted(glob(self.options.lmpdata_path[phase]))
            self.target_cache[phase], self.options.num_target_sample[phase] =\
                         distributions.average_all_targets(self.options, phase)
            # Prevents the same systems from being selected each time.
            if self.options.shuffle_data:
                random.seed(self.options.random_seed)
                random.shuffle(self.lmp_data[phase])
            self.sys_num[phase] = self.options.system_used[phase]
            self.current_lmpdata[phase] = self.lmp_data[phase][:self.sys_num[phase]]


    def initial_potentials(self):
        logging.info(f'-- Computing initial potentials.')
        for p in self.options.phases:
            logging.debug(f'---- Computing {p} pair potential.')
            initial_pair_potentials(p, self.target_cache[p]['rdf'], self.options)
            logging.debug(f'---- Computing {p} bond potential.')
            initial_bond_potentials(p, self.target_cache[p]['bdf'], self.options)
            logging.debug(f'---- Computing {p} angle potential.')
            initial_angle_potentials(p, self.target_cache[p]['adf'], self.options)


    def run_iterations(self):
        if self.iteration == 0:
            logging.info(f'# Starting BIBI for {self.options.max_iterations} iterations #')
            self.initial_potentials()
            self.weighted_average_potentials()
            self.sample_cg_model()
        else:
            logging.info('# Restarting IBI on iteration '
                  f'{self.iteration}/{self.options.max_iterations} #')
        while self.iteration <= self.options.max_iterations:
            logging.info(f'Iteration {self.iteration}')
            self.update_cg_potentials()
            self.iteration += 1
            self.weighted_average_potentials()
            self.pressure_coeff = correct_pressure(self.iteration,
                    self.current_lmpdata, self.pressure_coeff, self.options)
            self.sample_cg_model()
            if convergence_criterion(self.iteration, self.options):
                break


    def weighted_average_potentials(self):
        ''' Calculate weighted average of potentials for each phase. '''
        potentials = sorted(glob(f'potentials/crystal/*.table.*.{self.iteration}'))
        w = self.options.wc
        for p in potentials:
            m = re.match(f'potentials/crystal/(.+)\.table\.(.+)\.\d+', p)
            ptype, btypes = m.groups()
            tgt_a = self.get_target('amorphous', pot2dist(ptype), btypes)
            da = distributions.read_distribution_file(tgt_a)
            tgt_c = self.get_target('crystal', pot2dist(ptype), btypes)
            dc = distributions.read_distribution_file(tgt_c)
            xc, Uc, fc = lammps.read_tabular_potential('crystal', ptype, btypes, self.iteration)
            xa, Ua, fa = lammps.read_tabular_potential('amorphous', ptype, btypes, self.iteration)

            numerator = Uc*w*dc.p + Ua*(1.0-w)*da.p
            denominator = w*dc.p + (1.0-w)*da.p

            tol = self.options.tolerance[ptype]
            if w == 1.0:
                m = dc.p < tol
            elif w == 0.0:
                m = da.p < tol
            else:
                m = (da.p < tol) & (dc.p < tol)
            U = numpy.zeros(Uc.shape)
            U[~m] = numerator[~m] / denominator[~m]
            U = extrapolate_potential(xc, U, m)
            window = self.options.smooth_factor[ptype]['window']
            polyorder = self.options.smooth_factor[ptype]['polyorder']
            U = savgol_filter(U, window, polyorder)
            f = -compute_derivative(U, xc[1]-xc[0])
            lammps.write_potential_table('avg', xc, U, f, ptype, btypes, self.iteration)


    def sample_cg_model(self):
        logging.info('-- Sampling CG distributions')
        for phase in self.options.phases:
            logging.info(f'---- Running {phase} CG simulations')
            lammps.run_lammps(phase, self.iteration,
                    self.current_lmpdata[phase], 'sample', self.options)
            logging.info(f'---- Computing {phase} structure distributions')
            distributions.compute_distribution(phase, self.target_cache[phase],
                    self.current_lmpdata[phase], self.iteration, self.options)


    def update_cg_potentials(self):
        logging.info('-- Updating potentials')
        for phase in self.options.phases:
            logging.debug(f'---- Updating pair potential for {phase} system')
            for current in self.get_cg_distributions(phase, 'rdf'):
                m = re.match('.*rdf-.*_([A-Z][A-Z])\.avg', current)
                btypes = m.group(1)
                target = self.get_target(phase, 'rdf', btypes)
                update_pair_potentials(phase, current, target, btypes,
                        self.iteration, self.options)

            if self.options.bond_style:
                logging.debug(f'---- Updating bond potential for {phase} system')
                for current in self.get_cg_distributions(phase, 'bdf'):
                    m = re.match('.*bdf-.*_([A-Z][A-Z])\.avg', current)
                    btypes = m.group(1)
                    target = self.get_target(phase, 'bdf', btypes)
                    update_tabular_potentials(phase, 'bond', current, target,
                            btypes, self.iteration, self.options)

            if self.options.angle_style:
                logging.debug(f'---- Updating angle potential for {phase} system')
                for current in self.get_cg_distributions(phase, 'adf'):
                    m = re.match('.*adf-.*_([A-Z][A-Z][A-Z])\.avg', current)
                    btypes = m.group(1)
                    target = self.get_target(phase, 'adf', btypes)
                    update_tabular_potentials(phase, 'angle', current, target,
                            btypes, self.iteration, self.options)


    def get_cg_distributions(self, phase, dtype):
        ''' Returns averaged CG distributions file paths for a given phase
        (e.g. crystal or amorphous), distribution type (rdf, bdf, adf, etc)
        at the current iteration. '''
        return glob(f'cg-dist/{phase}/{dtype}-{self.iteration}_*.avg')


    def get_target(self, phase, dtype, btypes):
        ''' Returns path to averaged target distribution for a given
        phase, distribution type, and bead types. '''
        pattern = f'-{btypes}.txt'
        for x in self.target_cache[phase][dtype]:
            if pattern in x:
                return x


def find_start_point():
    ''' Determines last completed iteration from the potential files. '''
    # Track how many potential files are in each iteration.
    iteration_count = {}
    for f in glob('potentials/*.table.*.*'):
        m = re.match('potentials/(.+)\.table\.(.+)\.(\d+)', f)
        i = int(m.group(3))
        if i not in iteration_count:
            iteration_count[i] = 1
        else:
            iteration_count[i] += 1

    iterations = sorted(iteration_count.keys())
    if len(iterations) == 1:
        return 0
    for i in iterations[1:]:
        if iteration_count[i] < iteration_count[i-1]:
            return i-1
    return i-1


def make_folder(folder_name):
    ''' make directory if it does not exist '''
    folder_path = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


def create_folders(phase, cluster_mode):
    ''' Generates all folders used by ibi_optimization. '''
    folders_to_init = ['in_scripts', 'potentials', 'dump', 'log'
                     , 'averaged_targets', 'cg-dist', 'cg-data', 'cg-updates']
    for f in folders_to_init:
        folder_path = make_folder(f)
        if not os.path.exists(os.path.join(folder_path, phase)):
            os.mkdir("{}/{}".format(folder_path, phase))
    if cluster_mode: make_folder('slurms')


def clean_folders():
    ''' Deletes all folders used by ibi_optimization. '''
    folders_to_init = ['in_scripts', 'potentials', 'dump', 'log', 'plots'
                     , 'averaged_targets', 'cg-dist', 'cg-data', 'cg-updates']
    for f in glob('*.txt') + glob('*.png'):
        if os.path.exists(f): os.remove(f)
    for f in folders_to_init:
        if os.path.exists(f): shutil.rmtree(f)


def pot2dist(s):
    ''' Converts potential name to distribution name (e.g., angle -> adf). '''
    names = dict(pair='rdf', bond='bdf', angle='adf', dihedral='tdf')
    return names[s]


def dist2pot(s):
    ''' Converts distribution name to potential name (e.g., adf -> angle). '''
    names = dict(rdf='pair', bdf='bond', adf='angle', tdf='dihedral')
    return names[s]

