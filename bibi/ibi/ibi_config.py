import os
import sys
import re
from glob import glob
import numpy
import logging


class IbiConfig:
    def __init__(self, config_file):
        # Read the config file from the command line
        if not os.path.exists(config_file):
            logging.error(f'Input file {config_file} does not exist!')
            sys.exit(1)
        self.fid = open(config_file, 'r')

        #--------------------------------------------------------------
        #                       Material Properties
        #--------------------------------------------------------------
        # Weight of crystal potential
        self.wc = 0.0

        #--------------------------------------------------------------
        #       Path to LAMMPS Data and Target Distribution Files
        #--------------------------------------------------------------
        # Phases in the system
        self.phases = []
        # LAMMPS data file for each phase
        self.lmpdata_path = {}
        # Density of each phase
        self.density = {}
        # Pattern that matches target distribution files for each phase
        # and distribution type.
        self.target_path = {}

        #--------------------------------------------------------------
        #          Parameters for Runing LAMMPS Simulation
        #--------------------------------------------------------------
        # LAMMPS settings
        self.lmp_style = 'full'
        self.mdtemp  = 300.0
        self.timestep = 1.0
        self.nve_step = 1.0

        # Forcefield settings
        # Pair settings
        self.pairs = {}
        self.lmp_special_bond = '0 0 1'
        # Bond settings
        self.bonds = {}
        self.bond_style = 'table'
        # Angle settings
        self.angles = {}
        self.angle_style = 'table'

        #--------------------------------------------------------------
        #          Parameters for Updating the Potentials
        #--------------------------------------------------------------
        self.max_iterations = 100
        self.tolerance = {}
        self.update_range = 15.0
        self.smooth_factor = {}
        self.cfactor = {}
        self.pressure_factor = 0.1
        self.pressure_tolerance = 1000
        self.max_pressure_iterations = 30

        #--------------------------------------------------------------
        #            Parameters for Sampling CG Distributions
        #--------------------------------------------------------------
        # Beads definition
        self.num_target_sample = 20
        self.masses  = {}
        self.atoms = {}
        self.beads = {}
        self.sample_cutoff = {}
        self.sample_special_bond = '0 0 1'
        self.dumpstep_range = []

        #--------------------------------------------------------------
        #                   Convergence Parameters
        #--------------------------------------------------------------
        self.convergence_criteria = 'total'
        self.convergence_tolerance = dict(pair=1e-2, bond=1e-2, angle=1e-2)
        self.shuffle_data = False
        self.random_seed = 10
        self.system_used = {}

        #--------------------------------------------------------------
        #                     Cluster Parameters
        #--------------------------------------------------------------
        self.cluster_mode = False
        self.slurm = dict(lmp={}, cgd={}, exclude_nodes=None)
        # N=Nodes, c=CPUs, n=Tasks per cpu, m=Memory, p=Partition, q=QOS, t=time
        # Settings for running lammps on cluster
        self.slurm['lmp'] = dict(N=1, c=16, n=1, m=16, p='general', q='public', t=10)
        # Settings for running cg-distributions on cluster
        self.slurm['cgd'] = dict(N=1, c=16, n=1, m=16, p='general', q='public', t=10)
        self.lmp_ntasks = int(os.cpu_count() / 2.0) # Two threads per core
        self.cg_ntasks = int(os.cpu_count() / 2.0)

        # Read the config file
        self.read_config_file()


    def read_config_file(self):
        ''' Read the configuration file from the given path from command line. '''
        try:
            for line in self.fid:
                if line == '': break
                line = line.split('#')[0].strip().split()
                if len(line) == 0: continue
                #--------------------------------------------------------------
                #                       Material Properties
                #--------------------------------------------------------------
                if line[0] == 'crystal_weight':
                    self.wc = float(line[1]) / 100

                #--------------------------------------------------------------
                #       Path to LAMMPS Data and Target Distribution Files
                #--------------------------------------------------------------
                elif line[0] == 'path':
                    if line[1] == 'lmpdata':
                        self.lmpdata_path[line[2]] = line[3]
                        self.density[line[2]] = compute_number_density(line[3])
                        self.phases.append(line[2])
                    elif re.match('target_[rba]df_path', line[1]):
                        dt = line[1].split('_')[1]
                        phase = line[2]
                        if phase not in self.target_path:
                            self.target_path[phase] = {}
                        self.target_path[phase][dt] = line[3]
                    else:
                        print('Invalid command!\n' + line)
                        sys.exit(1)

                #--------------------------------------------------------------
                #          Parameters for Runing LAMMPS Simulation
                #--------------------------------------------------------------
                elif line[0] == 'lammps':
                    # Simulation settings
                    if line[1] == 'lmp_style':
                        self.lmp_style = line[2]
                    elif line[1] == 'nve_step':
                        self.nve_step = float(line[2])
                    elif line[1] == 'timestep':
                        self.timestep = float(line[2])

                    # Pair settings
                    elif line[1] == 'special_bonds':
                        self.lmp_special_bond ='{} {} {}'.format(line[2],line[3],line[4])

                    # Bond settings
                    elif line[1] == 'bond_style':
                        self.bond_style = line[2]
                    elif line[1] == 'bond_coeff':
                        bond = {}
                        bond['id'] = int(line[2])
                        bond['type'] = line[3]
                        self.bonds[int(line[2])] = bond

                    # Angle settings
                    elif line[1] == 'angle_style':
                        self.angle_style = line[2]
                    elif line[1] == 'angle_coeff':
                        angle = {}
                        angle['id'] = int(line[2])
                        angle['type'] = line[3]
                        self.angles[int(line[2])] = angle
                    else:
                        print('Invalid command!\n' + line)
                        sys.exit(1)

                #--------------------------------------------------------------
                #            Parameters for Sampling CG Distributions
                #--------------------------------------------------------------
                elif line[0] == 'sample':
                    if line[1] == 'dumpstep_range':
                        self.dumpstep_range = line[2:5]
                    # Define the beads and their type
                    elif line[1] == 'type':
                        self.atoms[line[3]] = int(line[2])
                    elif line[1] == 'bead':
                        bead = {}
                        bead['type'] = line[2]
                        # Half of the line length after the type belongs to
                        # atom data and the other half are the weights for them
                        for ii in range(3, len(line)):
                            if ii < 3 + int((len(line) - 3) / 2):
                                bead['data'] = line[ii]
                            else:
                                bead['weights'] = line[ii]
                        self.beads[self.atoms[line[3]]] = bead
                    elif line[1] == 'special_bonds':
                        self.sample_special_bond = f'{line[2]} {line[3]} {line[4]}'
                    elif line[1] == 'range':
                        self.sample_cutoff[line[2]] = dict(
                                beg=float(line[3]),
                                end=float(line[4]),
                                inv=float(line[5]))
                    else:
                        print('Invalid sample command!\n{}'.format(line))
                        sys.exit(1)

                #--------------------------------------------------------------
                #          Parameters for Updating the Potentials
                #--------------------------------------------------------------
                elif line[0] == 'update':
                    if line[1] == 'iterations':
                        self.max_iterations = int(line[2])
                    elif line[1] == 'tolerance':
                        self.tolerance[line[2]] = float(line[3])
                    elif line[1] == 'pressure_iterations':
                        self.max_pressure_iterations = int(line[2])
                    elif line[1] == 'pressure_factor':
                        self.pressure_factor = float(line[2])
                    elif line[1] == 'pressure_tolerance':
                        self.pressure_tolerance = float(line[2])
                    elif line[1] == 'fit_range':
                        self.update_range = float(line[2])
                    elif line[1] == 'scaling_factor':
                        self.cfactor[line[2]] = float(line[3])
                    elif line[1] == 'smooth_factor':
                        factor = {}
                        factor['window'] = int(line[3])
                        factor['polyorder'] = int(line[4])
                        self.smooth_factor[line[2]] = factor
                    else:
                        print('Invalid command.\n' + line)

                #--------------------------------------------------------------
                #                   Convergence Parameters
                #--------------------------------------------------------------
                elif line[0] == 'convergence':
                    if line[1] == 'criterion':
                        self.converge_cri = line[2]
                    elif line[1] == 'tolerance':
                        self.convergence_tolerance[line[2]] = float(line[3])
                    elif line[1] == 'system_used':
                        self.system_used[line[2]] = int(line[3])
                    elif line[0] == 'shuffle_data':
                        self.shuffle_data = True
                    elif line[0] == 'random_seed':
                        self.random_seed = int(line[1])
                    else:
                        print('Invalid command!\n' + line)
                        sys.exit(1)

                #--------------------------------------------------------------
                #                     Cluster Parameters
                #--------------------------------------------------------------
                elif line[0] == 'slurm':
                    if line[1] == 'cluster' and line[2] == 'on':
                        self.cluster_mode = True
                    if line[1] == 'exclude_nodes':
                        self.slurm['exclude_nodes'] = line[2]
                    # Settings for running lammps on clusters
                    if line[1] == 'lammps':
                        if line[2] == 'nodes':
                            self.slurm['lmp']['N'] = int(line[3])
                        elif line[2] == 'cpu':
                            self.slurm['lmp']['c'] = int(line[3])
                        elif line[2] == 'tasks':
                            self.slurm['lmp']['n'] = int(line[3])
                        elif line[2] == 'memory':
                            self.slurm['lmp']['m'] = int(line[3])
                        elif line[2] == 'partition':
                            self.slurm['lmp']['p'] = line[3]
                        elif line[2] == 'qos':
                            self.slurm['lmp']['q'] = line[3]
                        elif line[2] == 'time':
                            self.slurm['lmp']['t'] = int(line[3])
                    # Settings for running cg-distributions on clusters
                    if line[1] == 'cg':
                        if line[2] == 'nodes':
                            self.slurm['cgd']['N'] = int(line[3])
                        elif line[2] == 'cpu':
                            self.slurm['cgd']['c'] = int(line[3])
                        elif line[2] == 'tasks':
                            self.slurm['cgd']['n'] = int(line[3])
                        elif line[2] == 'memory':
                            self.slurm['cgd']['m'] = int(line[3])
                        elif line[2] == 'partition':
                            self.slurm['cgd']['p'] = line[3]
                        elif line[2] == 'qos':
                            self.slurm['cgd']['q'] = line[3]
                        elif line[2] == 'time':
                            self.slurm['cgd']['t'] = int(line[3])
                else:
                    print('Unknown input parameters !!!!', line)
                    sys.exit(1)
        except IndexError:
            print('Syntax was invalid:', ' '.join(line))
            sys.exit(1)


def compute_number_density(path):
    ''' Read the number of beads and compute the volume of the simulation box
    by reading its dimensions and compute the number density (1/A^3) of the system. '''
    data_file = sorted(glob(path))[0]
    fid = open(data_file)
    N, V = 0, numpy.zeros((3,3))
    for line in fid:
        args = line.split()
        if 'atoms' in line:
            N = int(args[0])
        if 'xlo xhi' in line:
            V[0,0] = float(args[1]) - float(args[0])
        elif 'ylo yhi' in line:
            V[1,1] = float(args[1]) - float(args[0])
        elif 'zlo zhi' in line:
            V[2,2] = float(args[1]) - float(args[0])
        elif 'xy xz yz' in line:
            V[0,1] = float(args[0])
            V[0,2] = float(args[1])
            V[1,2] = float(args[2])
    rho = N / numpy.linalg.det(V)
    return rho

