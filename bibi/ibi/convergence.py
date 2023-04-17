import numpy
from glob import glob
import logging
from . import lammps


def convergence_criterion(iteration, options):
    '''Assumed real error is composed of computation error and sample error. if
    computation error <= sample error, iteration is converged at current population.'''
    logging.info('-- Checking convergence')

    types = {'pair': [], 'bond': [], 'angle': []}
    convergence_condition = {pt: {} for pt in types}

    for pt in types:
        for f in sorted(glob(f'potentials/{pt}.table.*.{iteration}')):
            types[pt].append(f.split('.')[-2])
        for tt in types[pt]:
            U0 = lammps.read_tabular_potential('avg', pt, tt, 0)[1]
            Ui_1 = lammps.read_tabular_potential('avg', pt, tt, iteration-1)[1]
            Ui = lammps.read_tabular_potential('avg', pt, tt, iteration)[1]
            m = determine_mask(pt, tt, iteration, options)
            nominator = numpy.linalg.norm(Ui[~m] - Ui_1[~m])
            denominator = numpy.linalg.norm(Ui[~m] - U0[~m])
            criteria = round(nominator / denominator, 4)
            tol = options.convergence_tolerance[pt]
            convergence_condition[pt][tt] = {'criteria': criteria,
                                             'condition': criteria < tol}
    # Write a file to record all the error information
    if iteration == 1:
        with open(f'convergence.txt', 'w') as f:
            f.write('iteration\t')
            for pt in types:
                for tt in types[pt]:
                    f.write(f'{pt}_{tt}\t')
            f.write('\n')
    # Prevent double writing in case of a restart
    if iteration not in numpy.loadtxt(f'convergence.txt', usecols=(0), skiprows=1):
        with open(f'convergence.txt', 'a') as f:
            f.write(f'{iteration}\t')
            for pt in types:
                for tt in types[pt]:
                    criteria = convergence_condition[pt][tt]['criteria']
                    f.write(f'{criteria}\t')
            f.write('\n')

    if options.convergence_criteria == 'total':
        if all([[convergence_condition[pt][tt]['condition'] for tt in types[pt]][0]
                                                            for pt in types]):
            logging.info('---- Convergence criterion is reached!')
            return True
        else:
            for pt in convergence_condition:
                for tt in convergence_condition[pt]:
                    if not convergence_condition[pt][tt]['condition']:
                        logging.info(f'---- {pt.capitalize()} {tt} is not converged!')
            return False


def determine_mask(pt, tt, iteration, options):
    ''' Determine the mask using target distributions of both phases. '''
    pot2dist = dict(pair='rdf', bond='bdf', angle='adf')
    m = {}
    for p in options.phases:
        f = glob(f'averaged_targets/{p}/{pot2dist[pt]}-{tt}.txt')[0]
        m[p] = numpy.genfromtxt(f, usecols=(1)) < options.tolerance[pt]
    return m['amorphous'] & m['crystal']

