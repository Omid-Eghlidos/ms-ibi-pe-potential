import numpy
from glob import glob
import logging
from scipy.signal import savgol_filter
from . import distributions
from . import lammps
from .initial_potentials import kB, extrapolate_potential, compute_derivative


def update_pair_potentials(phase, current, target, pair_type, iteration, options):
    ''' Update tabulated pair potential of each phase. '''
    T = options.mdtemp
    alpha = options.cfactor['pair']
    rdf_i = distributions.read_distribution_file(current)
    rdf_t = distributions.read_distribution_file(target)
    r, U, f = lammps.read_tabular_potential(phase, 'pair', pair_type, iteration)
    assert len(rdf_i.x) == len(rdf_t.x), 'CG and target distribution are unequal size'

    tol = options.tolerance['pair']
    dU = numpy.zeros(U.shape)
    m = (rdf_t.p < tol) | (rdf_i.p < tol)
    dU[~m] = kB*T*numpy.log(rdf_i.p[~m] / rdf_t.p[~m])
    lammps.write_potential_updates(phase, r, dU, 'pair', pair_type, iteration)
    U += alpha*dU
    U = extrapolate_potential(r, U, m, region='first_only')

    f = -compute_derivative(U, r[1]-r[0])
    lammps.write_potential_table(phase, r, U, f, 'pair', pair_type, iteration+1)


def update_tabular_potentials(phase, potential_type, current, target, table_type,
                                                        iteration, options):
    ''' Update tabular potentials for bond and angle distribution
    potential_type: angle/bond; table_type: AEA/ABA/ADA/ABC... '''
    T = options.mdtemp
    if potential_type == 'angle':
        alpha = options.cfactor[potential_type]
        region = 'first_last'
    elif potential_type == 'bond':
        alpha = options.cfactor[potential_type]
        region = ''

    pdf_i = distributions.read_distribution_file(current)
    pdf_t = distributions.read_distribution_file(target)
    r, U, f = lammps.read_tabular_potential(phase, potential_type, table_type, iteration)
    assert len(pdf_i.x) == len(pdf_t.x), 'CG and target distribution are unequal size'

    tol = options.tolerance[potential_type]
    m = (pdf_t.p < tol) | (pdf_i.p < tol)
    dU = numpy.zeros(U.shape)
    dU[~m] = kB*T*numpy.log(pdf_i.p[~m] / pdf_t.p[~m])
    lammps.write_potential_updates(phase, r, dU, potential_type, table_type, iteration)
    U += alpha*dU
    U = extrapolate_potential(r, U, m, region=region)

    f = -compute_derivative(U, r[1]-r[0])
    lammps.write_potential_table(phase, r, U, f, potential_type, table_type, iteration+1)

