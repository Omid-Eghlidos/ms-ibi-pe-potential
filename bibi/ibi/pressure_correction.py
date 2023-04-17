import numpy
import logging
from scipy.stats import linregress
from scipy.integrate import simps
from glob import glob
from . import lammps
from .distributions import read_distribution_file
from .initial_potentials import compute_derivative


def correct_pressure(i, lmpdata, A, options):
    ''' Modifies pair tables in place to correct pressure using linear term: 
    dU = A*(1-r/r_cut). '''
    dA, A = A, 0.0
    dPdA = pressure_sensitivity(i, options)
    for j in range(options.max_pressure_iterations):
        logging.info(f'-- Pressure correction: iteration {j+1}')
        update_pair_tables(i, dA, options.update_range)
        A += dA
        weighted_pressure = 0.0
        for phase in options.phases:
            w = 1.0 - options.wc if phase == 'amorphous' else options.wc
            if w == 0.0:
                continue
            lammps.run_lammps(phase, i, lmpdata[phase], 'pressure', options)
            p = get_mean_pressure(glob(f'dump/{phase}/pressure*.txt'))
            logging.info(f'---- {phase.capitalize()} pressure is {p:.1f} atm')
            weighted_pressure += w*p
        dA = -weighted_pressure / dPdA * options.pressure_factor
        logging.info(f'---- Pressure coefficient A is {A:.4f}')
        logging.info(f'---- Weighted pressure is {weighted_pressure:.1f} atm')
        if abs(weighted_pressure) < options.pressure_tolerance:
            logging.info(f'-- Pressure converged at iteration {j+1}')
            return A
    logging.info(f'-- Pressure did not converge in {j+1} iterations')
    return A


def pressure_sensitivity(iteration, options):
    ''' Estimates the sensitivity of pressure correction parameter A on the
    weighted pressure, where dU = A*(1-r/rcut). '''
    dPdA = 0.0
    r_cut = options.update_range
    for phase, rho in options.density.items():
        w = 1.0 - options.wc if phase == 'amorphous' else options.wc 
        rdfs = glob(f'cg-dist/{phase}/rdf-{iteration-1}_*.avg')
        assert len(rdfs) == 1, 'Multiple bead types not supported'
        d = read_distribution_file(rdfs[0])
        m = d.x < r_cut
        int_gr3 = simps(d.x[m]**3*d.p[m], d.x[m])
        dPdA += w * 2.0*numpy.pi*rho**2/3.0 * int_gr3 / r_cut
    # Conversion of [kcal/mol] to [atm.A^3]
    return dPdA / 1.45836e-5


def update_pair_tables(i, dA, r_cut):
    ''' Updates pair tables at iteration i, by dA. '''
    table = glob(f'potentials/pair.table.*.{i}')
    assert len(table) == 1, 'Multiple bead types not supported.'
    tt = table[0].split('.')[-2]
    r, u, _ = lammps.read_tabular_potential('avg', 'pair', tt, i)
    u += dA * (1.0 - r/r_cut)
    f = -compute_derivative(u, r[1]-r[0])
    lammps.write_potential_table('avg', r, u, f, 'pair', tt, i)


def get_mean_pressure(paths):
    ''' Reads output from LAMMPS fix ave/time and returns the mean pressure. '''
    pressures = []
    min_pvalue = 1.0
    for path in paths:
        step, p = numpy.loadtxt(path).T
        min_pvalue = min(min_pvalue, linregress(step, p)[3])
        pressures.append(p.mean())
    if min_pvalue < 0.05:
        logging.warning('---- Pressure maybe not at equilibrium '
                       f'(pvalue = {min_pvalue:.3f})')
    return numpy.mean(pressures)


