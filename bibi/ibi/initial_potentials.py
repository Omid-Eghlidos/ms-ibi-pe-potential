import re
import numpy
import logging
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from . import distributions
from . import lammps
from .distributions import split_extrapolated_regions


# Boltzmann constant (kcal/mol/K)
kB = 0.0019872041


#------------------------------------------------------------------------------
#                      Initial Pair Potential Functions
#------------------------------------------------------------------------------
def initial_pair_potentials(phase, avg_rdf_path, options):
    ''' Compute initial pair potentials. '''
    # Savgol settings
    window = options.smooth_factor['pair']['window']
    polyorder = options.smooth_factor['pair']['polyorder']
    for target in avg_rdf_path:
        m = re.match('.*rdf-([A-Z][A-Z])\.txt', target)
        pair_type = m.group(1)
        rdf0 = distributions.read_distribution_file(target)
        U0 = pair_table_potential(rdf0.x, rdf0.p, options)
        U0 = savgol_filter(U0, window, polyorder)
        f0 = -compute_derivative(U0, rdf0.x[1]-rdf0.x[0])
        lammps.write_potential_table(phase, rdf0.x, U0, f0, 'pair', pair_type, 0)


def pair_table_potential(r, g, options):
    ''' Computes initial pair energy table from RDF, temperature, and cutoff. '''
    rcut = options.sample_cutoff['rdf']['end']
    T = options.mdtemp
    m = g < options.tolerance['pair']
    U = numpy.zeros(g.shape)
    U[~m] = -kB*T*numpy.log(g[~m])
    U = extrapolate_potential(r, U, m)
    # Shift potential to zero at the cutoff
    U -= U[-1]
    return U


#------------------------------------------------------------------------------
#                      Initial Bond Potential Functions
#------------------------------------------------------------------------------
def initial_bond_potentials(phase, avg_bdf_path, options):
    # Savgol settings
    window = options.smooth_factor['bond']['window']
    polyorder = options.smooth_factor['bond']['polyorder']
    if options.bond_style == 'table':
        for target in avg_bdf_path:
            m = re.match('.*bdf-.*([A-Z]{2})\.txt', target)
            bond_type = m.group(1)
            bdf0 = distributions.read_distribution_file(target)
            U0 = bond_table_potential(bdf0.x, bdf0.p, options)
            U0 = savgol_filter(U0, window, polyorder)
            f0 = -compute_derivative(U0, bdf0.x[1]-bdf0.x[0])
            lammps.write_potential_table(phase, bdf0.x, U0, f0, 'bond', bond_type, 0)


def bond_table_potential(l, p_l, options):
    ''' Compute initial bond energy for BDF for a given tempertaure. '''
    T = options.mdtemp
    U = numpy.zeros(p_l.shape)
    m = p_l < options.tolerance["bond"]
    U[~m] = -kB * T * numpy.log(p_l[~m] / l[~m]**2)
    U = extrapolate_potential(l, U, m)
    return U - min(U)


#------------------------------------------------------------------------------
#                      Initial Angle Potential Functions
#------------------------------------------------------------------------------
def initial_angle_potentials(phase, avg_adf_path, options):
    # Savgol settings
    window = options.smooth_factor['angle']['window']
    polyorder = options.smooth_factor['angle']['polyorder']
    if options.angle_style == 'table':
        for target in avg_adf_path:
            m = re.match('.*adf-([A-Z]{3})\.txt', target)
            angle_type = m.group(1)
            adf0 = distributions.read_distribution_file(target)
            U0 = angle_table_potential(adf0.x, adf0.p, options)
            U0 = savgol_filter(U0, window, polyorder)
            f0 = -compute_derivative(U0, adf0.x[1]-adf0.x[0])
            lammps.write_potential_table(phase, adf0.x, U0, f0, 'angle', angle_type, 0)


def angle_table_potential(theta, p_theta, options):
    ''' Compute initial angle energy from ADF and tempertaure '''
    T = options.mdtemp
    U = numpy.zeros(p_theta.shape)
    m = p_theta < options.tolerance['angle']
    U[~m] = -kB*T*numpy.log(p_theta[~m] / numpy.sin(numpy.pi*theta[~m]/180.0))
    U = extrapolate_potential(theta, U, m)
    return U - min(U)


#------------------------------------------------------------------------------
#                               Helper Functions
#------------------------------------------------------------------------------

def extrapolate_potential(x, U, mask, return_regions=False, **kwargs):
    ''' Extrapolate the potential energy for zero values. '''
    # Determine mask of values that are below the tolerance value.
    regions = split_extrapolated_regions(mask, min_sep=4)
    if kwargs.get('region','all') == 'first_only':
        regions = regions[0:1]
    elif kwargs.get('region','all') == 'first_last':
        regions = [regions[0], regions[-1]]
    for r in regions:
        ii = r.interpolate_from
        f = numpy.polynomial.Polynomial.fit(x[ii], U[ii], deg=2)
        if f.coef[-1] < 0.0:
            logging.warning('WARNING: potential quadratic coefficient diverges.')
            f = numpy.polynomial.Polynomial.fit(x[ii], U[ii], deg=1)
        U[r.points] = f(x[r.points])
    if return_regions:
        return U, regions
    return U


def compute_derivative(y, dx):
    # use finite difference to compute derivative
    yp = numpy.zeros(len(y))
    yp[1:-1] = (y[2:] - y[:-2]) / (2.0 * dx)
    yp[0] = (y[1] - y[0]) / dx
    yp[-1] = (y[-1] - y[-2]) / dx
    return yp

