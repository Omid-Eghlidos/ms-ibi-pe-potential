#!/usr/bin/env python3
"""
Goal
----
Plot convergence behavior of the BIBI code.
"""


import os
from glob import glob
import numpy
from scipy.signal import savgol_filter
# Set default plotting parameters
from matplotlib import pyplot
from matplotlib.colors import TABLEAU_COLORS as colors


pyplot.rc('font', family='Times New Roman')
pyplot.rc('font', size=8)
pyplot.rc('mathtext', fontset='stix')

colours = list(colors.values())


def plot_convergence(weight):
    ''' Plot the the potential convergence. '''

    pyplot.clf()
    fig, axes = pyplot.subplots(1, 3, figsize=(6.5,2.0))

    errors = read_errors('convergence.txt')
    for tt in errors:
        it = numpy.arange(1, len(errors[tt])+1)
        pt, bt = tt.split('_')[0], tt.split('_')[1]
        axes[0].plot(it, errors[tt], lw=1.0, label=f'{pt.capitalize()}')
    #yy = [1e-2] * len(it)
    #axes[0].plot(it, yy, color='tab:red', lw=1.0, dashes=(2,2), label='Tol')
    axes[0].set_xlabel('Iterations')
    axes[0].set_xscale('log')
    axes[0].set_xlim(1, max(it))
    axes[0].set_ylabel('$\dfrac{||U^i-U^{i-1}||_2}{||U^i-U^0||_2}$')
    axes[0].set_yscale('log')
    #axes[0].set_ylim(10**-2.1, 10**0.2)
    axes[0].legend(loc='best', fontsize=8, frameon=False)
    axes[0].minorticks_on()

    update = read_potential_update('bond', max(it)-1)
    for p in update:
        axes[1].plot(update[p][0], update[p][1], lw=1.0, label=f'{p.capitalize()}')
    axes[1].set_xlabel('$l (\AA)$')
    xmin, xmax = 1.0, 4.0
    xticks = numpy.linspace(xmin, xmax, 3)
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels(xticks)
    axes[1].set_xlim(xmin, xmax)
    axes[1].set_ylabel('$\Delta U(l)$ (kcal/mol)')
    #axes[1].set_ylim(-0.3, 1.2)
    axes[1].legend(loc='best', fontsize=8, frameon=False)
    axes[1].minorticks_on()

    Ps = read_target_distributions()
    m = determine_mask(Ps)
    m = m['bdf']
    denominator = numpy.ones(len(Ps['crystal']['bdf-AA']))
    denominator[~m] = weight*Ps['crystal']['bdf-AA'][~m]*update['crystal'][1][~m]\
                    + (1.0-weight)*Ps['amorphous']['bdf-AA'][~m]*update['amorphous'][1][~m]
    denominator[denominator==0.0] = 1.0
    alpha = 0.05
    for p in update:
        w = weight if p == 'crystal' else (1.0-weight)
        dU = alpha*w*Ps[p]['bdf-AA']*update[p][1] / denominator
        axes[2].plot(update[p][0], dU, lw=1.0)
    axes[2].set_xlabel('$l (\AA)$')
    xticks = numpy.linspace(xmin, xmax, 3)
    axes[2].set_xticks(xticks)
    axes[2].set_xticklabels(xticks)
    axes[2].set_xlim(xmin, xmax)
    axes[2].set_ylabel('Blended $\Delta U(l)$ (kcal/mol)')
    #axes[2].set_ylim(-0.3, 1.2)
    #axes[2].legend(loc='best', fontsize=8, frameon=False)
    axes[2].minorticks_on()

    # Adds labels (a), (b), (c) ...
    for s, ax in zip('abcdef', axes.flat):
        ax.text(-0.2, -0.3, f'({s})', transform=ax.transAxes)

    pyplot.tight_layout()
    job_dir = '{}/convergence_plots'.format(os.getcwd())
    make_dir(job_dir)
    pyplot.savefig(f'{job_dir}/convergence.eps', dpi=300)
    pyplot.close()


def read_errors(error_file):
    ''' Read errors for each potential type from the file. '''
    cols = {}
    errors = {}
    fid = open(error_file)
    for i, line in enumerate(fid):
        args = line.split()
        if i == 0:
            for j, tt in enumerate(args[1:]):
                errors[tt] = []
                cols[tt] = j+1
    for tt in errors:
        errors[tt] = numpy.genfromtxt(error_file, usecols=(cols[tt]), skip_header=1)
    return errors


def read_target_distributions():
    ''' Read the averaged target distribution for each phase from their folders. '''
    Ps = dict(amorphous=dict(), crystal=dict())
    for phase in Ps:
        for f in glob(f'averaged_targets/{phase}/*.txt'):
            tag = f.split('.')[0].split('/')[-1]
            Ps[phase][tag] = numpy.genfromtxt(f, usecols=(1))
    return Ps


def determine_mask(Ps):
    ''' Find the mask for high-confidence regions of the P*.'''
    tol = dict(rdf=7e-3, bdf=4e-2, adf=2e-3, tdf=1e-3)
    m = dict(rdf=0, bdf=0, adf=0)

    for tag in Ps['amorphous']:
        df = tag.split('-')[0]
        m[df] = (Ps['amorphous'][tag] < tol[df]) & (Ps['crystal'][tag] < tol[df])
    return m


def read_potential_update(pt, iteration):
    ''' Read the updates for amorphous and crystal phases for an iteration. '''
    update = {}
    for p in ['amorphous', 'crystal']:
        file = glob(f'cg-updates/{p}/{pt}.update.*.{iteration}')[0]
        update[p] = numpy.genfromtxt(file, usecols=(1,2)).T
    return update


def make_dir(directory):
    ''' make directory if it does not exist '''
    if not os.path.exists(directory):
        os.mkdir(directory)


if __name__ == '__main__':
    plot_convergence(0.5)

