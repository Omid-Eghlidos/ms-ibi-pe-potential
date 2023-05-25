#!/usr/bin/env python3
"""
Goal
----
Plot weighted pressure of each iteration.

Inputs
------
Log file generated by the bibi code.

Deployment
----------
./plot_pressure.py
"""

import sys
import os
import numpy
from matplotlib import pyplot

pyplot.rc('font', family='Times New Roman')
pyplot.rc('font', size=8)
pyplot.rc('mathtext', fontset='stix')


def main():
    ''' Read the log file and then plot the pressures. '''
    if len(sys.argv) == 2 and sys.argv[1] == 'M':
        folders = ['0%', '25%', '50%', '75%', '100%']
    else:
        folders = ['.']
    plot_weighted_pressures(folders, logfile='ibi.log')


def plot_weighted_pressures(folders, logfile):
    ''' Read the weighted pressures from the log file. '''
    fig = pyplot.figure(figsize=(6.5,3.5))
    row, col = 2, 3
    if len(folders) == 1:
        fig = pyplot.figure(figsize=(3.5,2.8))
        row, col = 1, 1
    for i, f in enumerate(folders):
        pp = []
        ends = []
        for line in open(f'{f}/{logfile}'):
            if 'Weighted pressure' in line:
                pp.append(float(line.split()[-2]))
            elif 'Pressure converged at iteration' in line:
                ends.append(len(pp)-1)
            elif 'Pressure did not converge' in line:
                ends.append(len(pp)-1)

        ii = numpy.array(range(1, len(pp)+1))
        pp = numpy.array(pp)

        ax = fig.add_subplot(row, col, i+1)
        if len(folders) == 1:
            w = float(os.getcwd().split('/')[-1]) / 100
        else:
            w = float(f.strip('%')) / 100.0
        ax.plot(ii, pp, '.', mec='none', mfc='C0', ms=3, label=f'$w={w}$')
        ax.plot(ii[ends], pp[ends], '.', mfc='none', mec='C1', ms=2)
        ax.axhline(y=100, lw=0.5, dashes=(2,1), c='k', alpha=0.2)
        ax.axhline(y=-100, lw=0.5, dashes=(2,1), c='k', alpha=0.2)
        ax.legend(loc='best', frameon=False)
    pyplot.tight_layout()
    pyplot.savefig('weighted_pressure.png', dpi=300)


if __name__ == '__main__':
    main()
