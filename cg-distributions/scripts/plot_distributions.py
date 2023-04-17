#!/usr/bin/env python3
"""
Goal
----
Plot the obtained distributions.
"""

import os
import numpy
from glob import glob
# Set default plotting parameters
from matplotlib import pyplot
import plotstyle
from plotstyle import colors

def main():
    plot_distribution("rdf")
    plot_distribution("bdf")
    plot_distribution("adf")
    plot_distribution("tdf")
    plot_distribution("idf")


def plot_distribution(distribution):
    """ Read data from all the distribution files, plot, and store them in a
        separate folder.
    """
    dir_path = make_directory("distribution_plots")
    files = glob("*{}*.txt".format(distribution))
    if distribution == "rdf":
        xlabel = "$r (\AA)$"
        xlim = [0.0, 15.0]
        ylabel = "$g(r)$"
        ylim = [0.0, 15.0]
        plot_label = "Pair "
    elif distribution == "bdf":
        xlabel = "$l(\AA)$"
        xlim = [0.0, 4.0]
        ylabel = "$P(l)$"
        ylim = [0.0, 1.0]
        plot_label = "Bond "
    elif distribution == "adf":
        xlabel = "$\\theta^{\circ}$"
        xlim = [0.0, 180.0]
        ylabel = "$P(\\theta)$"
        ylim = [0.0, 0.1]
        plot_label = "Angle "
    elif distribution == "tdf":
        xlabel = "$\phi^{\circ}$"
        xlim = [-180.0, 180.0]
        ylabel = "$P(\\phi)$"
        ylim = [0.0, 0.1]
        plot_label = "Dihedral "
    elif distribution == "idf":
        xlabel = "$\psi^{\circ}$"
        xlim = [0.0, 180.0]
        ylabel = "$P(\\psi)$"
        ylim = [0.0, 0.1]
        plot_label = "Improper "

    for f in files:
        pair_type = f.strip(distribution).strip('.txt').split('_')[-1]
        data = numpy.genfromtxt(f)
        pyplot.figure(figsize=(6.5,4.0))
        ax = pyplot.subplot(111)
        ax.plot(data[:,0], data[:,1], color=colors['blue'], label=plot_label + pair_type)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        pyplot.legend().draw_frame(0)
        pyplot.tight_layout()
        pyplot.savefig('{}/{}.png'.format(dir_path, distribution + "_" + pair_type\
                                         , dpi=400, bbox_inches='tight'))
        pyplot.close()


def make_directory(dir_name):
    """ make directory in the current working directory if it does not exist """
    dir_path = os.path.join(os.getcwd(), dir_name)
    try:
        os.mkdir(dir_path)
        exist = False
    except OSError:
        exist = True
    return dir_path


if __name__ == "__main__":
    main()
