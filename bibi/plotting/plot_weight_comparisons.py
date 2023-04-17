#!/usr/bin/env python3
"""
Goal
----
Plot each distribution and potential of each type for different weight of blended IBI.

Inputs
-------
Copy the final distribution files for amorphous and crystal to spearate folder
with their name as the weight were used to train the potential, e.g., if the
weight is 50% the folder name should be 50% and should has two phases of
amorphous and crystal, e.g., 50%/amorphous/ and 50%/crystal/. For target
distributions use the name target/amorphous/ and target/crystal/.
For potrntial copy the final potential files into the main folder, e.g., 50%.

Deployment
----------
./plot_different_weight_distribution_comparison.py
"""


import os
from glob import glob
import numpy
import scipy.stats
from scipy.signal import savgol_filter
import math
# Set default plotting parameters
import matplotlib
from matplotlib import pyplot
from matplotlib.ticker import MaxNLocator
import plotstyle
from plotstyle import colors, tcolors


tcolours = list(tcolors)
colours = list(colors)


def main():
    """ Read the errors for different from the files """
    # Go through all the folders
    folders = [0, 25, 50, 75, 100, "target"]

    # Read all distributions data
    dist_data = read_all_distributions_data(folders)
    # Plot all the distributions on the same figure for different weights
    plot_different_weight_distributions(dist_data)
    # Compute errors as the difference between final and target PDF
    errors = compute_errors(dist_data)
    # Distribution error bar chart
    plot_bar_chart_l2_errors(errors)

    # Read all potentials data
    U_data = read_all_potentials_data(folders)
    # Plot all the final potentials on the same figure for different weights
    plot_different_weight_potentials(U_data)


def compute_errors(dist_data):
    """ Compute errors by finding the difference between normalized distributions """
    # First go through each distribution function, then different weights, finally phase
    phases = list(dist_data.keys())
    weights = list(dist_data[phases[0]].keys())
    dfs = list(dist_data[phases[0]][weights[0]].keys())

    errors = {}
    for p in phases:
        if p not in errors:
            errors[p] = {}
        for df in dfs:
            if df not in errors[p]:
                errors[p][df] = {}
            for w in weights:
                if w == "target":
                    continue
                if df not in errors[p]:
                    errors[p][df][w] = []
                x = dist_data[p][w][df][:,0]
                dx = x[1] - x[0]
                pdf_i = dist_data[p][w][df][:,1]
                pdf_t = dist_data[p]["target"][df][:,1]
                e1 = numpy.sqrt(numpy.trapz((pdf_i - pdf_t)**2, dx=dx))
                e2 = numpy.sqrt(numpy.trapz(pdf_t**2, dx=dx))
                #errors[p][df][w] = numpy.linalg.norm(pdf_i - pdf_t) / numpy.linalg.norm(pdf_t)
                errors[p][df][w] = e1 /e2

    E = {}
    for w in weights:
        if w == "target":
            continue
        E[w] = 0
        for df in dfs:
            e = numpy.sqrt((errors["amorphous"][df][w])**2\
                         + (errors["crystal"][df][w])**2)
            print("{}, w = {:4.2f}, E = {:6.4f}".format(df, w, E[w]))
            E[w] += e**2
        E[w] = numpy.sqrt(E[w])
        print("Total, w = {:4.2f}, E = {:6.3f}\n".format(w, E[w]))
    return errors


def distribution_to_pdf(df, x, P):
    """ Convert the distribution to probability density function by dividing by
        the sum of all the values of the distribution """
    if df == "rdf":
        return P
    dx = x[1] - x[0]
    Sum = numpy.trapz(P, dx=dx)
    pdf = numpy.array(P / Sum)
    return pdf


def plot_bar_chart_l2_errors(errors):
    """ Find the l2 norm of errors of amorphous and crystal phase for each
        weights in a bar chart """
    # First go through each distribution function, then different weights, finally phase
    phases = list(errors.keys())
    dfs = list(errors[phases[0]].keys())
    weights = list(errors[phases[0]][dfs[0]].keys())
    # Plot
    pyplot.clf()
    pyplot.figure(figsize=(3.5, 2.8))
    # The matrix in which each figure is assigned to an element
    fig_rows = 2
    fig_cols = 1
    # Current figure index
    fig_id = 0
    # Width of each bar
    width = 2.0
    # Bar center locations
    xticks = numpy.arange(len(weights)) * 8
    ymax = []
    for p in phases:
        fig_id += 1
        ax = pyplot.subplot(int("{}{}{}".format(fig_rows, fig_cols, fig_id)))
        for i, df in enumerate(dfs):
            if df == "ADF":
                x = [w-width for w in xticks]
            elif df == "BDF":
                x = [w for w in xticks]
            elif df == "RDF":
                x = [w+width for w in xticks]
            y = [numpy.linalg.norm(errors[p][df][w]) for w in weights]
            ymax.append(max(y))
            if p == "amorphous":
                ax.bar(x, y, width=width, color=tcolours[i], label="${}$".format('{'+df+'}'))
            else:
                ax.bar(x, y, width=width, color=tcolours[i])
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        #ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', which='both', bottom=False)
        ax.set_title("${}$".format('{'+p.capitalize()+'}'), fontsize=10)
        ax.set_ylabel("Error")
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"$w = {w/100}$" for w in weights])
        #ax.set_xlim(0, 100)
        ax.set_ylim(0.0, round(max(ymax), 1))
        ax.minorticks_off()
        ax.legend(loc="best", fontsize=8, frameon=False)
    pyplot.tight_layout()
    pyplot.savefig("errors_bar_plot.png", dpi=300)
    pyplot.close()


def read_all_potentials_data(folders):
    """ Read and store the data for each potential function of each weights
        from the provided list of folders """
    # Final average potential data of each type
    U_data = {}
    for w in folders:
        if w == "target":
            continue
        if w not in U_data:
            U_data[w] = {}
        # All potential files
        potential_files = "{}%/*.*".format(w)
        for f in sorted(glob(potential_files)):
            df = f.split('/')[-1].split('.')[0]
            U_data[w][df] = numpy.genfromtxt(f, skip_header=3)
    return U_data


def plot_different_weight_potentials(U_data):
    """ Read and plot potential of each type on the same figure """
    # First go through each potential function, then different weights
    weights = list(U_data.keys())
    dfs = list(U_data[weights[0]].keys())

    pyplot.clf()
    pyplot.figure(figsize=(6.5, 2.0))
    for i, df in enumerate(dfs):
        ax = pyplot.subplot(1, 3, i+1)
        set_axes_labels(df, ax)
        for w in weights:
            x = U_data[w][df][:,1]
            y = U_data[w][df][:,2]
            y = savgol_filter(y, 11, 2)
            if df != "pair":
                y -= min(y)
            w /= 100
            label = "${}$".format("{w="+str(w)+"}")
            ax.plot(x, y, label=label, lw=1.0)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.minorticks_on()
        if df == "angle":
            ax.legend(loc="best", fontsize=8, frameon=False)
    pyplot.tight_layout()
    pyplot.savefig("final_different_weight_potentials.png", dpi=300)
    pyplot.close()


def read_all_distributions_data(folders):
    """ Read and store the data for each distribution function of each weights for
        each phase from the provided list of folders """
    # Weights
    weights = [w/100 for w in folders if w != "target"]
    # Distributions data of each type
    phases = ["amorphous", "crystal"]
    dist_data = {"amorphous": {}, "crystal": {}}

    for phase in phases:
        if phase not in dist_data:
            dist_data[phase] = {}
        for w in folders:
            if w not in dist_data[phase]:
                dist_data[phase][w] = {}
            if w == "target":
                dist_files = "{}/{}/*.txt".format(w, phase)
            else:
                dist_files = "{}%/{}/*.avg".format(w, phase)
            for f in sorted(glob(dist_files)):
                df = f.split('/')[-1].split('-')[0].upper()
                dist_data[phase][w][df] = numpy.genfromtxt(f)
    return dist_data


def plot_different_weight_distributions(dist_data):
    """ Read and plot each distribution function of different weights on the same
        figure """
    # First go through each distribution function, then different weights, finally phase
    phases = list(dist_data.keys())
    weights = list(dist_data[phases[0]].keys())
    dfs = list(dist_data[phases[0]][weights[0]].keys())

    for p in phases:
        pyplot.clf()
        pyplot.figure(figsize=(6.5, 2.0))
        for i, df in enumerate(dfs):
            ax = pyplot.subplot(1, 3, i+1)
            set_axes_labels(df, ax)
            for w in weights:
                x = dist_data[p][w][df][:,0]
                y = dist_data[p][w][df][:,1]
                if df == "ADF":
                    y *= (180.0 / numpy.pi)
                y = savgol_filter(y, 11, 2)
                if w == "target":
                    ax.plot(x, y, "--k", label="${}$".format(w.capitalize()), lw=1.0)
                else:
                    w /= 100
                    label = "${}$".format("{w="+str(w)+"}")
                    ax.plot(x, y, label=label, lw=1.0)
            ax.tick_params(axis='x', which='minor', direction='in')
            ax.tick_params(axis='y', which='minor', direction='in')
            ax.minorticks_on()
            if df == "ADF":
                ax.legend(title="${}$".format(p.capitalize()), loc="best", fontsize=8, frameon=False)
        pyplot.tight_layout()
        pyplot.savefig("{}_different_weight_distributions.png".format(p), dpi=300)
        pyplot.close()


def set_axes_labels(dist_type, ax):
    """ Based on the distribution chooses the appropriate axes labels """
    # RDF
    if dist_type == 'RDF' or dist_type == "pair":
        xmax = 15.0
        ax.set_xlim(0.0, xmax)
        xticks = numpy.round(numpy.linspace(0.0, xmax, 7), 2)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
        ax.set_xlabel('$r$ ($\AA$)')
        if dist_type == "RDF":
            ax.set_ylim(0.0, 3.5)
            ax.set_ylabel('$g(r)$')
        if dist_type == "pair":
            ax.set_ylabel('$U(r)$ (kcal/mol)')
            ax.set_ylim(-1.0, 1.0)
    # BDF
    if dist_type == 'BDF' or dist_type == "bond":
        xmin, xmax = 2.0, 3.0
        ax.set_xlim(xmin, xmax)
        xticks = numpy.linspace(xmin, xmax, 3)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
        ax.set_xlabel('$l$ ($\AA$)')
        if dist_type == "BDF":
            #ax.set_ylim(0.0, 20.0)
            ax.set_ylabel('$P(l)$')
        if dist_type == "bond":
            ax.set_ylabel('$U(l)$ (kcal/mol)')
            ax.set_ylim(0.0, 5.0)
    # ADF
    if dist_type == 'ADF' or dist_type == "angle":
        ax.set_xlim(90.0, 180.0)
        xticks = numpy.linspace(90.0, 180.0, 4)
        ax.set_xticks(xticks)
        xticklabels = ["$\pi/2$", "$2\pi/3$", "$5\pi/6$", "$\pi$"]
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('$\\theta$$^\circ$')
        if dist_type == "ADF":
            #ax.set_ylim(0.0, 1.0)
            ax.set_ylabel('$P$($\\theta$)')
        if dist_type == "angle":
            ax.set_ylabel('$U$($\\theta$) (kcal/mol)')
            ax.set_ylim(0.0, 5.0)
    # TDF
    if dist_type == 'TDF' or dist_type == "dihedral":
        ax.set_xlim(-180.0, 180.0)
        xticks = numpy.linspace(-180.0, 180.0, 5)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
        ax.set_xlabel('$\\phi$$^\circ$')
        if dist_type == "TDF":
            #ax.set_ylim(0.0, 1.0)
            ax.set_ylabel('$P$($\\phi$)')
        if dist_type == "dihedral":
            ax.set_ylabel('$U$($\\phi$) (kcal/mol)')
            ax.set_ylim(0.0, 5.0)
    # IDF
    if dist_type == 'IDF' or dist_type == "imporoper":
        ax.set_xlim(0.0, 180.0)
        xticks = numpy.linspace(0.0, 180.0, 7)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
        ax.set_xlabel('$\\psi$$^\circ$')
        if dist_type == "IDF":
            #ax.set_ylim(0.0, 1.0)
            ax.set_ylabel('$P$($\\psi$)')
        if dist_type == "imporoper":
            ax.set_ylabel('$U$($\\psi$)')


def make_dir(directory):
    """ make directory if it does not exist """
    if not os.path.exists(directory):
        os.mkdir(directory)


if __name__ == "__main__":
    main()

