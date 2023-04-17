#!/usr/bin/env python3
"""
Goal
----
Go through all the distribution files (e.g., RDF, BDF, ADF, TDF, and IDF) of
every type, find their final distributions obtained using corresponding
potentials and plot them and the target distributions on a plot.

Input
------
Averaged target distributions (inside averaged_targets/ folder) and the final
iteration's distributions (inside the cg-dist/ folder) both in the main directory.

Deployment
----------
In the main folder run:
./plot_ibi_results.py
"""


import os
import sys
import shutil
from glob import glob
import numpy
import scipy.stats
from scipy.signal import savgol_filter
# Set default plotting parameters
import matplotlib
from matplotlib import pyplot
from matplotlib.colors import TABLEAU_COLORS as colors


pyplot.rc('font', family='Times New Roman')
pyplot.rc('font', size=8)
pyplot.rc('mathtext', fontset='stix')
plot_modes = ['energy', 'dist', 'update']

colours = list(colors.values())


def main():
    """ Go through the files and check if each type exists plot it with its CI """
    # Output directory name and path (manual)
    main_folder = make_folder(os.getcwd(), "final_results")

    # Find and copy the distribution and potential's final iteration
    max_iter = find_final_distributions_potentials(main_folder)

    # Initializing
    rdfs, bdfs, adfs, tdfs, idfs = [], [], [], [], []
    # Total number of figures
    num_figures = 0
    # Read and store the data for each distribution of each phase
    for phase in ["amorphous", "crystal"]:
        if len(glob('{}/{}/rdf*'.format(main_folder, phase))) != 0:
            num_figures += 1
            rdfs.append(read_distributions(main_folder, phase, 'rdf'))
        if len(glob('{}/{}/bdf*'.format(main_folder, phase))) != 0:
            num_figures += 1
            bdfs.append(read_distributions(main_folder, phase, 'bdf'))
        if len(glob('{}/{}/adf*'.format(main_folder, phase))) != 0:
            num_figures += 1
            adfs.append(read_distributions(main_folder, phase, 'adf'))
        if len(glob('{}/{}/tdf*'.format(main_folder, phase))) != 0:
            num_figures += 1
            tdfs.append(read_distributions(main_folder, phase, 'tdf'))
        if len(glob('{}/{}/idf*'.format(main_folder, phase))) != 0:
            num_figures += 1
            idfs.append(read_distributions(main_folder, phase, 'idf'))
    if rdfs:
        # Plot RDFs, if there are any
        plot_rdfs(main_folder, rdfs)
    if bdfs:
        # Plot BDFs, if there are any
        plot_bdfs(main_folder, bdfs)
    if adfs:
        # Plot ADFs, if there are any
        plot_adfs(main_folder, adfs)
    if tdfs:
        # Plot TDFs, if there are any
        plot_tdfs(main_folder, tdfs)
    if idfs:
        # Plot IDFs, if there are any
        plot_idfs(main_folder, idfs)
    num_figures = int(num_figures / 2.0)

    # Plot all the distributions in one plot
    plot_all_distributions(main_folder, num_figures, rdfs, bdfs, adfs, tdfs, idfs)

    # Plot first and last potential
    plot_all_potentials(main_folder, num_figures, max_iter)


def find_final_distributions_potentials(main_folder):
    """ Go through the trained distributions and potentials and copy them into
        a folder named final_results to be used """
    # Find the max iteration
    max_iter = 0
    for f in glob("cg-dist/amorphous/*.avg"):
        name = f.split('/')[-1].split(".avg")[0]
        iteration = int(name.split('_')[0].split('-')[1])
        if iteration > max_iter:
            max_iter = iteration

    # Copy different phases final distributions
    for phase in ["amorphous", "crystal"]:
        phase_folder = make_folder(main_folder, phase)
        # Find and copy the target and final iteration distributions to the main folder
        for f in glob("averaged_targets/{}/*.txt".format(phase)):
            original = os.path.join(os.getcwd(), f)
            f = f.split('/')[-1]
            target = os.path.join(main_folder, phase, f)
            shutil.copyfile(original, target)
        for f in glob("cg-dist/{}/*-{}_*.avg".format(phase, max_iter)):
            original = os.path.join(os.getcwd(), f)
            f = f.split('/')[-1]
            target = os.path.join(main_folder, phase, f)
            shutil.copyfile(original, target)
    # Find and copy the target and final iteration distributions to the main folder
    for it in [0, max_iter]:
        for f in glob("potentials/*.{}".format(it)):
            original = os.path.join(os.getcwd(), f)
            f = f.split('/')[-1]
            target = os.path.join(main_folder, f)
            shutil.copyfile(original, target)
    return max_iter


def read_distributions(main_folder, phase, dist_type):
    """ Go through every distribution files and read the data for each rdf type """
    # For each type store the data of all system
    data = {}
    for f in sorted(glob('{}/{}/{}*.*'.format(main_folder, phase, dist_type))):
        tag = f.split('/')[-1].split('-')[1].split('.')[0]
        data[tag] = numpy.genfromtxt(f)
    return data


def plot_rdfs(job_dir, rdfs):
    """ Go through each RDF type and plot it with its CI and store it """
    pyplot.clf()
    pyplot.figure(figsize=(3.5,2.8))
    fig, ax = pyplot.subplots(2,1, gridspec_kw={"height_ratios": [3, 1]})
    plot_distribution("RDF", rdfs, ax[0])
    ax[0].set_xlabel(None)
    ax[0].set_xticklabels([])
    ax[0].legend(loc="best", fontsize=8, frameon=False)
    plot_distribution_difference("RDF", rdfs, ax[1])
    ax[1].set_ylim(-0.5, 0.5)
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["top"].set_visible(False)
    ax[1].legend(loc="lower left", fontsize=8, frameon=False)
    pyplot.tight_layout()
    pyplot.savefig(f'{job_dir}/rdfs.png', dpi=400)
    pyplot.close()


def plot_bdfs(job_dir, bdfs):
    """ Go through each BDF type and plot it with its CI and store it """
    pyplot.clf()
    pyplot.figure(figsize=(3.5,2.8))
    fig, ax = pyplot.subplots(2,1, gridspec_kw={"height_ratios": [3, 1]})
    plot_distribution("BDF", bdfs, ax[0])
    ax[0].set_xlim(1.5, 3.5)
    ax[0].set_xlabel(None)
    ax[0].set_xticklabels([])
    ax[0].legend(loc="best", fontsize=8, frameon=False)
    plot_distribution_difference("BDF", bdfs, ax[1])
    ax[1].set_xlim(1.5, 3.5)
    ax[1].set_ylim(-5.0, 5.0)
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["top"].set_visible(False)
    ax[1].legend(loc="lower left", fontsize=8, frameon=False)
    pyplot.tight_layout()
    pyplot.savefig(f'{job_dir}/bdfs.png', dpi=400)
    pyplot.close()


def plot_adfs(job_dir, adfs):
    """ Go through each ADF type and plot it with its CI and store it """
    pyplot.clf()
    pyplot.figure(figsize=(3.5,2.8))
    fig, ax = pyplot.subplots(2,1, gridspec_kw={"height_ratios": [3, 1]})
    plot_distribution("ADF", adfs, ax[0])
    ax[0].set_xlabel(None)
    ax[0].set_xticklabels([])
    ax[0].legend(loc="best", fontsize=8, frameon=False)
    plot_distribution_difference("ADF", adfs, ax[1])
    ax[1].set_ylim(-0.05, 0.05)
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["top"].set_visible(False)
    ax[1].legend(loc="lower left", fontsize=8, frameon=False)
    pyplot.tight_layout()
    pyplot.savefig(f'{job_dir}/adfs.png', dpi=400)
    pyplot.close()


def plot_tdfs(job_dir, tdfs):
    """ Go through each TDF type and plot it with its CI and store it """
    pyplot.clf()
    pyplot.figure(figsize=(3.5,2.8))
    fig, ax = pyplot.subplots(2,1, gridspec_kw={"height_ratios": [3, 1]})
    plot_distribution("TDF", tdfs, ax[0])
    plot_distribution_difference("TDF", tdfs, ax[1])
    pyplot.tight_layout()
    pyplot.savefig('{}/tdfs.png'.format(job_dir), dpi=400)
    pyplot.close()


def plot_idfs(job_dir, idfs):
    """ Go through each IDF type and plot it with its CI and store it """
    pyplot.clf()
    pyplot.figure(figsize=(3.5,2.8))
    ax = pyplot.subplot(111)
    plot_distribution("IDF", idfs, ax)
    pyplot.tight_layout()
    pyplot.savefig('{}/idfs.png'.format(job_dir), dpi=400)
    pyplot.close()


def plot_all_distributions(main_folder, num_figures, rdfs, bdfs, adfs, tdfs, idfs):
    """ Plot all the distributions in one plot """
    pyplot.clf()
    pyplot.figure(figsize=(6.5, 2.0))
    # The matrix in which each figure is assigned to an element
    fig_rows = 1
    fig_cols = 3
    # Plot all the distributions
    # Current figure index
    fig_id = 0
    # RDFs
    if len(rdfs) != 0:
        fig_id += 1
        ax = pyplot.subplot(fig_rows, fig_cols, fig_id)
        plot_distribution("RDF", rdfs, ax)
    # BDFs
    if len(bdfs) != 0:
        fig_id += 1
        ax = pyplot.subplot(fig_rows, fig_cols, fig_id)
        plot_distribution("BDF", bdfs, ax)
    # ADFs
    if len(adfs) != 0:
        fig_id += 1
        ax = pyplot.subplot(fig_rows, fig_cols, fig_id)
        plot_distribution("ADF", adfs, ax)
    # TDFs
    if len(tdfs) != 0:
        fig_id += 1
        ax = pyplot.subplot(fig_rows, fig_cols, fig_id)
        plot_distribution("TDF", tdfs, ax)
    # IDFs
    if len(idfs) != 0:
        fig_id += 1
        ax = pyplot.subplot(fig_rows, fig_cols, fig_id)
        plot_distribution("IDF", idfs, ax)

    pyplot.tight_layout()
    pyplot.savefig(f'{main_folder}/all_distributions.eps')
    pyplot.close()


def plot_distribution(dist_type, distributions, ax):
    """ For each distribution plot all types on the same plot for both AA & CG """
    for j, distribution in enumerate(distributions):
        if j == 0:
            phase = "amorphous"
            line = '--k'
        else:
            phase = "crystal"
            line = ':k'
        for tag, df in distribution.items():
            x, y = df[:, 0], df[:, 1]
            if len(tag.split('_')) == 1:
                label = set_plot_labels(dist_type, "Target", phase, tag)
                ax.plot(x, y, line, label=label, zorder=2)
            else:
                label = set_plot_labels(dist_type, "Trained", phase, tag)
                ax.plot(x[0:-1:3], y[0:-1:3], label=label, zorder=1)
    set_axes_labels(dist_type, ax)
    ax.minorticks_on()
    if dist_type == "ADF":
        ax.legend(loc='best', fontsize=8, frameon=False)


def plot_distribution_difference(dist_type, distributions, ax):
    """ Find the difference between target and distributions for each dist. function """
    # Target distribution of each phase
    xt, yt = {}, {}
    # Final distribution of each phase
    xf, yf = {}, {}
    for j, distribution in enumerate(distributions):
        if j == 0:
            phase = "amorphous"
        else:
            phase = "crystal"
        for tag, df in distribution.items():
            if len(tag.split('_')) == 1:
                xt[phase], yt[phase] = df[:, 0], df[:, 1]
            else:
                xf[phase], yf[phase] = df[:, 0], df[:, 1]
    # Find the difference of each phase
    max_, min_ = 0.0, 0.0
    for phase in xt:
        dy = yf[phase] - yt[phase]
        dy = savgol_filter(dy, 11, 2)
        ax.plot(xt[phase], dy, label="${}$".format('{'+phase.capitalize()+'}'))
        if max(dy) > max_:
            max_ = max(dy)
        if min(dy) < min_:
            min_ = min(dy)
    set_axes_labels(dist_type, ax, "difference")
    ax.set_ylim(1.1*min_, 1.1*max_)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.minorticks_on()


def set_plot_labels(dist_type, system, phase, tag):
    """ Based on the distribution chooses the appropriate plot labels """
    label = ""
    if system == "Target":
        tag = '{' + tag + '}'
        # RDF or Pair
        if dist_type == "RDF":
            label = "$g^*_{}$".format('{'+phase[0].capitalize()+'}')
        if dist_type == "pair":
            label = "$U^0(r_{})$".format(tag)
        # BDF or bond
        if dist_type == "BDF":
            label = "$P^*_{}$".format('{'+phase[0].capitalize()+'}')
        if dist_type == "bond":
            label = "$U^0(l_{})$".format(tag)
        # ADF or angle
        if dist_type == "ADF":
            label = "$P^*_{}$".format('{'+phase[0].capitalize()+'}')
        if dist_type == "angle":
            label = "$U^0$"
        # TDF or dihedral
        if dist_type == "TDF":
            label = "$P^*(\phi_{}^{})$".format(tag, '{'+phase.capitalize()+'}')
        if dist_type == "dihedral":
            label = "$U^0(\phi_{})$".format(tag)
        # IDF or imporoper
        if dist_type == 'IDF':
            label = "$P^*(\psi_{})^{}$".format(tag, '{'+phase.capitalize()+'}')
        if dist_type == "imporoper":
            label = "$U^0(\psi_{})$".format(tag)

    elif system == "Trained":
        #it = '{' + tag.split('_')[0] + '}'
        it = "{CG}"
        tag = '{' + tag.split('_')[1] + '}'
        # RDF or pair
        if dist_type == "RDF":
            label = "$g_{}$".format('{'+phase[0].capitalize()+'}')
        if dist_type == "pair":
            label = "$U^{}(r_{})$".format(it, tag)
        # BDF or bond
        if dist_type == "BDF":
            label = "$P_{}$".format('{'+phase[0].capitalize()+'}')
        if dist_type == "bond":
            label = "$U^{}(l_{})$".format(it, tag)
        # ADF or angle
        if dist_type == "ADF":
            label = "$P_{}$".format('{'+phase[0].capitalize()+'}')
        if dist_type == "angle":
            label = f"$U^{it}$"
        # TDF or dihedral
        if dist_type == "TDF":
            label = "$P^{}(\phi_{}^{})$".format(it, tag, '{'+phase.capitalize()+'}')
        if dist_type == "dihedral":
            label = f"$U^{it}$"
        # IDF or imporoper
        if dist_type == "IDF":
            label = "$P^{}(\psi_{}^{})$".format(it, tag, '{'+phase.capitalize()+'}')
        if dist_type == "imporoper":
            label = "$U^{}(\psi_{})$".format(it, tag)
    return label


def set_axes_labels(dist_type, ax, plot_type=None):
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
            if plot_type:
                ax.set_ylabel('$\Delta g(r)$')
        if dist_type == "pair":
            ax.set_ylabel('$U(r)$ (kcal/mol)')
            ax.set_ylim(-0.5, 1.0)
    # BDF
    if dist_type == 'BDF' or dist_type == "bond":
        ax.set_xlim(0.0, 5.0)
        xticks = numpy.round(numpy.linspace(0.0, 5.0, 6), 2)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
        ax.set_xlabel('$l$ ($\AA$)')
        if dist_type == "BDF":
            ax.set_ylim(bottom=0.0)
            ax.set_ylabel('$P(l)$')
            if plot_type:
                ax.set_ylabel('$\Delta P(l)$')
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
            ax.set_ylim(bottom=0.0)
            ax.set_ylabel('$P$($\\theta$)')
            if plot_type:
                ax.set_ylabel('$\Delta P(\\theta)$')
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
            ax.set_ylim(bottom=0.0)
            ax.set_ylabel('$P$($\\phi$)')
            if plot_type:
                ax.set_ylabel('$\Delta P(\\phi)$')
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
            ax.set_ylim(bottom=0.0)
            ax.set_ylabel('$P$($\\psi$)')
            if plot_type:
                ax.set_ylabel('$\Delta P(\\psi)$')
        if dist_type == "imporoper":
            ax.set_ylabel('$U$($\\psi$)')


def plot_all_potentials(main_folder, num_figures, max_iter):
    """ Plot the first and the last potential on the same figure for each type """
    pyplot.clf()
    pyplot.figure(figsize=(6.5, 2.0))
    fig_rows = 1
    fig_cols = 3
    fig_id = 0

    potential_functions = ["pair", "bond", "angle", "dihedral"]
    for pf in potential_functions:
        potential_files = glob(os.path.join(main_folder, "{}.table.*".format(pf)))
        if len(potential_files) != 0:
            fig_id += 1
            ax = pyplot.subplot(fig_rows, fig_cols, fig_id)
            for f in potential_files:
                tag = f.split('.')[2]
                # U_0
                if f"{tag}.0" in f:
                    U_0 = numpy.genfromtxt(f, skip_header=3)
                    x, y = U_0[:,1], U_0[:,2]
                    if pf != "pair":
                        y -= min(U_0[:,2])
                    else:
                        y -= U_0[-1,2]
                    label = set_plot_labels(pf, "Target", "avg", tag)
                    ax.plot(x, y, 'k', lw=1.0, label=label)
                # U_max
                if f"{tag}.{max_iter}" in f:
                    U_max = numpy.genfromtxt(f, skip_header=3)
                    tag = str(max_iter) + '_' + f.split('.')[2]
                    x, y = U_max[:,1], U_max[:,2]
                    if pf != "pair":
                        y -= min(U_max[:,2])
                    else:
                        y -= U_max[-1,2]
                    label = set_plot_labels(pf, "Trained", "avg", tag)
                    ax.plot(x, y, color= colors["tab:purple"], lw=1.0, label=label)
            set_axes_labels(pf, ax)
            ax.minorticks_on()
            if pf == "angle":
                ax.legend(loc='best', fontsize=8, frameon=False)

    pyplot.tight_layout()
    pyplot.savefig(f'{main_folder}/all_potentials.eps')
    pyplot.close()


def make_folder(path, folder_name):
    """ make directory in the given path if it does not exist """
    folder_path = os.path.join(path, folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

if __name__ == '__main__':
    main()

