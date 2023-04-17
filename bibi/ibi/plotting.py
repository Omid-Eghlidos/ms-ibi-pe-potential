from matplotlib import pyplot
from glob import glob
import os
from ibi.lammps import read_tabular_potential
from ibi.distributions import read_distribution_file
from numpy import pi, loadtxt
from scipy.signal import savgol_filter
import re


pyplot.rc('font', family='Times New Roman')
pyplot.rc('font', size=8)
pyplot.rc('mathtext', fontset='stix')
plot_modes = ['energy', 'dist', 'update', 'all']

energy_colors = dict(crystal='tab:blue', amorphous='tab:green', avg='tab:purple')
dist_colors = dict(crystal='tab:blue', amorphous='tab:green', avg='tab:purple')


def plot_cmd(args):
    try:
        os.mkdir('plots')
    except FileExistsError:
        pass
    if args.mode == 'energy':
        plot_potentials(args)
    elif args.mode == 'dist':
        plot_cg_distribution(args)
    elif args.mode == 'update':
        plot_update(args)
    elif args.mode == 'all':
        plot_all(args)


def plot_potentials(args):
    phases = get_all_phases()
    potentials = get_potentials()
    fig, axes = pyplot.subplots(1, len(potentials), figsize=(6.5, 2.0))
    for phase in ['avg'] + phases:
        for i, (pt, bt) in enumerate(potentials):
            try:
                x, U, f = read_tabular_potential(phase, pt, bt, args.iteration)
            except FileNotFoundError:
                continue
            if pt == 'angle':
                x *= pi / 180.0
            pp = dict(lw = 0.75)
            if phase != 'avg':
                pp['dashes'] = (1,0.75)
                pp['label'] = phase.capitalize()
            else:
                pp['label'] = 'Blended'
            if pt != 'pair':
                U -= min(U)
            axes.flat[i].plot(x, U, c=energy_colors[phase], **pp)
    for i, (pt, _) in enumerate(potentials):
        set_axis(axes.flat[i], pt, dU=False)
    axes.flat[0].legend(frameon=0, fontsize=8)
    pyplot.tight_layout()
    pyplot.savefig(f'plots/energy.{args.iteration:03d}.png', dpi=300)


def plot_cg_distribution(args):
    phases = get_all_phases()
    distributions = get_distributions()
    fig, axes = pyplot.subplots(1, len(distributions), figsize=(6.5, 2.0))
    for phase in phases:
        for i, (dt, bt) in enumerate(distributions):
            path = f'cg-dist/{phase}/{dt}-{args.iteration}_{bt}.avg'
            d = read_distribution_file(path)
            d.p = savgol_filter(d.p, 7, 2)
            if dt == 'adf':
                d.x *= pi / 180.0  # degrees to radians
                d.p *= 180.0 / pi  # 1/degrees to 1/radians
            pp = dict(lw = 0.75)
            axes.flat[i].plot(d.x, d.p, c=dist_colors[phase]
                                 , label=phase.capitalize(), **pp)
    for i, (dt, _) in enumerate(distributions):
        set_axis(axes.flat[i], dt)
    axes.flat[0].legend(frameon=0, fontsize=8)
    pyplot.tight_layout()
    pyplot.savefig(f'plots/cg-distribution.{args.iteration:03d}.png', dpi=300)


def plot_update(args):
    phases = get_all_phases()
    pt = dict(rdf='pair', bdf='bond', adf='angle', tdf='dihedral')
    distributions = get_distributions()
    fig, axes = pyplot.subplots(1, len(distributions), figsize=(6.5,2.0))
    for phase in phases:
        for i, (dt, bt) in enumerate(distributions):
            upath = f'cg-updates/{phase}/{pt[dt]}.update.{bt}.{args.iteration}'
            _, r, dU = loadtxt(upath).T

            s = pi/180.0 if dt == 'adf' else 1.0

            pp = dict(lw = 0.75)
            axes[i].plot(s*r, dU, c=energy_colors[phase], **pp,
                                label='${}$'.format('{'+phase.capitalize()+'}'))
        axes[0].legend(frameon=0, fontsize=8)

    for i, (dt, _) in enumerate(distributions):
        set_axis(axes[i], pt[dt], no_ylim=True, dU=True)
    pyplot.tight_layout()
    pyplot.savefig(f'plots/update.{args.iteration:03d}.eps', dpi=300)


def plot_all(args):
    phases = get_all_phases()
    potentials = get_potentials()
    distributions = get_distributions()
    pt = dict(rdf='pair', bdf='bond', adf='angle', tdf='dihedral')
    fig, axes = pyplot.subplots(3, len(distributions), figsize=(6.5,6.0))
    for phase in ['avg'] + phases:
        for i, (dt, bt) in enumerate(distributions):
            s = pi/180.0 if dt == 'adf' else 1.0
            try:
                x, U, f = read_tabular_potential(phase, pt[dt], bt, args.iteration)
            except FileNotFoundError:
                continue
            if pt[dt] != 'pair':
                U -= min(U)
            if phase == 'avg':
                pp = dict(lw = 0.75)
                pp['label'] = '$U_{Blended}$'
                axes[0][i].plot(s*x, U, c=energy_colors[phase], **pp)
            else:
                upath = f'cg-updates/{phase}/{pt[dt]}.update.{bt}.{args.iteration}'
                _, r, dU = loadtxt(upath).T

                dpath0 = f'averaged_targets/{phase}/{dt}-{bt}.txt'
                dpath1 = f'cg-dist/{phase}/{dt}-{args.iteration}_{bt}.avg'
                d0 = read_distribution_file(dpath0)
                d1 = read_distribution_file(dpath1)

                pp = dict(lw = 0.75)
                pp['dashes'] = (1,0.75)
                pp['label'] = f'$U_{phase[0].capitalize()}$'
                axes[0][i].plot(s*x, U, c=energy_colors[phase], **pp)

                pp = dict(lw = 0.75)
                axes[1][i].plot(s*r, dU, c=energy_colors[phase], **pp,
                label="$\Delta U^{}_{}$".format('{'+str(args.iteration)+'}'
                                    ,'{'+phase[0].capitalize()+'}'))
                c = dist_colors[phase]
                axes[2][i].plot(s*d0.x, d0.p/s, c=c, dashes=(1.0, 0.75), **pp
                                , label="$P^*_{}$".format('{'+phase[0].capitalize()+'}'))
                axes[2][i].plot(s*d1.x, d1.p/s, c=c, **pp,
                label="$P^{}_{}$".format('{'+str(args.iteration)+'}'
                                ,'{'+phase[0].capitalize()+'}'))
    axes[0][0].legend(frameon=0, fontsize=8)
    axes[1][0].legend(frameon=0, fontsize=8)
    axes[2][0].legend(frameon=0, fontsize=8)
    for i, (dt, _) in enumerate(distributions):
        set_axis(axes[0][i], pt[dt], dU=False)
        set_axis(axes[1][i], pt[dt], no_ylim=True, dU=True)
        set_axis(axes[2][i], dt)
    pyplot.tight_layout()
    pyplot.savefig(f'plots/all.{args.iteration:03d}.png', dpi=300)


def get_all_phases():
    ''' Returns list of all phases (e.g. [crystal, amorphous]). '''
    return [os.path.basename(p) for p in glob('cg-data/*') if os.path.isdir(p)]


def get_distributions():
    ''' Returns a list of distribution types and bead types. '''
    distributions = []
    phases = get_all_phases()
    for p in sorted(glob(f'cg-dist/{phases[0]}/*-0_*.avg')):
        dt, bt = re.match('(\w+)-\d+_(\w+)\.avg', os.path.basename(p)).groups()
        distributions.append((dt, bt))
    return distributions


def get_potentials():
    potentials = []
    for p in sorted(glob(f'potentials/*.*.*.0')):
        pt, _, bt, _ = os.path.basename(p).split('.')
        potentials.append((pt, bt))
    return potentials


def set_axis(ax, t, **kwargs):
    if t in ['pair', 'rdf']:
        ax.set_xlabel('$r$ (Å)')
        ax.set_xlim(0.0, 15.0)
    elif t in ['bond', 'bdf']:
        ax.set_xlabel('$l$ (Å)')
        ax.set_xlim(2.0, 3.0)
    elif t in ['angle', 'adf']:
        ax.set_xlabel(r'$\theta$ (rad)')
        ax.set_xlim(0.5*pi, pi)
        ax.set_xticks([0.5*pi, 2/3*pi, 5/6*pi, pi],
                '$\pi/2$  $2\pi/3$ $5\pi/6$ $\pi$'.split())

    if t == 'pair':
        if kwargs.get('dU', True):
            ax.set_ylabel('$\Delta U(r)$ (kcal/mol)')
        else:
            ax.set_ylabel('$U(r)$ (kcal/mol)')
        if not kwargs.get('no_ylim', False):
            ax.set_ylim(-1.5, 5.0)
    elif t == 'bond':
        if kwargs.get('dU', True):
            ax.set_ylabel('$\Delta U(l)$ (kcal/mol)')
        else:
            ax.set_ylabel('$U(l)$ (kcal/mol)')
        if not kwargs.get('no_ylim', False):
            ax.set_ylim(0.0, 5.0)
    elif t == 'angle':
        if kwargs.get('dU', True):
            ax.set_ylabel(r'$\Delta U(\theta)$ (kcal/mol)')
        else:
            ax.set_ylabel(r'$U(\theta)$ (kcal/mol)')
        if not kwargs.get('no_ylim', False):
            ax.set_ylim(0.0, 5.0)
    elif t == 'rdf':
        ax.set_ylabel('$g(r)$')
        if not kwargs.get('no_ylim', False):
            ax.set_ylim(0.0, 5.0)
    elif t == 'bdf':
        ax.set_ylabel('$P(l)$ (1/Å)')
        if not kwargs.get('no_ylim', False):
            ax.set_ylim(0.0, ax.get_ylim()[1])
    elif t == 'adf':
        ax.set_ylabel(r'$P(\theta)$ (1/rad)')
        if not kwargs.get('no_ylim', False):
            ax.set_ylim(0.0, ax.get_ylim()[1])

