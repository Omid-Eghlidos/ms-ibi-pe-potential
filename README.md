# ms-ibi-pe-potential
The contents are a CG data file, potentials for PE obtained from blended MS-IBI,
and a LAMMPS input script.

## data
Contains LAMMPS format data file for CG crystal phase of PE. The name of the
file ("a7b11c60") denote that the crystal unit cell is repeated 7, 11, and 60
along axes a, b, and c, respectively.

## potentials
This folder contains the tabular pair, bond, angle, potentials trained for
each value of the crystalline weight factor w (%) = 0, 25, 50, 75, and 100.
NOTE: Name of each file contains the potential type, weight, and bead type.

## diffusion.in
The LAMMPS input script to perform 100 ns crystal diffusion for generating the
Arrhenius plot for different values of w.

### Deployment
To run the crystal diffusion at a specific temperature using a potential trained
with a specific w (e.g., w = 25 and T = 360 K) on a workstation/cluster without
the task scheduler, run:

mpiexec -n <np> lmp_mpi -in diffusion.in  -v w 25 -v T 360

* <np> (integer): number of processors to use

