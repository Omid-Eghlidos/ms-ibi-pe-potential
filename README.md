# ms-ibi-pe-potential
Training data and potentials for PE obtained from blended multistate IBI.

## crystal_diffusion
* diffusion.in: The LAMMPS input script that performs 100 ns crystal diffusion
for generating the arrhenius plot for different values of the crystalline weight
factor.

* 0, 25, 50, 75, 90, 100: Folders containing the tabular pair, bond, angle, potentials
trained using the crystalline weight factor values of w = 0.00, 0.25, 0.50, 0.75, and 100,
respectivley.

