#include "lammps_data.h"
#include "string_tools.h"
#include "cgsystem.h"
#include "input.h"
#include "distribution.h"

const char *help_msg = 
"cg-distributions - computes coarse-grained structure distributions\n"
"\n"
"usage: cg-distributions [arguments]\n"
"\n"
"Arguments:\n"
"    -i <inputfile>    Use input file (default cg.ini)\n"
"    -v                Verbose mode (print chains)\n"
"    -h                Print help (this message) and exit\n\n";

int main(int n, char **argv) {    
    std::string input = "cg.ini";
    bool print_chains = false;
    std::vector<std::string> args(argv, argv+n);
    for (auto a=args.begin()+1; a!=args.end(); ++a) {
        if (*a=="-i" && ++a!=args.cend()) {
            input = *a;
        }
        else if (*a=="-v") {
            print_chains = true;
        }
        else {
            std::cout << help_msg;
            return 0;
        }
    }    
    auto settings = cg::read_inputfile(input);
    cg::LAMMPS_Data   lmp(settings);
    cg::CgSystem      system(settings, lmp);
    cg::Distribution  distribution(system, settings);

    if (print_chains) {
        system.print_cg_chains();
    }
    std::cout << "Computing histograms for " << lmp.num_timesteps() << " steps.\n";
    if (settings.rhobar.enabled) {
        distribution.compute_density_field();
    }
    if (settings.bdf.enabled) {
        distribution.compute_bond_correlations();
    }
    if (settings.rdf.enabled) {
        distribution.compute_pair_correlations();
    }
    if (settings.adf.enabled) {
        distribution.compute_angle_correlations();
    }
    if (settings.tdf.enabled) {
        distribution.compute_torsion_correlations();
    }
    if (settings.idf.enabled) {
        distribution.compute_improper_correlations();
    }
}

