#include "catch.hpp"
#include "input.h"
#include "lammps_data.h"
#include "cgsystem.h"
#include "distribution.h"
#include "string_tools.h"

using namespace cg;
TEST_CASE("rdfs are computed correctly", "[rdf]") {
    InputSettings input;
    input.lmp_dump_path = "../test-data/test-2C/pe-2.lammpstrj";    
    input.lmp_data_path = "../test-data/test-2C/pe-2.data";
    input.atom_types = {{1, "h"}, {2, "c"}, {3, "c1"}};
    input.rdf.lo = 0.0;
    input.rdf.hi = 16.0;
    input.rdf.delta = 0.05;
    BeadType E;
    E.atoms = {"c*", "c*"};
    E.weights = {0.5, 0.5};
    input.bead_types = {{"E", E}}; 

    auto lmp_data = LAMMPS_Data(input);
    auto cg_system = CgSystem(input, lmp_data);
    auto distribution = Distribution(cg_system, input);

    Histogram rdf;
    auto t = cg_system.unique_pair_types()[0];
    #pragma omp parallel for
    for (int ts=0; ts<lmp_data.num_timesteps(); ++ts) {
        auto temp = distribution.compute_rdf(t, ts);
        #pragma omp critical 
        rdf += temp;
    }
    rdf.scale(1.0 / lmp_data.num_timesteps(), false);

    std::fstream fid("../test-data/test-2C/rdf-ref.txt", std::ios::in);
    double error = 0.0, total = 0.0;
    for (size_t i=0; i<rdf.size(); ++i) {
        auto row = split(read_line(fid));
        if (!fid) break;
        auto r = str2dbl(row[0]), g = str2dbl(row[1]);
        total += g*g;
        error += (rdf.histogram[i]-g)*(rdf.histogram[i]-g);       
    }
    REQUIRE(error/total < 1e-6);
}

