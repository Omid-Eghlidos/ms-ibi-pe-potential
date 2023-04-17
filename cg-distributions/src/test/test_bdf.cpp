#include "catch.hpp"
#include "input.h"
#include "lammps_data.h"
#include "cgsystem.h"
#include "distribution.h"
#include "string_tools.h"

using namespace cg;
TEST_CASE("bdfs are computed correctly", "[bdf]") {
    InputSettings input;
    input.lmp_dump_path = "../test-data/test-2C/pe-2.lammpstrj";    
    input.lmp_data_path = "../test-data/test-2C/pe-2.data";
    input.atom_types = {{1, "h"}, {2, "c"}, {3, "c1"}};
    input.bdf.lo = 0.0;
    input.bdf.hi = 10.0;
    input.bdf.delta = 0.025;
    BeadType E;
    E.atoms = {"c*", "c*"};
    E.weights = {0.5, 0.5};
    input.bead_types = {{"E", E}}; 

    auto lmp_data = LAMMPS_Data(input);
    auto cg_system = CgSystem(input, lmp_data);
    auto distribution = Distribution(cg_system, input);

    Histogram bdf;
    auto t = cg_system.unique_pair_types()[0];
    #pragma omp parallel for
    for (int ts=0; ts<lmp_data.num_timesteps(); ++ts) {
        auto temp = distribution.compute_bdf(t, ts);
        #pragma omp critical 
        bdf += temp;
    }
    bdf.normalize();

    std::fstream fid("../test-data/test-2C/bdf-ref.txt", std::ios::in);
    double error = 0.0;
    for (size_t i=0; i<bdf.size(); ++i) {
        auto row = split(read_line(fid));
        if (!fid) break;
        auto r = str2dbl(row[0]), g = str2dbl(row[1]);
        error += fabs(g - bdf.histogram[i]);
    }
    REQUIRE(error == Approx(0.0).epsilon(5e-5));
}
