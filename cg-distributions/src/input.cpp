#include "input.h"
#include <fstream>
#include <iostream>
#include "string_tools.h"

namespace cg {

    //! Returns a reference to a bead type if it exists.
    const BeadType& InputSettings::get_bead_type(std::string beadtype) const {
        auto iter = bead_types.find(beadtype);
        if (iter == bead_types.end()) {
            std::cerr << "Bead type " << beadtype << " not defined.\n";
            exit(1);
        }
        return iter->second;
    }

    InputSettings read_inputfile(std::string inputfile) {
        InputSettings inp;

        std::fstream fid(inputfile, std::ios::in);
        if (!fid) {
            std::cerr << "Error input " << inputfile << " cannot be opened.\n";
            exit(1);
        }
        while (fid) {
            auto line = read_line(fid);
            if (line.empty()) continue;
            line = line.substr(0, line.find_first_of("#"));
            auto cmd = split(line);
            size_t n = cmd.size();

            if      (cmd.empty()) continue;
            else if (cmd[0]=="input" && n==2) {
                inp.lmp_data_path = cmd[1];
            }
            else if (cmd[0]=="dump" && n==2) {
                inp.lmp_dump_path = cmd[1];
            }
            else if (cmd[0]=="type" && n==3) {
                inp.atom_types[str2u32(cmd[1])] = cmd[2];
            }
            else if (cmd[0]=="output_path" && n==2) {
                inp.output_path = cmd[1];
            }
            else if (cmd[0] =="output_tag" && n==2) {
                inp.output_tag = cmd[1];
            }
            else if (cmd[0]=="rdf" || cmd[0]=="bdf" || 
                     cmd[0]=="adf"||cmd[0]=="tdf"||cmd[0]=="idf") {
                HistogramSettings *h;
                if (cmd[0] == "adf") h = &inp.adf;
                if (cmd[0] == "bdf") h = &inp.bdf;
                if (cmd[0] == "rdf") h = &inp.rdf;
                if (cmd[0] == "tdf") h = &inp.tdf;
                if (cmd[0] == "idf") h = &inp.idf;
                auto h_settings = args_to_tuple<double,double,double>(cmd,1);
                std::tie(h->lo, h->hi, h->delta) = h_settings;
                h->enabled = true;

            }
            else if (cmd[0]=="density_field") {
                DensityfieldSettings *h;
                h = &inp.rhobar;
                h->rcut = from_string<double>(cmd[1]);
                h->enabled = true;

            }
            else if (cmd[0]=="bond_angle") {
               inp.use_bond_angle = true;
            }
            else if (cmd[0]=="dihedral_correlations") {
               inp.dihedral_correlations = true;
            }
            else if (cmd[0]=="improper_correlations") {
               inp.improper_correlations = true;
            }
            else if (cmd[0] == "rdftype" && cmd[1] == "kernel" && n==3) {
                inp.rdf_kde = true;
                inp.kde_width = from_string<double>(cmd[2]);
            }
            else if (cmd[0]=="bead" && n>2) {
                auto &bead_id = cmd[1];
                BeadType new_bead_type;
                for (auto e=cmd.cbegin()+2; e!=cmd.cend(); ++e) {
                    new_bead_type.atoms.push_back(*e);
                }
                inp.bead_types[bead_id] = new_bead_type;
            }
            else if (cmd[0]=="share" && n>2) {
                auto iter = inp.bead_types.find(cmd[1]);
                if (iter == inp.bead_types.end()) {
                    std::cerr << "Error: bead " << cmd[1] << " not found.\n";
                    exit(1);
                }
                // Push back any atoms indices after the id to indicate sharing.
                for (auto i=cmd.cbegin()+2; i!=cmd.cend(); ++i)
                    iter->second.share.push_back(str2u32(*i));
            }
            // special_bonds [w1] [w2] [w3]
            // w1, w2, w3 set bonded, angle, and dihedral flags for inclusion
            // in RDF calculations.  Value of wt may be either 0 or 1.
            // Default: {0 1 1}
            else if (cmd[0]=="special_bonds" && n==4) {
                inp.special_bonds[0] = from_string<int>(cmd[1]);
                inp.special_bonds[1] = from_string<int>(cmd[2]);
                inp.special_bonds[2] = from_string<int>(cmd[3]);
            }
            // timestep_range [step_begin] [step_end]
            else if (cmd[0] == "timestep_range" && n == 3) { 
                inp.step_begin = from_string<int>(cmd[1]);
                inp.step_end = from_string<int>(cmd[2]);
            }
            else if (cmd[0]=="weights" && n > 2) {
            // weights [beadtype] [w1] [w2] [w3] ...
                auto iter = inp.bead_types.find(cmd[1]);
                if (iter == inp.bead_types.end()) {
                    std::cerr << "Error: bead " << cmd[1] << " not found.\n";
                    exit(1);
                }
                for (auto arg=cmd.begin()+2; arg!=cmd.end(); ++arg) {
                    iter->second.weights.push_back(str2dbl(*arg));
                }
                // Normalize weights so that their sum is unity.
                auto total = 0.0;
                for (auto w: iter->second.weights) total += w;
                for (auto &w: iter->second.weights) w /= total;
            }
            else if (cmd[0]=="massscale") {
            // massscale [beadtype] [x]
                auto iter = inp.bead_types.find(cmd[1]);
                if (iter == inp.bead_types.end()) {
                    std::cerr << "Error: bead " << cmd[1] << " not found.\n";
                    exit(1);
                }
                iter->second.mscale = str2dbl(cmd[2]);
            }
            else if (cmd[0]=="use_center_of_mass" && n==1) {
                inp.use_bead_center_of_mass = true;
            }
            else if (cmd[0]=="compute_intdof") {
                inp.compute_internal_dof = true;   
            }
            else std::cerr << "Invalid command \"" << join(cmd) << "\".\n";
        }
        return inp;
    }
}

