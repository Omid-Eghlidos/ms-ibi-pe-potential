#pragma once
#include <map>
#include <string>
#include <vector>

namespace cg {
    //! Define beads as a sequence of atom identifiers (string)
    struct BeadType {
        //! Atoms types that make up the bead.
        std::vector<std::string> atoms;
        //! CG coordinates will be computed as weighted average of the atoms.
        std::vector<double> weights;
        //! Atoms that can be shared by two beads.
        std::vector<int> share;
        //! Mass scale for calculating the density field.
        double mscale = 0.0;
    };

    //! Settings for histogram bins.
    struct HistogramSettings {
        double lo=0.0, hi=1.0, delta=0.1;
        bool enabled = false;
    };

    //! Settings for histogram bins.
    struct DensityfieldSettings {
        double rcut = 0.0;
        bool enabled = false;
    };

    //! Structure of all input settings used by cg-post.
    struct InputSettings {
        //! Compute rdf with kernel density estimation
        bool rdf_kde = false;
        //! the width parameter used in kde rdf
        double kde_width = 1.5;
        //! Whether or not to use center of mass for bead position.
        bool use_bead_center_of_mass = false;
        //! Whether or not to compute internal degrees of freedom.
        bool compute_internal_dof = false;
        //! Range of time steps to compute distributions for (-1 means ignore).
        int step_begin=-1, step_end=-1;

        //! Path to LAMMPS dump file.
        std::string lmp_dump_path;
        //! Path to LAMMPS data file (for bond topology).
        std::string lmp_data_path;
        //! Path to a folder to write the output files.
        std::string output_path = "./";
        //! Tag for constructing output file names.
        std::string output_tag;
        //! Converts from integer type (LAMMPS) to element type.
        std::map<int, std::string> atom_types;
        //! Map of bead types.
        std::map<std::string, BeadType> bead_types;
        //! Settings for each histogram.
        //! Settings for densiy field computation
        DensityfieldSettings rhobar;
        //! Settings for bond angle caluculations
        bool use_bond_angle = false;
        std::string output_bond_angle;
        HistogramSettings adf, rdf, bdf, tdf, idf;
        //! Which bonded pairs to exclude (same as LAMMPS special_bonds).
        int special_bonds[3] = {0, 1, 1};
        //! Settings for dihedral and improper correlations
        bool dihedral_correlations = false;
        bool improper_correlations = false;

        //! Returns a reference to a bead type if it exists.
        const BeadType& get_bead_type(std::string beadtype) const;
    };

    //! Reads cgpost input file and returns structure of all settings.
    InputSettings read_inputfile(std::string inputfile);
}
