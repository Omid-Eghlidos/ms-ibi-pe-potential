#pragma once
#include "histogram.h"
#include "cgsystem.h"
#include <string>
#include <tuple>

namespace cg {

    class InputSettings;
    class LAMMPS_Data;
    using BondAngle = std::tuple<Histogram, std::vector<std::vector<double>>>;

    //! This class computes bond, angle and radial density distributions.
    class Distribution {
    public:

        void compute_pair_correlations() const;
        void compute_bond_correlations() const;
        void compute_angle_correlations() const;
        void compute_density_field() const;
        void compute_torsion_correlations() const;
        void compute_improper_correlations() const;

        //! Computes pair, bond, and angle distributions for all time steps.
        void compute_correlations() const;
        //! Sets references to LAMMPS data and CG mapping.
        Distribution(const CgSystem &sys, const InputSettings &set);
        //! Computes the radial distribution function between beads.
        Histogram compute_rdf(const PairTypes &t, int step) const;
        //! Computes the radial distribution function between beads with kde.
        Histogram compute_rdf_kde(const PairTypes &t, int step) const;
        //! Computes the bond-length distribution function between beads.
        Histogram compute_bdf(const PairTypes &t, int step) const;
        //! Computes the bond-angle distribution function between beads.
        BondAngle compute_adf(const AngleTypes &t, int step) const;
        //! Computes the density field of bead at step.
        double compute_density(const std::string &t, int step) const;
        //! Computes the bond-torsion distribution function.
        BondAngle compute_tdf(const TorsionTypes &t, int step) const;
        //! Computes the bond-torsion distribution function.
        BondAngle compute_idf(const ImproperTypes &t, int step) const;

        const CgSystem &system;
        const InputSettings &input_settings;
        const LAMMPS_Data &lmp_data;
    };
}

