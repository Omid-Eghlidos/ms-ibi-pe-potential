#pragma once
#include <vector>
#include <string>
#include <map>
#include <memory>
#include "histogram.h"
#include "vector.h"
#include <Eigen/Dense>

using Eigen::Matrix3d;

namespace cg {

class InputSettings;
class LAMMPS_Data;

struct Bead;
typedef std::vector<Bead> VBead;
typedef VBead::const_iterator BeadCIter;
typedef std::tuple<std::string, std::string> PairTypes;
typedef std::tuple<std::string, std::string, std::string> AngleTypes;
typedef std::tuple<std::string, std::string, std::string, std::string> TorsionTypes;
typedef std::tuple<std::string, std::string, std::string, std::string> ImproperTypes;

/*! Types of ways that atoms can be mapped to a bead.
 * UNMAPPED: atom is not assigned to any beads.
 * MAPPED: atom is assigned by bead definition.
 * SHARED: atom is mapped to two beads.
 * SIDE:   atom is bonded to a mapped atom.
 * SIDE_SHARED: atom is bonded to a shared atom. 
 */
enum class AtomMapping { UNMAPPED, MAPPED, SHARED, SIDE, SIDE_SHARED };

//! Object that defines a coarse grain bead.
struct Bead {
    //! Index of bead in bead list.
    int index;
    //! Type of this bead.
    std::string type;                 
    //! End atoms of bead.
    std::vector<int> ends;
    //! Neighbors to this bead (probably 2).    
    std::vector<int> neighbors;
    //! kinetic energy of the bead
    std::vector<double> ext_ke;
    //! internal kinetic energy of the bead
    std::vector<double> int_ke;
    //! Vector to store the atomid(index+1) which form bead
    std::vector<int> atoms;
};

class CgSystem {
friend class Distribution;
public:
    //! Initializes the cg_system.
    CgSystem(const InputSettings &in, const LAMMPS_Data &lmp);
    //! Returns the element of an atom type.
	std::string type_to_element(int type) const;
    //! Finds unique cg beads of a type in lammps system, returns # found.
    int find_beads(std::string beadtype);    
    //! Finds the neighbors for each bead.
    void find_bead_neighbors();
    //! Calculate the internal dof of beads
    void compute_intdof();
    //! Adds the atoms which are not in main branch of beads
    void add_unmapped_atoms();
    //! Compute Ext ke of beads
    void compute_bead_kinetic_energy();
    //! compute the bead temperature at each timestep
    void compute_bead_temperature();

    //! Computes coordinate of bead using defined CG map.
    Vec3 bead_coordinate(int step, int i) const;
    //! Computes nearest image distance between beads i and j in a cubic box.
    Vec3 distance_vector(int step, int i, int j, bool triclinic_box) const;
    //! Determines if beads, i, j are 1st neighbors.
    bool neighbors(const Bead &i, const Bead &j) const;
    //! Determines if beads, i, j are 2nd neighbors.
    bool second_neighbors(const Bead &i, const Bead &j) const;
    //! Returns the output tag.
    std::string output_tag() const { return _output_tag; }
    //! Returns the number of beads by type.
    int bead_count(std::string bead_type) const;
    //! Returns a vector of all bead types defined.
    std::vector<std::string> defined_bead_types() const;
    // Computes all unique combinations of pair types.
    std::vector<PairTypes> unique_pair_types() const;
    // Computes all unique combinations of angle types.
    std::vector<AngleTypes> unique_angle_types() const;
    // Computes all unique combinations of torsion types.
    std::vector<TorsionTypes> unique_torsion_types() const;
    // Computes all unique combinations of improper types.
    std::vector<ImproperTypes> unique_improper_types() const;
    //! Writes a representation of each chain (won't work for branches).
    void print_cg_chains() const;

    //! Reference to input settings.
    const InputSettings &input_settings;
    //! Reference to LAMMPS data.
    const LAMMPS_Data& lmp_data;
    //! Beads defined in this system.
    std::vector<Bead> beads;
    
private:
    //! Atoms mapped to beads
    std::vector<AtomMapping> _atom_map_status;
    //! Tag to add to the output files.
    std::string _output_tag;
    //! store the total ext ke of the system at each timestep
    std::vector<double> ext_ke;
    //! store the total int ke of the system at each timestep
    std::vector<double> int_ke;
};
    //! Checks if type1 matches type2 (allows wildcards on type2).
    bool match(std::string type1, std::string type2);
    //! Returns vector of all possible pair distances given a distance vector.
    std::vector<double> pair_distances(const Vec3 &dr, const Vec3 &box
                                     , double rcut, bool triclinic_box);
    // Returns the nearest PBC image coordinate of an atom in a triclinic box
    Vec3 triclinic_pbc_image(Vec3 r, Matrix3d tri_box);
}
