#pragma once
#include <string>
#include <vector>
#include "vector.h"
#include <Eigen/Dense>


using Eigen::Matrix3d;
using Eigen::Vector3d;


namespace cg {

//! Output from LAMMPS dump style atom.
struct SnapshotDump {
    //! Actual count of timestep.
    int timestep; 
    //! Box sizes along each dimension.
    double dx, dy, dz;  
    //! Indices where the required data is located.
    int xc, yc, zc, tc;  // DELETE
    int vxc, vyc, vzc;   // DELETE
    //! Whether each dof is scaled or not.
    bool xs, ys, zs;
    std::vector<Vec3> scaled_coordinates;
    //! Whether each dof is unwrapped or not.
    bool xu, yu, zu;
    std::vector<Vec3> unwrapped_coordinates;
    //! Triclinic box tilts
    double xy, xz, yz;
    // Triclinic box
    Matrix3d tri_box;
    // Transformation matrix from triclinic to orthogonal
    Matrix3d T;
    //vector to store the velocities of atoms
    std::vector<Vec3> velocity;
};

//! Basic storage class for atoms.
struct Atom {
    int type;           // Atom type - one-based type index.
    int molecule;       // Molecule atom is part of - one based.
};

//! Stores data from the LAMMPS data file.
struct AtomicSystem {    
    //! Mass of each atom type.
    std::vector<double> masses;
    //! Stores each atom's type and molecule type.
    std::vector<Atom> atoms;
    //! Stores list of atoms bonded to each atom.
    std::vector<std::vector<int>> connect;  
};

class InputSettings;
class LAMMPS_Data {
    friend class CgSystem;
public:
    //! Builds lammps data structure; optional: read input and dump.
    LAMMPS_Data(const InputSettings &i);
    //! Reads a LAMMPS input file.
    void read_data(std::string path);
    //! Reads a LAMMPS dump output file.
    void read_dump(std::string path, int step_begin, int step_end);
    //! Returns the maximum possible distance between atoms in a periodic box.
    double min_box_size(int step) const;
    //! Returns the position of an atom.
    Vec3 atom_position(int step, int i, bool triclinic_box=false) const;
    //! Returns how many timesteps there are.
    int num_timesteps() const { return _dump.size(); }
    //! Returns how many atoms there are in the system.
    int num_atoms() const { return _system.atoms.size(); }
    //! Simulation box dimensions
    double box_dx(int step) const { return _dump[step].dx; }
    double box_dy(int step) const { return _dump[step].dy; }
    double box_dz(int step) const { return _dump[step].dz; }
    //! Tilts of triclinic box
    double box_xy(int step) const { return _dump[step].xy; }
    double box_xz(int step) const { return _dump[step].xz; }
    double box_yz(int step) const { return _dump[step].yz; }
    //! Returns the orthogonal box lengths in a vector.
    Vec3 box_size(int step) const {
        return {_dump[step].dx, _dump[step].dy, _dump[step].dz}; }

    //! Returns the triclinic box as a matrix
    Matrix3d tri_box(int step) const {
        return _dump[step].tri_box;
    }
    //! Returns the volume at a time step.
    double volume(int step, bool triclinic_box) const;
    //! Simulation box type (orthogonal / triclinic)
    bool triclinic_box = false;
private:
    AtomicSystem _system;    
    std::vector<SnapshotDump> _dump;
    //! type of computing model(0->only rdf and 1-> rdf+internaldof)
    int modeltype;
};

// Nearest PBC image coordinate of an atom in a triclinic box
Vec3 triclinic_pbc_image(Vec3 r, Matrix3d tri_box);
}

