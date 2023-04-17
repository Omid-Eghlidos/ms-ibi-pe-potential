#include "lammps_data.h"
#include "string_tools.h"
#include "histogram.h"
#include "input.h"
#include <fstream>
#include <iostream>
#include <map>
#include <cmath>

namespace cg {

    // Returns the scaled coordinate xs in [0,1) from unscaled coordinates x.
    inline Vec3 unscaled_to_scaled(Vec3 r, Matrix3d tri_box) {
        // Convert Vec3 to Vector3d
        Vector3d x; for (int i=0; i<3; i++) {x(i) = r(i);}
        // Find the PBC image coordinate
        Vector3d xs = tri_box.inverse() * x;
        // Convert Vector3d to Vec3
        Vec3 rs; for (int i=0; i<3; i++) {rs(i) = xs(i);}
        return rs;
    }

    // Returns the scaled coordinate xs in [0,1) from unscaled coordinates x.
    inline double unscaled_to_scaled(double x, double dx) {
        x /= dx;
        while (x <  0.0) x += 1.0;
        while (x >= 1.0) x -= 1.0;
        return x;
    }

    // Returns the unscaled coordinate x in simulation box from unwrapped coordinates xu.
    inline double unwrapped_to_unscaled(double x, double dx) {
        while (x <  0.0) x += dx;
        while (x >= dx) x -= dx;
        return x;
    }

    // Reads the data from LAMMPS.
    LAMMPS_Data::LAMMPS_Data(const InputSettings &in) {
        modeltype = in.compute_internal_dof;
        read_data(in.lmp_data_path);
        read_dump(in.lmp_dump_path, in.step_begin, in.step_end);

        std::map<int,int> count;
        for (auto &atom: _system.atoms) {
            ++count[atom.type];
        }
        for (auto &c: count) {
            std::cout << c.first << ": " << c.second << "\n";
        }
    }

    // Reads a LAMMPS input file.
    void LAMMPS_Data::read_data(std::string path) {
        std::fstream fid(path, std::ios_base::in);
        if (!fid) {
            std::cerr<< "Error opening output file " << path << "!\n";
            exit(1);
        }
        std::cout << "Opened data file " << path << "\n";
        int section = 0;
        size_t num_atoms_read = 0;
        read_line(fid); // Reads header line.
        while (fid) {
            auto line = split(read_line(fid));
            if (line.empty()) continue;
            else if (line[0]=="Masses")    section=1;
            else if (line[0]=="Pair")      section=2;
            else if (line[0]=="Bond")      section=3;
            else if (line[0]=="Angle")     section=4;
            else if (line[0]=="Dihedral")  section=5;
            else if (line[0]=="Atoms")     section=6;
            else if (line[0]=="Bonds")     section=7;
            else if (line[0]=="Angles")    section=8;
            else if (line[0]=="Dihedrals") section=9;

            switch (section) {
            // This reads the header and box information of the data file.
            case 0:
                if (line.size() == 2 && line[1] == "atoms") {
                    auto num_atoms = str2u32(line[0]);
                    _system.atoms.assign(num_atoms, {});
                }
                if (line.size() == 6 && line[3] == "xy") {
                    std::cout << "Triclinic box detected.\n";
                    triclinic_box = true;
                }
                break;
            case 1:
                if (line.size()==2) {
                    //_system.atom_types.push_back(str2u32(line[0]));
                    _system.masses.push_back(str2dbl(line[1]));
                }
                break;
            case 6:
                if (line.size() >= 6) {
                    auto i = str2u32(line[0]) - 1;
                    _system.atoms[i].molecule = str2u32(line[1]);
                    _system.atoms[i].type = str2u32(line[2]);
                    num_atoms_read += 1;
                }

                break;
            case 7:
                if (_system.connect.empty()) {
                    _system.connect.assign(_system.atoms.size(), {});
                }
                if (line.size()==4) {
                    auto i1 = str2u32(line[2])-1;
                    auto i2 = str2u32(line[3])-1;
                    _system.connect[i1].push_back(i2);
                    _system.connect[i2].push_back(i1);
                }
                break;
            case 8: break; // No need to keep going.
            default: continue;
            };
        }
        // Report total number of bonds.
        int num_bonds = 0;
        for (auto &b: _system.connect) {
            num_bonds += b.size();
        }
        if (_system.atoms.size() != num_atoms_read) {
            std::cout << "Error data file had inconsistent number of atoms.\n";
            exit(1);
        }
        std::cout << "Read " << num_atoms_read << " atoms from data file\n";
        std::cout << "Read " << num_bonds/2 << " bonds from data file.\n";
    }

    // Reads a LAMMPS dump output file, in atom format.
    void LAMMPS_Data::read_dump(std::string path, int step_begin, int step_end) {
        std::fstream fid(path, std::ios_base::in);
        if (!fid) {
            std::cout << "Error opening output file " << path << "!\n";
            exit(1);
        }
        std::cout << "Opened dump " << path << ".\n";

        SnapshotDump *step=NULL;
        while (fid) {
            auto line = trim(read_line(fid));
            if (line.empty()) continue;
            else if (line=="ITEM: TIMESTEP") {
                int timestep = from_string<int>(trim(read_line(fid)));
                // Skip lines until timestep is greater or equal to 
                // step_begin.  (if default value of step_begin = -1, then
                // continue with current timestep.
                while (fid && timestep < step_begin) {
                    line = trim(read_line(fid));
                    if (line == "ITEM: TIMESTEP") {
                        timestep = from_string<int>(trim(read_line(fid)));
                    }
                }
                // If file ends (no steps found) or if current step is beyond
                // step_end, then return.
                if (!fid || (step_end >= 0 && timestep > step_end)) {
                    return;
                }
                _dump.push_back(SnapshotDump());
                step = &_dump.back();
                step->timestep = timestep;
            }
            else if (line=="ITEM: NUMBER OF ATOMS") {
                if (!step) {
                    std::cerr << "Error: NUMBER OF ATOMS specified before TIMESTEP\n";
                    exit(1);
                }
                unsigned n = str2u32(trim(read_line(fid)));
                step->scaled_coordinates.assign(n, zeros<3>());
            }
            else if (line.find("ITEM: BOX BOUNDS")==0) {
                if (!step) {
                    std::cerr << "Error: BOX BOUNDS specified before TIMESTEP\n";
                    exit(1);
                }

                if (!triclinic_box) {
                    for (double *delta: {&step->dx, &step->dy, &step->dz}) {
                        auto bounds  = split(read_line(fid));
                        if (bounds.size() > 3) {
                            std::cerr << "Box bounds has wrong sizes.\n";
                            exit(1);
                        }
                        double lo = from_string<double>(bounds[0]);
                        double hi = from_string<double>(bounds[1]);
                        *delta = fabs(hi - lo);
                    }
                }
                else if (triclinic_box) {
                    // Following https://docs.lammps.org/Howto_triclinic.html
                    // Read xlo_bound xhi_bound xy
                    auto bounds  = split(read_line(fid));
                    double xlo_bound = from_string<double>(bounds[0]);
                    double xhi_bound = from_string<double>(bounds[1]);
                    double xy = from_string<double>(bounds[2]);
                    // Read ylo_bound yhi_bound xz
                    bounds  = split(read_line(fid));
                    double ylo_bound = from_string<double>(bounds[0]);
                    double yhi_bound = from_string<double>(bounds[1]);
                    double xz = from_string<double>(bounds[2]);
                    // Read zlo_bound zhi_bound yz
                    bounds  = split(read_line(fid));
                    double zlo_bound = from_string<double>(bounds[0]);
                    double zhi_bound = from_string<double>(bounds[1]);
                    double yz = from_string<double>(bounds[2]);

                    // Box parameters xlo, xhi, ylo, yhi, zlo, zhi
                    double xlo = xlo_bound - std::min(0.0, std::min(xy, std::min(xz, xy+xz)));
                    double xhi = xhi_bound - std::max(0.0, std::max(xy, std::max(xz, xy+xz)));
                    double lx = xhi - xlo;
                    double ylo = ylo_bound - std::min(0.0, yz);
                    double yhi = yhi_bound - std::max(0.0, yz);
                    double ly = yhi - ylo;
                    double zlo = zlo_bound;
                    double zhi = zhi_bound;
                    double lz = zhi - zlo;
                    // Step simulation box dimensions
                    step->dx = lx;
                    step->dy = ly;
                    step->dz = lz;
                    step->xy = xy;
                    step->xz = xz;
                    step->yz = yz;

                    // Simulation Box
                    Matrix3d L;
                    L <<  lx,  xy, xz,
                         0.0,  ly, yz,
                         0.0, 0.0, lz;
                    step->tri_box = L;
                    // Orthogonal box with its origin at xlo, ylo, zlo with
                    // lx, ly, and lz as its edges
                    Vector3d A, B, C;
                    A <<  lx, 0.0, 0.0;
                    B << 0.0,  ly, 0.0;
                    C << 0.0, 0.0,  lz;
                    // Orthogonal box volume
                    double V = (A.cross(B)).dot(C);
                    // Transformation matrix
                    Matrix3d H;
                    H.row(0) = B.cross(C);
                    H.row(1) = C.cross(A);
                    H.row(2) = A.cross(B);
                    Matrix3d T = L * H / V;
                    step->T = T.inverse();
                }
            }
            // The only thing left should be the ATOMS data.
            else {
                auto pieces = split(line);
                if (pieces.size() < 4) continue;
                if (pieces[0]=="ITEM:" && pieces[1]=="ATOMS") {
                    std::vector<std::string> var(pieces.begin()+2, pieces.end());
                    // Search for coordinate and tag columns.
                    step->xc  = step->yc  = step->zc  = step->tc = -1;
                    step->xs  = step->ys  = step->zs  = 0;
                    step->vxc = step->vyc = step->vzc = -1;
                    for (size_t i=0; i<var.size(); ++i) {
                        if (var[i]=="x")  {step->xc=i;}
                        if (var[i]=="xs") {step->xc=i; step->xs=1; }
                        if (var[i]=="xu") {step->xc=i; step->xu=1; }
                        if (var[i]=="y")  {step->yc=i;}
                        if (var[i]=="ys") {step->yc=i; step->ys=1; }
                        if (var[i]=="yu") {step->yc=i; step->yu=1; }
                        if (var[i]=="z")  {step->zc=i;}
                        if (var[i]=="zs") {step->zc=i; step->zs=1; }
                        if (var[i]=="zu") {step->zc=i; step->zu=1; }
                        if (var[i]=="id")  step->tc=i;
                        if (var[i]=="vx")  step->vxc=i;
                        if (var[i]=="vy")  step->vyc=i;
                        if (var[i]=="vz")  step->vzc=i;
                    }
                    if (step->xc<0 || step->yc<0 || step->zc<0) {
                        std::cout << "Error: coordinate column not found\n";
                        exit(1);
                    }
                    if (step->tc<0) {
                        std::cout << "Error: atom tag column not found.\n";
                        exit(1);
                    }

                    if (step->vxc >= 0 || step->vyc >= 0 || step->vzc >= 0) {
                        step->velocity.assign(num_atoms(), zeros<3>());
                    }

                   continue;
                }
                if (!step) {
                    std::cout << "Error: data encountered before TIMESTEP\n";
                    exit(1);
                }
                auto maxc=std::max(step->tc, std::max(step->xc, std::max(step->yc, step->zc)));
                auto minc=std::min(step->tc, std::min(step->xc, std::min(step->yc, step->zc)));

                if (maxc >= (int)pieces.size()) {
                    std::cout << "Not enough columns in dump data.\n";
                    exit(1);
                }
                if (minc < 0) {
                    std::cout << "Missing data column in dump data.\n";
                    exit(1);
                }

                unsigned id = str2u32(pieces[step->tc])-1;
                if (step->scaled_coordinates.size() <= id) {
                    std::cout << "Error: invalid atom id found " << id << "\n";
                    exit(1);
                }

                // Set positions (must exist).
                auto &r = step->scaled_coordinates[id];
                r(0) = str2dbl(pieces[step->xc]);
                r(1) = str2dbl(pieces[step->yc]);
                r(2) = str2dbl(pieces[step->zc]);

                // Find the PBC image coordinates in a triclinic box
                if (triclinic_box) {
                    // Store scaled coordinates from unscaled coordinates
                    if (!step->xs) r = unscaled_to_scaled(r, step->tri_box);
                }
                else {
                    // Find unscaled coordinates from unwrapped coordinates
                    if (step->xu) r(0) = unwrapped_to_unscaled(r(0), step->dx);
                    if (step->yu) r(1) = unwrapped_to_unscaled(r(1), step->dy);
                    if (step->zu) r(2) = unwrapped_to_unscaled(r(2), step->dz);

                    // Store scaled coordinates from unscaled coordinates
                    if (!step->xs) r(0) = unscaled_to_scaled(r(0), step->dx);
                    if (!step->ys) r(1) = unscaled_to_scaled(r(1), step->dy);
                    if (!step->zs) r(2) = unscaled_to_scaled(r(2), step->dz);
                }

                // Set velocities (if they exist).
                if (step->vxc >= 0 || step->vyc >= 0 || step->vzc >= 0) {
                    auto &v = step->velocity[id];
                    v(0) = step->vxc > 0 ? str2dbl(pieces[step->vxc]) : 0.0;
                    v(1) = step->vxc > 0 ? str2dbl(pieces[step->vyc]) : 0.0;
                    v(2) = step->vxc > 0 ? str2dbl(pieces[step->vzc]) : 0.0;
                }
            }
        }
        std::cout << "Read " << _dump.size() << " time steps.\n";
    }

    // Returns the maximum possible distance between atoms in a periodic box.
    double LAMMPS_Data::min_box_size(int step) const {
        const auto &d = _dump[step];
        return std::min(std::min(d.dx, d.dy), d.dz);
    }

    // Returns the volume at a time step for a cubic box.
    double LAMMPS_Data::volume(int step, bool triclinic_box) const {
        if (triclinic_box) {return _dump[step].tri_box.determinant();}
        return _dump[step].dx*_dump[step].dy*_dump[step].dz;
    }

    // Returns the position of an atom.
    Vector<3> LAMMPS_Data::atom_position(int step, int i, bool triclinic_box) const {
        auto r = _dump[step].scaled_coordinates[i];
        if (triclinic_box) {
            // Convert Vec3 to Vector3d
            Vector3d x; for (int i=0; i<3; i++) {x(i) = r(i);}
            // Find the scaled coordinates
            x = _dump[step].tri_box * x;
            // Convert Vector3d to Vec3
            for (int i=0; i<3; i++) {r(i) = x(i);}
        }
        else {
            r(0) *= _dump[step].dx;
            r(1) *= _dump[step].dy;
            r(2) *= _dump[step].dz;
        }
        return r;
    }
}

