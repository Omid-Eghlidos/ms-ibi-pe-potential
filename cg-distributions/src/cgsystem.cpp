#include "cgsystem.h"
#include "lammps_data.h"
#include "input.h"
#include "string_tools.h"
#include <numeric>
#include <set>

namespace cg { 

    // Constructor for the cg_system.
    CgSystem::CgSystem(const InputSettings &in, const LAMMPS_Data &lmp)
      : input_settings(in), lmp_data(lmp) {
        _atom_map_status.assign(lmp_data.num_atoms(), AtomMapping::UNMAPPED);
        for (auto &&b: in.bead_types) {
            int ct = find_beads(b.first);
            std::cout << "Bead " << b.second.atoms << '\t' << ct << " found.\n";
        }
        find_bead_neighbors();
        add_unmapped_atoms();
        if (input_settings.compute_internal_dof) {
            compute_intdof();
        }
    }

    // Returns the element of an atom type.
    std::string CgSystem::type_to_element(int type) const { 
        auto iter = input_settings.atom_types.find(type);
        if (iter == input_settings.atom_types.end()) {
            std::cerr << "Atom type " << type << " not found.\n";
            exit(1);
        }
        return iter->second; 
    }
    
    // Crawls though the connectivity and determines beads, returns # found.
    int CgSystem::find_beads(std::string beadtype) {
        std::cout << "Finding beads of type " << beadtype << "\n";
        auto &btype = input_settings.get_bead_type(beadtype);
        auto &share = btype.share;
        
        const AtomicSystem &sys = lmp_data._system;
        if (btype.atoms.empty()) {
            std::cerr << "CG bead definition has no atoms.\n";
            exit(1);
        }

        int num_found = 0;
        auto start_type = btype.atoms.front();
        // Loop over atoms try to find first bead atom.
        for (size_t i = 0; i < sys.atoms.size(); ++i) {
            // Skip if atom is already in a bead and not shared.
            bool shared = std::count(share.begin(), share.end(), 0) > 0;
            //if (_atom_map_status[i] == AtomMapping::MAPPED) continue;
            if (!shared && _atom_map_status[i] != AtomMapping::UNMAPPED) continue; 
            if (!match(type_to_element(sys.atoms[i].type), start_type)) continue;

            std::vector<std::vector<int>> potential_beads = {{int(i)}};
            for (size_t t = 1; t < btype.atoms.size(); ++t) {
                // Temporary vector to store potential beads
                std::vector<std::vector<int>> temp_beads;
                // Loop over all possible combinations of atoms
                for (auto potential_bead : potential_beads) {
                    int current = potential_bead.back();
                    std::string next_type = btype.atoms[t];
                    shared = std::count(share.begin(),share.end(),potential_bead.size()) > 0;
                    
                    // Try to find next bead.
                    std::vector<int> potential_atoms;
                    auto next = sys.connect[current].begin();
                    for (; next != sys.connect[current].end(); ++next) {
                        //if (_atom_map_status[*next] == AtomMapping::MAPPED) continue;
                        if (!shared && _atom_map_status[*next] != AtomMapping::UNMAPPED) continue;
                        // Skip if focus atom already in potential bead
                        if (std::count(potential_bead.begin(), potential_bead.end(), *next)) continue;
                        if (match(type_to_element(sys.atoms[*next].type), next_type)) {
                            potential_atoms.push_back(*next);
                        }
                    }
                    
                    for (int atom : potential_atoms) {
                        potential_bead.push_back(atom);
                        temp_beads.push_back(potential_bead);
                        potential_bead.pop_back();
                    }
                }
                potential_beads = temp_beads;
            }
            // New bead is completed.
            for (auto potential_bead: potential_beads) {
                if (potential_bead.size() == btype.atoms.size()) {
                    Bead new_bead;
                    new_bead.type = beadtype;
                    new_bead.ends = {int(i)};
                    // Don't repeat end atom for beads that only have one atom.
                    if (potential_bead.size() > 1) {
                        new_bead.ends.push_back(potential_bead.back());
                    }
                    for (auto id: potential_bead) {
                        _atom_map_status[id] = AtomMapping::MAPPED;
                        new_bead.atoms.push_back(id);
                    }
                    new_bead.index = beads.size();
                    beads.push_back(new_bead);
                    num_found += 1;
                    break;
                }
            }
        }
        return num_found;
    }

    // Finds the neighbors for each bead.
    void CgSystem::find_bead_neighbors() {

        // For each atom, build a table of which bead(s) it is mapped to.
        std::vector<std::vector<int>> lookup_table(lmp_data.num_atoms());
        for (const auto &b: beads) {
            for (auto i: b.atoms) {
                lookup_table[i].push_back(b.index);
            }
        }
        for (auto &row: lookup_table) {
            std::sort(row.begin(), row.end());
        }
        // Add bond to any pair of beads that either share an 
        // atom or contain bonded atoms.
        for (int i=0; i<lmp_data.num_atoms(); ++i) {
            if (lookup_table[i].empty()) {
                continue;
            }
            if (lookup_table[i].size() > 1) {
                for (int bi: lookup_table[i]) {
                    for (int bj: lookup_table[i]) {
                        if (bi != bj) {
                            beads[bi].neighbors.push_back(bj);
                        }
                    }
                }
            }
            int b1 = lookup_table[i].front();
            for (auto j: lmp_data._system.connect[i]) {
                if (i < j && !lookup_table[j].empty()) {
                    int b2 = lookup_table[j].back();
                    if (b1 != b2) {
                        beads[b1].neighbors.push_back(b2);
                        beads[b2].neighbors.push_back(b1);
                    }
                }
            }
        }
        // Lastly, look through neighbors to eliminate any double counting.
        int ct = 0;
        for (auto &b: beads) {
            std::sort(b.neighbors.begin(), b.neighbors.end());
            auto end = std::unique(b.neighbors.begin(), b.neighbors.end());
            b.neighbors.assign(b.neighbors.begin(), end);
            ct += b.neighbors.size();
        }
        std::cout << "Found " << ct/2 << " number of bead neighbors.\n";
    }

    // Add the atoms to the bead which are not in main branch
    void CgSystem::add_unmapped_atoms()  {
        const AtomicSystem &sys = lmp_data._system;
        for (auto &bead: beads) {
            std::vector<int> shared_atom;
            for (auto i: input_settings.get_bead_type(bead.type).share) {
                shared_atom.push_back(bead.atoms[i]);
            }
            for (auto i: bead.atoms) {
                bool shared = std::count(shared_atom.begin(), shared_atom.end(), i) > 0;
                // Loops over all atoms bonded to atom, i.
                for (int j: sys.connect[i]) {
                    if (_atom_map_status[i] == AtomMapping::UNMAPPED) {
                        bead.atoms.push_back(j);
                        if (shared) {
                            _atom_map_status[j] = AtomMapping::SIDE_SHARED;
                        }
                        else {
                            _atom_map_status[j] = AtomMapping::SIDE;
                        }
                    }
                    else if (shared && _atom_map_status[i] == AtomMapping::SIDE_SHARED) {
                        bead.atoms.push_back(j);
                        _atom_map_status[j] = AtomMapping::SIDE_SHARED;
                    }
                }
            }
        }
    }

    //calculate the internal dof of the beads
    void CgSystem::compute_intdof()  {
        compute_bead_kinetic_energy();
        compute_bead_temperature();
    }

    // compute the int ke of the beads
    void CgSystem::compute_bead_kinetic_energy()  {
        const auto &dump        = lmp_data._dump;
        const AtomicSystem &sys = lmp_data._system;

        for (auto &i: beads) {
            i.int_ke.assign(lmp_data.num_timesteps(), 0.0);
            i.ext_ke.assign(lmp_data.num_timesteps(), 0.0);
        }

        for (int t=0; t<lmp_data.num_timesteps(); ++t){
            for (auto &i: beads) {
                double bead_total_ke=0.0;
                double beadmass=0.0;

                // P is the cumulative momentum vector.
                auto p = zeros<3>();
                for (auto id : i.atoms){
                    int atype = sys.atoms[id].type;
                    double m  =  sys.masses[atype-1];
                    if(_atom_map_status[id] == AtomMapping::SHARED || 
                       _atom_map_status[id] == AtomMapping::SIDE_SHARED) {
                        m = 0.5*m;
                    }
                    beadmass += m;
                    auto v = dump[t].velocity[id];
                    p.scaled_add(v, m);
                    bead_total_ke += 0.5*m*dot(v,v);
                }
                i.ext_ke[t] = 0.5*dot(p,p) / beadmass;
                i.int_ke[t] = bead_total_ke - i.ext_ke[t];
            }
        }
    }

    // compute the bead temperature
    void CgSystem::compute_bead_temperature()  {
        const double kb = 0.0019872041; //in kcal/mol/k
        const double c1 = 2390.05736; // convert ke from lammpsunit to kcal/mol
        int n = lmp_data.num_timesteps();
        std::vector<double> step_temperature(n,0.0);
        int_ke.assign(n,0.0);
        ext_ke.assign(n,0.0);
        for(int t=0; t<n; ++t){
            for (auto &i: beads) {
                ext_ke[t]+=i.ext_ke[t];
                int_ke[t]+=i.int_ke[t];
            }
            step_temperature[t]= 2.0/(3.0*beads.size()-3.0)*ext_ke[t] / kb;
            step_temperature[t] *= c1;
        }
        auto final_T = std::accumulate(step_temperature.cbegin(),
                                       step_temperature.cend(), 0.0) / n;
        std::cout << "Average system temperature is : " << final_T << '\n';
    }

    // Returns the nearest PBC image coordinate of an atom in a triclinic box
    Vec3 triclinic_pbc_image(Vec3 r, Matrix3d tri_box) {
        // Convert Vec3 to Vector3d
        Vector3d x; for (int i=0; i<3; i++) {x(i) = r(i);}
        // Find the PBC image coordinate
        Vector3d ds = (tri_box.inverse() * x).array().round().matrix();
        x -= tri_box * ds;
        // Convert Vector3d to Vec3
        Vec3 X; for (int i=0; i<3; i++) {X(i) = x(i);}
        return X;
    }

    // Computes coordinate of bead using weighted average of atoms.
    Vec3 CgSystem::bead_coordinate(int step, int i) const {
        auto &bead = beads[i];
        auto &bead_type = input_settings.get_bead_type(bead.type);
        Vec3 X = zeros<3>();
        if (lmp_data.triclinic_box) {
            auto x0 = lmp_data.atom_position(step, bead.atoms[0], true);
            auto box = lmp_data.tri_box(step);
            for (size_t i=0; i<bead_type.weights.size(); ++i) {
                auto wt = bead_type.weights[i];
                auto x = lmp_data.atom_position(step, bead.atoms[i], true);
                auto dx = triclinic_pbc_image(x - x0, box);
                X += (x0 + dx) * wt;
            }
        }
        else {
            auto x0 = lmp_data.atom_position(step, bead.atoms[0]);
            auto box = lmp_data.box_size(step);
            for (size_t i=0; i<bead_type.weights.size(); ++i) {
                auto wt = bead_type.weights[i];
                auto x = lmp_data.atom_position(step, bead.atoms[i]);
                for (int j: {0,1,2}) {
                    // Maps x to the periodic cell w/ the least distance to x0.
                    while (x(j) - x0(j) > 0.5*box(j)) {
                        x(j) -= box(j);
                    }
                    while (x(j) - x0(j) < -0.5*box(j)) {
                        x(j) += box(j);
                    }
                }
                X += wt * x;
            }
            // Shifts bead so that it is in the periodic cell.
            for (int i: {0,1,2}) {
                if (X(i) <  0.0) {
                    X(i) += box(i);
                }
                if (X(i) >= box(i)) {
                    X(i) -= box(i);
                }
            }

        }
        return X;
    }

    // Computes a vector between beads i and j inside a unit cell
    Vec3 CgSystem::distance_vector(int step, int i, int j, bool triclinic_box) const {
        auto dr = bead_coordinate(step, i) - bead_coordinate(step, j);
        Vec3 dx;
        if (!triclinic_box) {
            dx = dr;
            auto box = lmp_data.box_size(step);
            for (int i: {0,1,2}) {
                if (dx(i) > 0.5*box(i))  {
                    dx(i) -= box(i);
                }
                if (dx(i) < -0.5*box(i)) {
                    dx(i) += box(i);
                }
            }
        }
        else {
            dx = triclinic_pbc_image(dr, lmp_data.tri_box(step));
        }
        return dx;
    }

    // Determines if beads, i, j are 1st neighbors.
    bool CgSystem::neighbors(const Bead &i, const Bead &j) const {
        return std::count(i.neighbors.cbegin(), i.neighbors.cend(), j.index);
    }

    // Determines if beads, i, j share a common neighbor.
    bool CgSystem::second_neighbors(const Bead &i, const Bead &j) const {
        for (auto k=i.neighbors.cbegin(); k!=i.neighbors.cend(); ++k) {
            if (std::count(j.neighbors.cbegin(), j.neighbors.cend(), *k)) return true;
        }
        return false;
    }

    // Returns the number of beads of a given type.
    int CgSystem::bead_count(std::string bead_type) const {
        int count = 0;
        for (auto &b: beads) {
            count += int(b.type == bead_type);
        }        
        return count;
    }

    // Returns a vector of all bead types defined.
    std::vector<std::string> CgSystem::defined_bead_types() const {
        std::vector<std::string> types;
        for (auto &it: input_settings.bead_types) types.push_back(it.first);
        return types;
    }

    // Checks if type1 matches type2 (allows wildcards on type2).
    bool match(std::string type1, std::string type2) {
        if (type1.find('*') != type1.npos) {
            std::cout << "Warning: Wildcard found in type.\n";
        }        
        
        if (type2.find('*') == type2.npos) return type1 == type2;
            
        // If there is a wildcard, then we need to match everything before the wildcard.
        auto pieces = split(type2, "*");
        // Type2 was only stars.
        if (pieces.empty()) return true; 

        size_t loc=type1.find(pieces[0]);
        if (loc > 0) return false;
        for (auto p=pieces.cbegin()+1; p!=pieces.cend(); ++p) {
            loc = type1.find(*p, loc+1);
            if (loc == type1.npos) return false;
        }
        return true;
    }

    // Returns a vector of all possible pair distances between x and y in cubic box.
    std::vector<double> pair_distances(const Vec3 &dr, const Vec3 &box
                                      , double rcut, bool triclinic_box) {
        std::vector<double> r;
        // Triclinic box
        if (triclinic_box) {
            r.push_back(norm(dr));
            return r;
        }
        // Cubic box
        auto abs_dr = abs(dr);
        auto square = [](double x) { return x*x; };
        auto dx2min = square(std::min(box(0)-abs_dr(0), abs_dr(0)));
        auto dy2min = square(std::min(box(1)-abs_dr(1), abs_dr(1)));
        auto dz2min = square(std::min(box(2)-abs_dr(2), abs_dr(2)));

        // If the box is larger than the cutoff distance, then just compute the
        // nearest periodic distance.
        if (min(box) > 2.0*rcut) {
            return {sqrt(dx2min + dy2min + dz2min)};
        }

        r.reserve(27);
        auto c2 = square(rcut);
        float pp[] = {-1.0, 0.0, 1.0};
        for (auto px: pp) {
            auto dx2 = square(dr(0) + px*box(0));
            if (dx2 + dy2min + dz2min > c2) continue;
            for (auto py: pp) {
                auto dy2 = square(dr(1) + py*box(1));
                if (dx2 + dy2 + dz2min > c2) continue;
                for (auto pz: pp) {
                    auto r2 = square(dr(2) + pz*box(2)) + dx2 + dy2;

                    if (r2 < c2) {
                        r.push_back(sqrt(r2));
                    }
                }
            }
        }
        return r;
    }

    // Computes all combinations of pair types considering symmetry (AB = BA).
    std::vector<PairTypes> CgSystem::unique_pair_types() const {
        auto bead_types = defined_bead_types();
        // Loops over all possible combinations of pair types and returns the unique
        // pair considering symmetry.
        std::set<PairTypes> pairs;
        for (auto b1: bead_types) {            
            for (auto b2: bead_types) {
                if (b1 > b2) pairs.insert(std::tie(b2, b1));
                else pairs.insert(std::tie(b1, b2));
            }
        }
        return std::vector<PairTypes>(pairs.begin(), pairs.end());
    }

    // Computes all combinations of angle types considering symmetry (AAB = BAA).
    std::vector<AngleTypes> CgSystem::unique_angle_types() const {
        auto bead_types = defined_bead_types();
        // Loops over all possible combinations of angle types and returns the unique
        // triplets considering symmetry.
        std::set<AngleTypes> angles;
        for (auto b1: bead_types) {
            for (auto b2: bead_types) {
                for (auto b3: bead_types) {
                    if (b1 > b3) angles.insert(std::tie(b3, b2, b1));
                    else angles.insert(std::tie(b1, b2, b3));
                }
            }
        }
        return std::vector<AngleTypes>(angles.begin(), angles.end());
    }

    // Computes all combinations of torsion angle types.
    std::vector<TorsionTypes> CgSystem::unique_torsion_types() const {
        auto bead_types = defined_bead_types();
        std::set<TorsionTypes> torsions;
        for (auto t1: bead_types) {
            for (auto t2: bead_types) {
                for (auto t3: bead_types) {
                    for (auto t4: bead_types) {
                        if (t1 < t4 || (t1 == t4 && t2 <= t3)) {
                            torsions.insert(std::tie(t1, t2, t3, t4));
                        }
                    }
                }
            }
        }
        return std::vector<TorsionTypes>(torsions.begin(), torsions.end());
    }

    // Computes all combinations of improper angle types.
    std::vector<ImproperTypes> CgSystem::unique_improper_types() const {
        auto bead_types = defined_bead_types();
        std::set<ImproperTypes> impropers;
        for (auto t1: bead_types) {
            for (auto t2: bead_types) {
                for (auto t3: bead_types) {
                    for (auto t4: bead_types) {
                        if (t1 <= t4 || (t1 == t2 && t2 == t3 && t3 < t4)) {
                            impropers.insert(std::tie(t1, t2, t3, t4));
                        }
                    }
                }
            }
        }
        return std::vector<ImproperTypes>(impropers.begin(), impropers.end());
    }

    // Writes a representation of each chain (won't work for branches).
    void CgSystem::print_cg_chains() const {
        for (auto &b: beads) {
            if (b.neighbors.size() == 1 && b.index < b.neighbors[0]) {
                std::cout << b.type;
                auto prev = beads.begin() + b.index;
                auto next = beads.begin() + b.neighbors[0];
                while (true) {
                    std::cout << next->type;
                    if (next->neighbors.size() == 1) {
                        break;
                    }
                    for (int i: next->neighbors) {
                        if (i != prev->index) {
                            prev = next;
                            next = beads.begin() + i;
                            break;
                        }
                    }
                }
                std::cout << "\n";     
            }
        }
    }
}
