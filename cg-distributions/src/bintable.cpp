#include "bintable.h"
#include "lammps_data.h"
#include "cgsystem.h"
#include <cmath>
#include <set>

namespace cg {

    BinTable::BinTable(double cutoff, int step, const CgSystem *cg) { 
        // Keep pointer reference to cgsystem so we can look up bins later.
        _cgsys = cg;
        _cutoff = cutoff;
        _step = step;
        // Number of bins along each direction.
        // Granulity to make bins smaller.
        _xbins = std::max(int(cg->lmp_data.box_dx(step)/cutoff), 1);
        _ybins = std::max(int(cg->lmp_data.box_dy(step)/cutoff), 1);
        _zbins = std::max(int(cg->lmp_data.box_dz(step)/cutoff), 1);
        _bins.assign(_xbins*_ybins*_zbins, {});
        for (size_t i=0; i<cg->beads.size(); ++i) {
            _bins[_find_bin(i)].push_back(i);
        }
    }

    // Returns an array of all atoms that can be within cutoff of i.
    // This might not be the best implimentation because a lot of 
    // vectors need to be copied.
    std::vector<int> BinTable::neighbors(int i) const {
        // Scaled coordinate of the atom.
        int bi = _find_bin(i);
        int bx = bi % _xbins;
        int by = (bi/_xbins)%_ybins;
        int bz = (bi/_xbins/_ybins);
        std::vector<int> atoms;
        for (int i: {-1, 0, 1}) {
            if ((i == -1 && _xbins < 2) ||
                (i ==  1 && _xbins < 3)) {
                continue;
            }
            int nx = (bx+i+_xbins) % _xbins;
            for (int j: {-1, 0, 1}) {
                if ((j == -1 && _ybins < 2) ||
                    (j ==  1 && _ybins < 3)) {
                    continue;
                }
                int ny = (by+j+_ybins) % _ybins;
                for (int k: {-1, 0, 1}) {
                    if ((k == -1 && _zbins < 2) ||
                        (k ==  1 && _zbins < 3)) {
                        continue;
                    }
                    int nz = (bz+k+_zbins) % _zbins;
                    int b = nx + ny*_xbins + nz*_xbins*_ybins;
                    atoms.insert(atoms.end(), _bins[b].begin(), _bins[b].end());
                }
            }
        }
        return atoms;
    }

    // Returns the index of the bin.
    int BinTable::_find_bin(int i) const {
        auto box = _cgsys->lmp_data.box_size(_step);
        auto r = _cgsys->bead_coordinate(_step, i);
        int bx = std::min(int(double(_xbins)*r(0)/box(0)), _xbins-1);
        int by = std::min(int(double(_ybins)*r(1)/box(1)), _ybins-1);
        int bz = std::min(int(double(_zbins)*r(2)/box(2)), _zbins-1);
        return bx + by*_xbins + bz*_xbins*_ybins;
    }
}

