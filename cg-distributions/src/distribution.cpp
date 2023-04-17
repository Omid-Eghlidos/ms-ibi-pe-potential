#include "distribution.h"
#include "cgsystem.h"
#include "input.h"
#include "lammps_data.h"
#include "bintable.h"
#include <cmath>
#include <numeric>
#include <omp.h>
#include <fstream>
#include <set>
#include "string_tools.h"
#include "vector.h"
#include "geometry.h"

const double pi = acos(-1.0);
const double rad2deg = 180.0/acos(-1.0);

namespace cg {
    Distribution::Distribution(const CgSystem &s, const InputSettings &i)
      : system(s), input_settings(i), lmp_data(s.lmp_data)
    {}

    // Computes all pair distance distributions over the dump timesteps.
    void Distribution::compute_pair_correlations() const {
        auto pairs = system.unique_pair_types();
        std::map<PairTypes, Histogram> rdf;

        std::cout << "Computing RDFs ";
        int ct = 0, progress = 0, n_steps = lmp_data.num_timesteps();
        #pragma omp parallel for
        for (int ts=0; ts<n_steps; ++ts) {
            for (auto &t: pairs) {
                Histogram temp(input_settings.rdf.lo,
                               input_settings.rdf.hi,
                               input_settings.rdf.delta);
                if (input_settings.rdf_kde) temp = compute_rdf_kde(t, ts);
                else temp = compute_rdf(t, ts);
                #pragma omp critical
                rdf[t] += temp;
            }
            #pragma omp critical
            ct += 1;
            // Show 10 progress * indictators.
            if (omp_get_thread_num() == 0) {
                if (10*ct/n_steps > progress)  {
                    progress += 1;
                    std::cout << "*";
                }
            }
        }
        std::cout << " done.";
        for (auto &r: rdf) {
            if (r.second.histogram.size()==0) return;
            r.second.scale(1.0 / lmp_data.num_timesteps(), false);
            r.second.write(input_settings.output_path + "rdf-"
                         + input_settings.output_tag + "_"
                         + std::get<0>(r.first) + std::get<1>(r.first) + ".txt");
        }
    }

    // Computes all bond length distributions over the dump timesteps.
    void Distribution::compute_bond_correlations() const {
        // Assuming bond type are the same as pair types.
        auto bonds = system.unique_pair_types();
        std::map<PairTypes, Histogram> bdf;

        #pragma omp parallel for
        for (int ts=0; ts<lmp_data.num_timesteps(); ++ts) {
            for (auto &t: bonds) {
                auto temp = compute_bdf(t, ts);
                #pragma omp critical
                bdf[t] += temp;
            }
        }
        for (auto &b: bdf) {
            if (b.second.sum() > 0.0) {
                b.second.normalize();
                b.second.write(input_settings.output_path + "bdf-"
                              + input_settings.output_tag + "_"
                              + std::get<0>(b.first) + std::get<1>(b.first)
                              + ".txt");
            }
        }
    }

    // Computes all bond angle distributions over the dump timesteps.
    void Distribution::compute_angle_correlations() const {
        auto angles = system.unique_angle_types();
        std::map<AngleTypes, Histogram> adf;
        std::map<AngleTypes, std::vector<std::vector<double>>> badata;
        #pragma omp parallel for
        for (int ts=0; ts<lmp_data.num_timesteps(); ++ts) {
            for (auto &t: angles) {
                auto temp = compute_adf(t, ts);
                auto batemp = std::get<1>(temp);
                #pragma omp critical
                {
                adf[t] += std::get<0>(temp);
                badata[t].insert(badata[t].end(),batemp.begin(),batemp.end());
                }
            }
        }
        for (auto &a: adf) {
           if (a.second.sum() > 0.0) {
               a.second.normalize();
               a.second.write(input_settings.output_path + "adf-"
                             + input_settings.output_tag + "_"
                             + std::get<0>(a.first)
                             + std::get<1>(a.first)
                             + std::get<2>(a.first) + ".txt");
           }
        }


        for (auto &t: angles){
            if(badata[t].size()>0){
                    std::fstream fid(input_settings.output_path + "ba-"
                                    + input_settings.output_tag + "_"
                                    + std::get<0>(t)
                                    + std::get<1>(t)
                                    + std::get<2>(t)+".txt", std::ios::out);
                    for(auto &bat: badata[t]){
                        fid<<bat[0]<<"\t"<<bat[1]<<"\t"<<bat[2]<<"\n";
                    }
                    fid.close();
            }
        }

    }

    // Computes all pair distance distributions over the dump timesteps.
    void Distribution::compute_density_field() const {
        auto bead_types = system.defined_bead_types();
        std::map< std::string, double> rho;
        #pragma omp parallel for
        for (int ts=0; ts<lmp_data.num_timesteps(); ++ts) {
            for (auto &t: bead_types) {
                auto temp = compute_density(t, ts);
                #pragma omp critical
                rho[t] += temp;
            }
        }
        std::fstream fid("density_field-"+input_settings.output_tag+".txt", std::ios::out);
        for(auto &rhoi:rho){
            rhoi.second *= 1.0/lmp_data.num_timesteps();
            fid << rhoi.first << "\t" << rhoi.second << "\n";
        }
    }

    // Computes all bond torsion angle distributions over the dump timesteps.
    void Distribution::compute_torsion_correlations() const {
        auto torsions = system.unique_torsion_types();
        std::map<TorsionTypes, Histogram> tdf;
        std::map<TorsionTypes, std::vector<std::vector<double>>> dcdata;
        #pragma omp parallel for
        for (int ts=0; ts<lmp_data.num_timesteps(); ++ts) {
            for (auto &t: torsions) {
                auto temp = compute_tdf(t, ts);
                auto dctemp = std::get<1>(temp);
                #pragma omp critical
                {
                tdf[t] += std::get<0>(temp);
                dcdata[t].insert(dcdata[t].end(), dctemp.begin(), dctemp.end());
                }
            }
        }
        for (auto &tor: tdf) {
            if (tor.second.sum() > 0.0){
                tor.second.normalize();
                tor.second.write(input_settings.output_path + "tdf-"
                                + input_settings.output_tag + "_"
                                + std::get<0>(tor.first)
                                + std::get<1>(tor.first)
                                + std::get<2>(tor.first)
                                + std::get<3>(tor.first) + ".txt");
            }
        }

        for (auto &t: torsions){
            if(dcdata[t].size()>0){
                    std::fstream fid(input_settings.output_path + "dc-"
                                    + input_settings.output_tag + "_"
                                    + std::get<0>(t)
                                    + std::get<1>(t)
                                    + std::get<2>(t)
                                    + std::get<3>(t)
                                    +".txt", std::ios::out);
                    // Header of the 6 parameters of the dihedral torsions
                    fid<<"a"<<"\t"<<"b"<<"\t"<<"c"<<"\t"
                       <<"q1"<<"\t"<<"q2"<<"\t"<<"phi"<<"\n";
                    for(auto &dct: dcdata[t]){
                        fid<<dct[0]<<"\t"<<dct[1]<<"\t"<<dct[2]<<"\t"
                           <<dct[3]<<"\t"<<dct[4]<<"\t"<<dct[5]<<"\n";
                    }
                    fid.close();
            }
        }
    }

    // Computes all bond torsion angle distributions over the dump timesteps.
    void Distribution::compute_improper_correlations() const {
        auto impropers = system.unique_improper_types();
        std::map<ImproperTypes, Histogram> idf;
        std::map<ImproperTypes, std::vector<std::vector<double>>> icdata;
        #pragma omp parallel for
        for (int ts=0; ts<lmp_data.num_timesteps(); ++ts) {
            for (auto &t: impropers) {
                auto temp = compute_idf(t, ts);
                auto ictemp = std::get<1>(temp);
                #pragma omp critical
                {
                idf[t] += std::get<0>(temp);
                icdata[t].insert(icdata[t].end(), ictemp.begin(), ictemp.end());
                }
            }
        }
        for (auto &tor: idf) {
            if (tor.second.sum() > 0.0){
                tor.second.normalize();
                tor.second.write(input_settings.output_path + "idf-"
                                + input_settings.output_tag + "_"
                                + std::get<0>(tor.first)
                                + std::get<1>(tor.first)
                                + std::get<2>(tor.first)
                                + std::get<3>(tor.first) + ".txt");
            }
        }

        for (auto &t: impropers){
            if(icdata[t].size()>0){
                    std::fstream fid("ic-"+input_settings.output_tag+"_"
                                    + std::get<0>(t)
                                    + std::get<1>(t)
                                    + std::get<2>(t)
                                    + std::get<3>(t)
                                    +".txt", std::ios::out);
                    // Header of the 7 parameters of the impropers
                    fid<<"a"<<"\t"<<"b"<<"\t"<<"c"<<"\t"
                       <<"q1"<<"\t"<<"q2"<<"\t"<<"q3"<<"\t"<<"psi"<<"\n";
                    for(auto &ict: icdata[t]){
                        fid<<ict[0]<<"\t"<<ict[1]<<"\t"<<ict[2]<<"\t"
                           <<ict[3]<<"\t"<<ict[4]<<"\t"<<ict[5]<<"\t"
                           <<ict[6]<<"\n";
                    }
                    fid.close();
            }
        }
    }

    // Computes the radial distribution function between beads.
    Histogram Distribution::compute_rdf(const PairTypes &t, int step) const {

        Histogram rdf(input_settings.rdf.lo,
                      input_settings.rdf.hi,
                      input_settings.rdf.delta);
        auto box = lmp_data.box_size(step);
        BinTable bintable(input_settings.rdf.hi, step, &system);
        // Loop over all pairs of beads (w/ no cutoff).
        // If bead i matches b1 or b2, then bead j must match the other.
        for (auto &bi: system.beads) {
            std::string j_type;
            if      (bi.type == std::get<0>(t)) j_type = std::get<1>(t);
            else if (bi.type == std::get<1>(t)) j_type = std::get<0>(t);
            else continue;
            //  Use bintable for efficient neighbor searching.
            for (auto j: bintable.neighbors(bi.index)) {
                auto &bj = system.beads[j];
                if (j >= bi.index || bj.type != j_type) {
                    continue;
                }
                if (input_settings.special_bonds[0] == 0 
                    && system.neighbors(bi, bj)) {
                    continue;
                }
                if (input_settings.special_bonds[1] == 0
                    && system.second_neighbors(bi, bj)) {
                    continue;
                }
                if (input_settings.special_bonds[2] == 0) {
                    std::cerr << "ERROR: Special bonds for torsions not implemented!!!\n";
                    exit(1);
                }
                auto v = system.distance_vector(step, bi.index, bj.index
                                                    , lmp_data.triclinic_box);
                for (auto r: pair_distances(v, box, input_settings.rdf.hi
                                                  , lmp_data.triclinic_box)) {
                    rdf.add(r);
                }
            }
        }
        auto n1 = system.bead_count(std::get<0>(t));
        auto n2 = system.bead_count(std::get<1>(t));
        auto s = lmp_data.volume(step, lmp_data.triclinic_box) / (double(n1)*double(n2));

        if (std::get<0>(t) == std::get<1>(t)) {
            s *= 2.0;
        }
        rdf.scale(s, true);
        return rdf;
    }

    Histogram Distribution::compute_rdf_kde(const PairTypes &t, int step) const {
        Histogram rdf(input_settings.rdf.lo,
                      input_settings.rdf.hi,
                      input_settings.rdf.delta,
                      input_settings.kde_width);
        auto box = lmp_data.box_size(step);
        BinTable bintable(input_settings.rdf.hi, step, &system);
        // Loop over all pairs of beads (w/ no cutoff).
        // If bead i matches b1 or b2, then bead j must match the other.
        for (auto &bi: system.beads) {
            std::string j_type;
            if      (bi.type == std::get<0>(t)) j_type = std::get<1>(t);
            else if (bi.type == std::get<1>(t)) j_type = std::get<0>(t);
            else continue;
            //  Use bintable for efficient neighbor searching.
            for (auto j: bintable.neighbors(bi.index)) {
                auto &bj = system.beads[j];
                if (j >= bi.index || bj.type != j_type) {
                    continue;
                }
                if (input_settings.special_bonds[0] == 0
                    && system.neighbors(bi, bj)) {
                    continue;
                }
                if (input_settings.special_bonds[1] == 0
                    && system.second_neighbors(bi, bj)) {
                    continue;
                }
                if (input_settings.special_bonds[2] == 0) {
                    std::cerr << "ERROR: Special bonds for torsions not implemented!!!\n";
                    exit(1);
                }
                auto v = system.distance_vector(step, bi.index, bj.index
                                                    , lmp_data.triclinic_box);
                for (auto r: pair_distances(v, box, input_settings.rdf.hi
                                                  , lmp_data.triclinic_box )) {
                    rdf.kde_add(r);
                }
            }
        }
        auto n1 = system.bead_count(std::get<0>(t));
        auto n2 = system.bead_count(std::get<1>(t));
        auto s = lmp_data.volume(step, lmp_data.triclinic_box) / (double(n1)*double(n2));

        if (std::get<0>(t) == std::get<1>(t)) {
            s *= 2.0;
        }
        rdf.scale(s, true);
        return rdf;
    }

    // Computes the radial distribution function between beads.
    double Distribution::compute_density(const std::string &t, int step) const {
        auto temprho = 0.0;
        auto rcut = input_settings.rhobar.rcut;
        auto box = lmp_data.box_size(step);
        auto path = "df-" + input_settings.output_tag + "_" + t + "_step_"
                   + std::to_string(step) + ".txt";
        std::fstream fid1(path, std::ios::out);
        BinTable bintable(rcut, step, &system);
        // Loop over all pairs of beads (w/ no cutoff).
        // If bead i matches b1 or b2, then bead j must match the other.
        int num_samples = 0;
        for (auto bi: system.beads) {
            if (bi.type != t) continue;
            num_samples++;
            auto temprho1 = 0.0;
            for (auto j: bintable.neighbors(bi.index))  {
                auto v = system.distance_vector(step, bi.index, j
                                                    , lmp_data.triclinic_box);
                for (auto ri: pair_distances(v, box, rcut
                                                   , lmp_data.triclinic_box)) {
                    auto &jtype = system.beads[j].type;
                    auto mscale = input_settings.get_bead_type(jtype).mscale;
                    if (ri<rcut) {
                        temprho1 += 15.0*(1.0-ri/rcut)*(1.0-ri/rcut)*mscale/(2.0*pi*rcut*rcut*rcut);
                    }
                }
            }
            fid1 << temprho1 << "\n";
            temprho += temprho1;
        }
        temprho *= 1.0/num_samples;
        return temprho;
    }

    // Computes the bond-length distrubution function between beads.
    Histogram Distribution::compute_bdf(const PairTypes &t, int step) const {
        Histogram bdf(input_settings.bdf.lo, input_settings.bdf.hi,
                      input_settings.bdf.delta);

        // Loop over all pairs of beads (w/ no cutoff).
        for (auto &bi: system.beads) {
            // What type bead j must match.
            std::string j_type;
            if      (bi.type == std::get<0>(t)) j_type = std::get<1>(t);
            else if (bi.type == std::get<1>(t)) j_type = std::get<0>(t);
            else continue;
            // TODO: bonds are probably computed twice.
            for (auto j: bi.neighbors) {
                auto &bj = system.beads[j];
                if (bj.type != j_type) continue;
                auto d = norm(system.distance_vector(step, bi.index, j
                                                   , lmp_data.triclinic_box));
                bdf.add(d);
            }
        }
        return bdf;
    }

    // Computes the bond-angle distribution function between beads.
    BondAngle Distribution::compute_adf(const AngleTypes &t, int step) const {
        std::vector<std::vector<double>> bastep;
        auto rad2deg = 180.0/acos(-1.0);
        Histogram adf(input_settings.adf.lo, input_settings.adf.hi,
                      input_settings.adf.delta);
        // Each bead may be the center of one or more angles.
        for (auto &bj: system.beads) {
            if (bj.type != std::get<1>(t)) continue;
            const int j = bj.index;
            // Consider all possible combinations of angles.
            for (int i: bj.neighbors) {
                for (int k: bj.neighbors) {
                    if (i < k) {
                        // Verify that the bead types match this adf.
                        if (system.beads[i].type != std::get<0>(t)) {
                            std::swap(i, k);
                        }
                        if (system.beads[i].type != std::get<0>(t) ||
                            system.beads[k].type != std::get<2>(t)) {
                            continue;
                        }

                        auto a = norm(system.distance_vector(step, i, j
                                                    , lmp_data.triclinic_box));
                        auto b = norm(system.distance_vector(step, j, k
                                                    , lmp_data.triclinic_box));
                        auto c = norm(system.distance_vector(step, i, k
                                                    , lmp_data.triclinic_box));
                        auto q = rad2deg * acos((a*a + b*b - c*c)/(2.0*a*b));
                        adf.add(q);

                        if (input_settings.use_bond_angle) {
                            bastep.push_back({a, b, q});
                        }
                    }
                }
            }
        }
        return std::make_tuple(adf,bastep);
    }

    // Computes the bond-torsion-angle distrubution function between beads.
    BondAngle Distribution::compute_tdf(const TorsionTypes &t, int step) const {
        std::vector<std::vector<double>> dcstep;
        std::string t1, t2, t3, t4;
        std::tie(t1,t2,t3,t4) = t;
        Histogram tdf(input_settings.tdf.lo, input_settings.tdf.hi,
                      input_settings.tdf.delta);
        // Construct dihedrals: i-j-k-l, starting from j.
        for (auto &bj: system.beads) {
            // To ensure that torsions aren't double counted, enforce j < k.
            for (int i: bj.neighbors) {
                for (int k: bj.neighbors) {
                    if (i == k || k >= bj.index) continue;
                    auto &bi = system.beads[i], &bk = system.beads[k];
                    for (int l: bk.neighbors) {
                        // Avoid backtracking.
                        if (l == bj.index) continue;
                        auto &bl = system.beads[l];
                        bool match = bi.type==t1 && bj.type==t2
                                  && bk.type==t3 && bl.type==t4;
                        bool match_flip = bi.type==t4 && bj.type==t3
                                       && bk.type==t2 && bl.type==t1;
                        // Check that types match (with symmetry ABCD = DCBA).
                        if (match || match_flip) {
                            auto x1 = system.bead_coordinate(step, bj.index);
                            auto x0 = x1 + system.distance_vector(step, i, bj.index
                                                     , lmp_data.triclinic_box);
                            auto x2 = x1 + system.distance_vector(step, k, bj.index
                                                     , lmp_data.triclinic_box);
                            auto x3 = x1 + system.distance_vector(step, l, bj.index
                                                     , lmp_data.triclinic_box);
                            double phi;
                            if (match) {
                                phi = rad2deg * torsion_angle(x0, x1, x2, x3);
                            }
                            else {
                                phi = rad2deg * torsion_angle(x3, x2, x1, x0);
                            }
                            tdf.add(-phi);

                            if (input_settings.use_bond_angle) {
                                auto a = norm(system.distance_vector(step, bj.index, i
                                                     , lmp_data.triclinic_box));
                                auto b = norm(system.distance_vector(step, k, bj.index
                                                     , lmp_data.triclinic_box));
                                auto c = norm(system.distance_vector(step, l, k
                                                     , lmp_data.triclinic_box));
                                auto d = norm(system.distance_vector(step, k, i
                                                     , lmp_data.triclinic_box));
                                auto e = norm(system.distance_vector(step, l, bj.index
                                                     , lmp_data.triclinic_box));
                                auto q1 = rad2deg * acos((a*a + b*b - d*d)/(2.0*a*b));
                                auto q2 = rad2deg * acos((b*b + c*c - e*e)/(2.0*b*c));
                                dcstep.push_back({a, b, c, q1, q2, -phi});
                            }
                        }
                    }
                }
            }
        }
        return std::make_tuple(tdf, dcstep);
    }

    // Computes the improper-torsion-angle distrubution function between beads.
    BondAngle Distribution::compute_idf(const ImproperTypes &t, int step) const {
        std::vector<std::vector<double>> icstep;
        std::string t1, t2, t3, t4;
        std::tie(t1,t2,t3,t4) = t;
        Histogram idf(input_settings.idf.lo, input_settings.idf.hi,
                      input_settings.idf.delta);
        // Construct dihedrals: i-j-k-l, starting from j.
        for (auto &bj: system.beads) {
            if (bj.neighbors.size() == 3) {
                int i = bj.neighbors[0];
                int k = bj.neighbors[1];
                int l = bj.neighbors[2];
                auto &bi = system.beads[i], &bk = system.beads[k], &bl = system.beads[l];
                // Check that types match (with symmetry ABCD = DCBA).
                if ((bi.type==t1 && bj.type==t2 && bk.type==t3 && bl.type==t4)
                 || (bi.type==t4 && bj.type==t3 && bk.type==t2 && bl.type==t1)) {
                    auto x1 = system.bead_coordinate(step, bj.index);
                    auto x0 = x1 + system.distance_vector(step, i, bj.index
                                                    , lmp_data.triclinic_box);
                    auto x2 = x1 + system.distance_vector(step, k, bj.index
                                                    , lmp_data.triclinic_box);
                    auto x3 = x1 + system.distance_vector(step, l, bj.index
                                                    , lmp_data.triclinic_box);
                    auto psi = rad2deg * improper_angle(x0, x1, x2, x3);
                    idf.add(psi);

                    if (input_settings.use_bond_angle) {
                        auto a = norm(system.distance_vector(step, i, bj.index
                                                     , lmp_data.triclinic_box));
                        auto b = norm(system.distance_vector(step, k, bj.index
                                                     , lmp_data.triclinic_box));
                        auto c = norm(system.distance_vector(step, l, bj.index
                                                     , lmp_data.triclinic_box));
                        auto d = norm(system.distance_vector(step, k, i
                                                     , lmp_data.triclinic_box));
                        auto e = norm(system.distance_vector(step, l, k
                                                     , lmp_data.triclinic_box));
                        auto f = norm(system.distance_vector(step, l, i
                                                     , lmp_data.triclinic_box));
                        auto q1 = rad2deg * acos((a*a + b*b - d*d)/(2.0*a*b));
                        auto q2 = rad2deg * acos((b*b + c*c - e*e)/(2.0*b*c));
                        auto q3 = rad2deg * acos((a*a + c*c - f*f)/(2.0*a*c));
                        icstep.push_back({a, b, c, q1, q2, q3, psi});
                    }
                }
            }
        }
        return std::make_tuple(idf, icstep);
    }
}

