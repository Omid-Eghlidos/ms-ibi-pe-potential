#include "histogram.h"
#include <iostream>
#include <fstream>
#include <numeric>
#include <cmath>
#include <functional>

using std::cout;
using std::string;
using std::fstream;

namespace cg {

    // Initializes a histogram with zero bins.
    Histogram::Histogram(double x_min, double x_max, double dx, double kde_width) {
        double range = x_max - x_min;
        int n = (int)ceil(range/dx);
        _x0 = x_min;
        _dx = range / double(n);
        _dxi = 1.0/_dx;
        _kde_width = kde_width;
        histogram.assign(n, 0.0);
    }

    //! Adds a new value to a bin in the histogram.
    void Histogram::add(double x) {
        size_t bin = (size_t)floor((x - _x0) * _dxi);
        if (bin < histogram.size()) histogram[bin] += 1.0;
    }
    //! Add values to the histogram with kernel density estimation.
    void Histogram::kde_add(double x) {
        size_t bin = (size_t)floor((x - _x0) * _dxi);
        double w = _kde_width * _dx;
        double Z = 1.0/sqrt(2*acos(-1.0))/w;
        for (size_t i = bin; i < histogram.size(); ++i){
            auto r = (i+0.5)*_dx - x;
            auto dg = Z*exp(-0.5*r*r/w/w)*_dx;
            if (histogram[i] > 1e16*dg) break;
            histogram[i] += dg;
        }
        for (int i = bin-1; i >= 0 && i < (int)histogram.size(); --i){
            auto r = (i+0.5)*_dx - x;
            auto dg = Z*exp(-0.5*r*r/w/w)*_dx;
            if (histogram[i] > 1e16*dg) break;
            histogram[i] += dg;
        }
        if (bin < histogram.size() + 10*_kde_width && bin > histogram.size()) {
            for (auto i = histogram.size()-10*_kde_width; i < histogram.size(); i++ ){
                auto r = (i+0.5)*_dx - x;
                auto dg = Z*exp(-0.5*r*r/w/w)*_dx;
                histogram[i] += dg;
            }
        }
    }
    // Writes a histogram to a file readable by gnuplot.
    void Histogram::write(string path) const {
        std::fstream fid(path, std::ios::out);
        double x(_x0 + 0.5*_dx);
        for (auto h=histogram.begin(); h!=histogram.end(); ++h, x+=_dx) {
            fid << x << "\t" << *h << "\n";
        }
    }

    // Scales the histogram by a factor and by radial density.
    void Histogram::scale(double s, bool radial_density) {
        static const double four_thirds_pi = 4.0/3.0*acos(-1.0);
        double x(_x0);
        for (auto &y: histogram) {
            if (radial_density) {
                auto xp = x + _dx;
                auto vshell = four_thirds_pi*(xp*xp*xp - x*x*x);
                // If RDF then we want count per shell volume.
                y *= s/vshell;
                x = xp;
            }
            else {
                y *= s;
            }
        }
    }

    // Adds two (compatible) histograms.
    Histogram& Histogram::operator+=(const Histogram &y) {
        if (histogram.empty()) {
            *this = y;
            return *this;
        }
        if (y.histogram.size() != histogram.size())
            cout << "Error in histogram add: wrong # of bins.\n";
        else if (y._dx != _dx) cout << "Error in histogram add: wrong dx.\n";
        else if (y._x0 != _x0) cout << "Error in histogram add: wrong x0.\n";
        else {
            auto hi = histogram.begin();
            auto yi = y.histogram.cbegin();
            for (; hi!=histogram.end(); ++hi, ++yi) {
                *hi += *yi;
            }
        }
        return *this;
    }

    // Sum of all histogram bins.
    double Histogram::sum() const {
        return std::accumulate(histogram.begin(), histogram.end(), 0.0);
    }

    //! Divides the histogram by the sum of its contents.
    void Histogram::normalize() {
        double d = sum();
        if (d == 0) {
            cout << "Warning: histogram is empty.\n";
            return;
        }
        else d = _dxi / d;
        for (auto h=histogram.begin(); h!=histogram.end(); ++h) *h *= d;
    }
}
