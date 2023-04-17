#pragma once
#include <vector>
#include <string>
#include <map>

namespace cg {

    class LAMMPS_Data;

    //! A histogram with a bin size and starting bin position.
    class Histogram {
    public:
        //! Initializes a histogram with zero bins.
        Histogram(double x_min=0.0, double x_max=0.0, double dx=1.0, double kde_width = 1.5);
        //! Adds a new value to a bin in the histogram.
        void add(double x);
        //! Adds a data point to the histogram in the kde fashion.
        void kde_add(double x);
        //! Writes a histogram to a file readable by gnuplot.
        void write(std::string path) const;
        //! Scales the histogram by a factor and by radial density.
        void scale(double s, bool radial_density);
        //! Sum of all histogram bins.
        double sum() const;
        //! Divides the histogram by the sum of its contents.
        void normalize();

        Histogram& operator+=(const Histogram &y);
        //! Returns the number of bins.
        size_t size() const { return histogram.size(); }

        std::vector<double> histogram;

        double min() const { return _x0; }
        double max() const { return _x0 + _dx*double(histogram.size()); }
        double bin_size() const { return _dx; }

    private:
        double _x0, _dx, _dxi, _kde_width;
    };

}
