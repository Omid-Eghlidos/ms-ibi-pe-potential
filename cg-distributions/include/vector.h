#pragma once
#include <cmath>
#include <iostream>
#include <vector>

// Fixed size vector.  Size of vector must be known at compile time.
template <int N>
class Vector {
public:
    Vector() {}
    Vector(std::initializer_list<double> x);
    double  operator()(int i) const { return _x[i]; }
    double& operator()(int i) { return _x[i]; }
    double* begin() { return _x; }
    const double* begin() const { return _x; }
    double* end() { return _x+N; }
    const double* end() const { return _x+N; }
    int size() const { return N; }

    // Adds scaled-vector (v*s) to this vector.
    void scaled_add(const Vector &v, double s=1.0) {
        for (int i=0; i<N; ++i) _x[i] += v(i)*s;
    }

private:
    double _x[N];
};

// Define some handy shortcuts.
typedef Vector<3> Vec3;

template <int N>
Vector<N>::Vector(std::initializer_list<double> x) {
  std::copy(x.begin(), x.end(), _x);
}

//! Adds to fixed size vectors together.
template<int N>
Vector<N> operator+(Vector<N> x, const Vector<N> &y) {
    x += y;
    return x;
}
//! In place addition operator.
template<int N>
Vector<N>& operator+=(Vector<N> &x, const Vector<N> &y) {
    for (int i=0; i<N; ++i) x(i) += y(i);
    return x;
}

template<int N>
Vector<N> operator-(Vector<N> x, const Vector<N> &y) {
    x -= y;
    return x;
}

template<int N>
Vector<N>& operator-=(Vector<N> &x, const Vector<N> &y) {
    for (int i=0; i<N; ++i) x(i) -= y(i);
    return x;
}

// Scalar-vector product. 
template<int I>
Vector<I> operator*(const double a, Vector<I> x) {
    for (auto &v: x) v *= a;
    return x;
}

template<int I>
Vector<I> operator*(const Vector<I> &x, const double a) {
    return a*x;
}

template<int I>
Vector<I> operator/(Vector<I> x, double a) {
    x *= (1.0/a);
    return x;
}

template<int I>
Vector<I>& operator*=(Vector<I> &x, double a) {
    for (int i=0; i<I; ++i) x(i) *= a;
    return x;
}


template <int N>
double dot(const Vector<N> &x, const Vector<N> &y) {
    double r = 0.0;
    for (int i=0; i<N; ++i) r += x(i)*y(i);
    return r;
}

// Computes a cross product for R3 vectors.
inline Vector<3> cross(const Vector<3> &x, const Vector<3> &y) {
    Vector<3> z;
    z(0) = x(1)*y(2)-x(2)*y(1);
    z(1) = x(2)*y(0)-x(0)*y(2);
    z(2) = x(0)*y(1)-x(1)*y(0);
    return z;
}

// Returns the maximum of a vector.
template <int N>
double max(const Vector<N> &x) {
    double m = x(0);
    for (auto xi: x) {
        if (m < xi) m = xi;
    }
    return m;
}

// Returns the minimum of a vector.
template <int N>
double min(const Vector<N> &x) {
    double m = x(0);
    for (auto xi: x) {
        if (m > xi) m = xi;
    }
    return m;
}

template <int I>
double sum(const Vector<I> &x) {
    double s = 0.0;
    for (auto _x: x) s += _x;
    return s;
}

template <int N>
double norm(const Vector<N> &x) {
    double r = 0.0;
    for (auto _x: x) r += _x*_x;
    return sqrt(r);
}

template<int N>
Vector<N> abs(Vector<N> v) {
    for (auto &x: v) x = std::fabs(x);
    return v;
}

template <int N>
Vector<N> zeros() {
    Vector<N> v;
    for (auto &x: v) x = 0.0;
    return v;
}


// Output vector to a stream (like cout).
template<int N>
std::ostream& operator<<(std::ostream &o, const Vector<N> &x) {
    if (N == 0) return o << "()";
    o << "(" << x(0);
    for (int i=1; i<N; ++i) {
        o << ", " << x(i);
    }
    return o << ")";
}

