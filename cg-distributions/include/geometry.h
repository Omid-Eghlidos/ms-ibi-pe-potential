#pragma once
#include "vector.h"

/*! Compute the dihedral angle formed by x1--x2--x3--x4.
 *! According to: http://math.stackexchange.com/a/47084
 *! This is the angle between the normals of the two planes.  */
inline double torsion_angle(const Vec3 &x1, const Vec3 &x2, 
                            const Vec3 &x3, const Vec3 &x4) {
    auto r1 = (x2-x1)/norm(x2-x1);
    auto r2 = (x3-x2)/norm(x3-x2);
    auto r3 = (x4-x3)/norm(x4-x3);
    auto mm = cross(r1,r2);
    auto nn = cross(r2,r3);
    auto y = dot(cross(mm,r2),nn);
    auto x = dot(mm,nn);
    return atan2(y,x);
}


/* 
 * temporary
improper_angle(x1, x2, x3, x4) + 
improper_angle(x4, x2, x1, x3) + 
improper_angle(x3, x2, x4, x1) +   */


/*! Computes the improper angle formed by a quadruplet of atoms where
 *  atom j is bonded to all other atoms (class2 style).
 *  E.g. chi_ijkl is the angle between the plane of ijk and jkl.    */
inline double improper_angle(const Vec3& xi, const Vec3& xj, 
                             const Vec3& xk, const Vec3& xl)  {
    auto x_ji = xi - xj;
    auto x_jk = xk - xj;
    auto x_jl = xl - xj;
    auto n1 = cross(x_ji, x_jk);
    auto n2 = cross(x_jk, x_jl);
    return acos(dot(n1, n2)/(norm(n1)*norm(n2)));
}
