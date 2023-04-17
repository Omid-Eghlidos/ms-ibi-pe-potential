#include "geometry.h"
#include "catch.hpp"

TEST_CASE("torsion angle is computed correctly", "[torsion]") {
    Vector<3> x1 = {0.0,0.0,0.1};
    Vector<3> x2 = {1.0,0.0,0.0};
    Vector<3> x3 = {2.0,0.0,0.0};
    Vector<3> x4 = {3.0,0.0,-0.1};
    auto q = torsion_angle(x1, x2, x3, x4);
    REQUIRE(q == Approx(acos(-1)));
}
