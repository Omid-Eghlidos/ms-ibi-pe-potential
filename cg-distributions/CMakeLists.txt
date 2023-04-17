cmake_minimum_required (VERSION 2.8.8)
project (cg-distributions)

find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED COMPONENTS)
find_package(ZLIB REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -std=c++14 -Wall")
set(COMMON_SRC src/bintable.cpp src/cgsystem.cpp src/distribution.cpp src/histogram.cpp src/input.cpp src/lammps_data.cpp)

add_executable(cg-distributions src/main.cpp)
add_executable(tests src/test/main.cpp src/test/test_geometry.cpp src/test/test_bdf.cpp
               src/test/test_rdf.cpp src/test/test_adf.cpp src/test/test_tdf.cpp)
add_library(common ${COMMON_SRC})

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories( ${EIGEN3_INCLUDE_DIR} )

target_link_libraries(cg-distributions common ${Boost_LIBRARIES})
target_link_libraries(tests common ${Boost_LIBRARIES})

set(CMAKE_BUILD_TYPE RELEASE)

#install(TARGETS fast-split segment RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin/)
