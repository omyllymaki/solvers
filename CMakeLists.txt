cmake_minimum_required(VERSION 3.10)

add_definitions(-DBOOST_ALL_DYN_LINK)

find_package(Armadillo REQUIRED)
find_package(Boost COMPONENTS log)

ADD_LIBRARY(MathLib STATIC
            src/common.cpp
            src/solver.cpp
            src/ls_solver.cpp
            src/nnls_solver.cpp
            src/gd_solver.cpp
            src/gd_linear_solver.cpp)

link_libraries(MathLib)
link_libraries(${ARMADILLO_LIBRARIES})
link_libraries(Boost::log)

add_subdirectory(tests)
add_subdirectory(samples)
