cmake_minimum_required(VERSION 3.10)

add_definitions(-DBOOST_ALL_DYN_LINK)

find_package(Armadillo REQUIRED)
find_package(Boost COMPONENTS log)

ADD_LIBRARY(MathLib STATIC
            src/ls.cpp 
            src/nnls.cpp 
            src/gd_fit.cpp 
            src/common.cpp)

link_libraries(MathLib)
link_libraries(${ARMADILLO_LIBRARIES})
link_libraries(Boost::log)

add_executable(generate_test_data.exe samples/generate_test_data.cpp)
add_executable(analyze.exe samples/analyze.cpp)
add_executable(ls_fit_test.exe tests/ls_fit_test.cpp)
add_executable(nnls_fit_test.exe tests/nnls_fit_test.cpp)
add_executable(common_test.exe tests/common_test.cpp)
add_executable(gd_fit_test.exe tests/gd_fit_test.cpp)

include(CTest)
add_test(ls_fit_test.exe ls_fit_test.exe)
add_test(nnls_fit_test.exe nnls_fit_test.exe)
add_test(common_test.exe common_test.exe)
add_test(gd_fit_test.exe gd_fit_test.exe)