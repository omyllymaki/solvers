cmake_minimum_required(VERSION 3.10)

add_definitions(-DBOOST_ALL_DYN_LINK)

find_package(Armadillo REQUIRED)
find_package(Boost COMPONENTS log)

link_libraries(${ARMADILLO_LIBRARIES})
link_libraries(Boost::log)

include(CTest)

add_executable(generate_test_data.exe samples/generate_test_data.cpp)
add_executable(analyze.exe samples/analyze.cpp src/ls.cpp src/common.cpp)
add_executable(ls_fit_test.exe tests/ls_fit_test.cpp src/ls.cpp src/common.cpp)
add_executable(nnls_fit_test.exe tests/nnls_fit_test.cpp src/ls.cpp src/nnls.cpp src/common.cpp)
add_executable(svd_inverse_test.exe tests/svd_inverse_test.cpp src/common.cpp)

add_test(ls_fit_test.exe ls_fit_test.exe)
add_test(nnls_fit_test.exe nnls_fit_test.exe)
add_test(svd_inverse_test.exe svd_inverse_test.exe)