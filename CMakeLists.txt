cmake_minimum_required(VERSION 3.10)

add_definitions(-DBOOST_ALL_DYN_LINK)

find_package(Armadillo REQUIRED)
find_package(Boost COMPONENTS log)

add_subdirectory(src)

link_libraries(MathLib)
link_libraries(${ARMADILLO_LIBRARIES})
link_libraries(Boost::log)

add_subdirectory(tests)
add_subdirectory(samples)
