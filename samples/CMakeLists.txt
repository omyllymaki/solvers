option(HAS_MATPLOTLIB_LIB "Plot some figures" OFF)
if(HAS_MATPLOTLIB_LIB)
    find_package(PythonLibs 2.7)
    link_libraries(${PYTHON_LIBRARIES})
    include_directories(PRIVATE ${PYTHON_INCLUDE_DIRS})
    add_definitions(-DPLOT_FIGURES)
endif(HAS_MATPLOTLIB_LIB)

add_executable(find_lr find_lr.cpp)
add_executable(ls_solver ls_solver.cpp)
add_executable(gd_solver gd_solver.cpp)
add_executable(ea_solver ea_solver.cpp)
add_executable(gn_solver gn_solver.cpp)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/logging-config.conf
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/run_samples.sh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
