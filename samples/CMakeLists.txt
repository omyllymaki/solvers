option(HAS_MATPLOTLIB_LIB "Plot some figures" OFF)
if(HAS_MATPLOTLIB_LIB)
    find_package(PythonLibs 2.7)
    link_libraries(${PYTHON_LIBRARIES})
    include_directories(PRIVATE ${PYTHON_INCLUDE_DIRS})
    add_definitions(-DPLOT_FIGURES)
endif(HAS_MATPLOTLIB_LIB)

add_executable(analysis_example analysis_example.cpp)
