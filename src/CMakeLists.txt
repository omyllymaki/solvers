ADD_LIBRARY(MathLib STATIC
            common.cpp
            solver.cpp
            linear/ls_solver.cpp
            non-negative/nnls_solver.cpp
            non-negative/nngn_solver.cpp
            gradient_descent/gd_solver.cpp
            gradient_descent/gd_linear_solver.cpp
            evolutionary_algorithm/ea_solver.cpp
            evolutionary_algorithm/robust_ea_solver.cpp
            gauss-newton/gn_solver.cpp
            logging/easylogging++.cc)