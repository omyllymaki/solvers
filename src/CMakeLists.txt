ADD_LIBRARY(MathLib STATIC
            common.cpp
            solver.cpp
            analytical/linear/ls_solver.cpp
            non-negative/greedy_nn_solver.cpp
            numerical/numerical_solver.cpp
            numerical/gradient_descent/gd_solver.cpp
            numerical/gradient_descent/gd_linear_solver.cpp
            numerical/evolutionary_algorithm/ea_solver.cpp
            numerical/evolutionary_algorithm/robust_ea_solver.cpp
            numerical/gauss-newton/gn_solver.cpp
            logging/easylogging++.cc)