add_executable(ls_solver_test.exe ls_solver_test.cpp)
add_executable(linear_nnls_solver_test.exe linear_nnls_solver_test.cpp)
add_executable(common_test.exe common_test.cpp)
add_executable(ea_solver_test.exe ea_solver_test.cpp)
add_executable(robust_ea_solver_test.exe robust_ea_solver_test.cpp)
add_executable(gn_solver_test.exe gn_solver_test.cpp)
add_executable(set_initial_solution_test set_initial_solution_test.cpp)
add_executable(ransac_solver_test ransac_solver_test.cpp)
add_executable(gd_solver_test gd_solver_test.cpp)

include(CTest)
add_test(ls_solver_test.exe ls_solver_test.exe)
add_test(linear_nnls_solver_test.exe linear_nnls_solver_test.exe)
add_test(common_test.exe common_test.exe)
add_test(ea_solver_test.exe ea_solver_test.exe)
add_test(robust_ea_solver_test.exe robust_ea_solver_test.exe)
add_test(gn_solver_test.exe gn_solver_test.exe)
add_test(set_initial_solution_test set_initial_solution_test)
add_test(ransac_solver_test ransac_solver_test)
add_test(gd_solver_test gd_solver_test)