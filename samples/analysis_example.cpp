#include "plotting.cpp"
#include "data_generation.cpp"
#include "../src/linear/ls_solver.h"
#include "../src/gradient_descent/gd_linear_solver.h"
#include "../src/evolutionary_algorithm/ea_solver.h"
#include "../src/evolutionary_algorithm/robust_ea_solver.h"
#include "../src/gauss-newton/gn_solver.h"
#include "../src/logging/easylogging++.h"
#include <math.h>
#include <armadillo>

using arma::mat;

INITIALIZE_EASYLOGGINGPP

mat WEIGHTS = {100, -20, 50, -0.5};

arma::mat quadratic_model(arma::mat x, arma::mat L)
{
    return x * arma::pow(L, 2);
}

int main(int argc, char *argv[])
{
    el::Configurations conf("./logging-config.conf");
    el::Loggers::reconfigureLogger("default", conf);

    auto data_generator = DataGenerator();
    auto L = data_generator.generate_library();
    auto s = data_generator.generate_linear_signal(WEIGHTS);
    auto s_with_outliers = data_generator.generate_linear_signal_with_outliers(WEIGHTS);
    auto s_quadratic = data_generator.generate_quadratic_signal(WEIGHTS);

    LOG(INFO) << "True: " << WEIGHTS;

    LSSolver ls_solver = LSSolver(L);
    mat result1 = ls_solver.solve(s);
    LOG(INFO) << "Linear LS fit: " << result1;

    mat result2 = ls_solver.nn_solve(s);
    LOG(INFO) << "Linear NNLS fit: " << result2;

    GDLinearSolver gd_linear_solver = GDLinearSolver(L, 5000, 500);
    mat result3 = gd_linear_solver.solve(s);
    LOG(INFO) << "GD linear fit: " << result3;

    EASolver ea_solver = EASolver(L);
    ea_solver.set_initial_guess(arma::mat{50, -10, 5, 0});
    mat result4 = ea_solver.solve(s);
    LOG(INFO) << "EA fit: " << result4;

    RobustEASolver robust_ea_solver = RobustEASolver(L);
    mat result5 = robust_ea_solver.solve(s_with_outliers);
    LOG(INFO) << "Robust EA fit: " << result5;

    GNSolver gn_solver = GNSolver(L);
    arma::mat result6 = gn_solver.solve(s);
    LOG(INFO) << "GN fit: " << result6;

    GNSolver gn_solver_quadratic = GNSolver(L);
    gn_solver_quadratic.set_model(quadratic_model);
    arma::mat result7 = gn_solver_quadratic.solve(s_quadratic);
    LOG(INFO) << "GN quadratic fit: " << result7;

    mat result8 = gn_solver_quadratic.nn_solve(s_quadratic);
    LOG(INFO) << "NNGN quadratic fit: " << result8;

#ifdef PLOT_FIGURES
    plot_arma_mat(L, 1, "Pure components");

    plot_arma_vec(s_with_outliers, 2);
    plot_arma_vec(robust_ea_solver.get_signal_estimate(), 2);
    plt::show();
#endif

    return 0;
}