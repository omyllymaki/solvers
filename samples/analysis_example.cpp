#include "utils.cpp"
#include "../src/linear/ls_solver.h"
#include "../src/non-negative/nnls_solver.h"
#include "../src/gradient_descent/gd_linear_solver.h"
#include "../src/evolutionary_algorithm/ea_solver.h"
#include "../src/evolutionary_algorithm/robust_ea_solver.h"
#include "../src/gauss-newton/gn_solver.h"
#include "../src/logging/easylogging++.h"
#include "../src/non-negative/nngn_solver.cpp"
#include <math.h>
#include <armadillo>

using arma::mat;

INITIALIZE_EASYLOGGINGPP

mat WEIGHTS = {100, -20, 50, -0.5};
mat CENTERS = {20.0, 35.0, 40.0, 45.0};
mat SIGMAS = {3.0, 10.0, 5.0, 2.0};
mat CHANNELS = linspace(0, 99, 100);
double NOISE = 0.05;

arma::mat quadratic_model(arma::mat x, arma::mat L)
{
    return x * arma::pow(L, 2);
}

mat generate_signal_matrix(mat channels, mat centers, mat sigmas)
{
    int n_components = centers.n_elem;
    mat signals = zeros(0, channels.n_elem);
    for (int i = 0; i < n_components; ++i)
    {
        float center = centers[i];
        float sigma = sigmas[i];
        mat signal = calculate_gaussian(sigma, center, channels);
        signals = join_vert(signals, signal.t());
    };
    return signals;
}

int main(int argc, char *argv[])
{
    el::Configurations conf("./logging-config.conf");
    el::Loggers::reconfigureLogger("default", conf);

    mat L = generate_signal_matrix(CHANNELS, CENTERS, SIGMAS);

    // Generate different type of signals for testing
    mat s = WEIGHTS * L;
    mat s_quadratic = quadratic_model(WEIGHTS, L);
    mat s_noisy = s + NOISE * arma::randn(1, s.n_elem);
    mat s_with_outliers = s;
    s_with_outliers[10] += 10;
    s_with_outliers[20] -= 10;

    LOG(INFO) << "True: " << WEIGHTS;

    LSSolver ls_solver = LSSolver(L);
    mat result1 = ls_solver.solve(s);
    LOG(INFO) << "LS fit: " << result1;

    NNLSSolver nnls_solver = NNLSSolver(L);
    mat result2 = nnls_solver.solve(s);
    LOG(INFO) << "NNLS fit: " << result2;

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

    NNGNSolver nngn_solver = NNGNSolver(L);
    nngn_solver.set_model(quadratic_model);
    mat result8 = nngn_solver.solve(s_quadratic);
    LOG(INFO) << "NN quadratic fit: " << result8;

#ifdef PLOT_FIGURES
    plot_arma_mat(L, 1, "Pure components");

    plot_arma_vec(s_with_outliers, 2);
    plot_arma_vec(robust_ea_solver.get_signal_estimate(), 2);
    plt::show();
#endif

    return 0;
}