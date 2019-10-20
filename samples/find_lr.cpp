#include "utils.cpp"
#include "../src/gradient_descent/gd_solver.h"
#include "../src/logging/easylogging++.h"
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
    mat s = WEIGHTS * L;
   
    LOG(INFO) << "True: " << WEIGHTS;

    GDSolver solver = GDSolver(L, 1, 1000);
    solver.find_optimal_lr(s);
    auto result = solver.solve(s);
    LOG(INFO) << "Result: " << result;

    return 0;
}