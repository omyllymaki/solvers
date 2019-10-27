#include "data_generation.cpp"
#include "../src/numerical/gauss-newton/gn_solver.h"
#include "../src/non-negative/greedy_nn_solver.cpp"
#include "../src/logging/easylogging++.h"
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
    auto s_quadratic = data_generator.generate_quadratic_signal(WEIGHTS);

    LOG(INFO) << "True: " << WEIGHTS;

    GNSolver gn_solver_quadratic = GNSolver(L);
    gn_solver_quadratic.set_model(quadratic_model);
    arma::mat result1 = gn_solver_quadratic.solve(s_quadratic);
    LOG(INFO) << "GN quadratic fit: " << result1;

    auto nn_solver = GreedyNNSolver<GNSolver>(gn_solver_quadratic);
    arma::mat result2 = nn_solver.solve(s_quadratic);
    LOG(INFO) << "Non-negative GN quadratic fit: " << result2;

    return 0;
}