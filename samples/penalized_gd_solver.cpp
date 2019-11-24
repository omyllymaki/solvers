#include "utils/data_generator.cpp"
#include "../src/numerical/gradient_descent/penalized_gd_solver.h"
#include "../src/logging/easylogging++.h"
#include <armadillo>

using arma::mat;

INITIALIZE_EASYLOGGINGPP

mat WEIGHTS = {100, -20, 50, -0.5};

int main(int argc, char *argv[])
{
    el::Configurations conf("./logging-config.conf");
    el::Loggers::reconfigureLogger("default", conf);

    auto data_generator = DataGenerator();
    auto L = data_generator.generate_library();
    auto s = data_generator.generate_signal(WEIGHTS);

    LOG(INFO) << "True: " << WEIGHTS;

    PenalizedGDSolver solver = PenalizedGDSolver(L, 100, 500, 0.0000000001, 1000000);
    mat result = solver.solve(s);
    LOG(INFO) << "Penalized GD solver fit: " << result;

    return 0;
}