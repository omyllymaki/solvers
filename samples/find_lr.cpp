#include "utils/data_generator.cpp"
#include "../src/numerical/gradient_descent/gd_solver.h"
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

    GDSolver solver = GDSolver(L, 1, 1000);
    solver.find_optimal_lr(s);
    auto result = solver.solve(s);
    LOG(INFO) << "Result: " << result;

    return 0;
}