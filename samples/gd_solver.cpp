#include "data_generation.cpp"
#include "../src/numerical/gradient_descent/gd_linear_solver.h"
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

    GDLinearSolver gd_linear_solver = GDLinearSolver(L, 5000, 500);
    mat result = gd_linear_solver.solve(s);
    LOG(INFO) << "GD linear fit: " << result;

    return 0;
}