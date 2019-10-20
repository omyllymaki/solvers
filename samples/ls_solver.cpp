#include "plotting.cpp"
#include "data_generation.cpp"
#include "../src/linear/ls_solver.h"
#include "../src/logging/easylogging++.h"
#include <math.h>
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
    auto s = data_generator.generate_linear_signal(WEIGHTS);

    LOG(INFO) << "True: " << WEIGHTS;

    LSSolver ls_solver = LSSolver(L);
    mat result1 = ls_solver.solve(s);
    LOG(INFO) << "Linear LS fit: " << result1;

    mat result2 = ls_solver.nn_solve(s);
    LOG(INFO) << "Linear NNLS fit: " << result2;

    return 0;
}