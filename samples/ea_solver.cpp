#include "utils/data_generator.cpp"
#include "../src/numerical/evolutionary_algorithm/ea_solver.h"
#include "../src/numerical/evolutionary_algorithm/robust_ea_solver.h"
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
    auto s = data_generator.generate_signal(WEIGHTS);
    std::vector<int> outlier_channels = {5, 10};
    auto s_with_outliers = data_generator.generate_signal(WEIGHTS, outlier_channels);

    LOG(INFO) << "True: " << WEIGHTS;

    EASolver ea_solver = EASolver(L);
    ea_solver.set_initial_guess(arma::mat{50, -10, 5, 0});
    mat result1 = ea_solver.solve(s);
    LOG(INFO) << "EA fit: " << result1;

    RobustEASolver robust_ea_solver = RobustEASolver(L);
    mat result2 = robust_ea_solver.solve(s_with_outliers);
    LOG(INFO) << "Robust EA fit: " << result2;

    return 0;
}