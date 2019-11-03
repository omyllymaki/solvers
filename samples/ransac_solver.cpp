#include "utils/plotting.cpp"
#include "utils/data_generator.cpp"
#include "../src/analytical/linear/ls_solver.h"
#include "../src/numerical/gauss-newton/gn_solver.h"
#include "../src/numerical/gradient_descent/gd_solver.h"
#include "../src/robust/ransac_solver.cpp"
#include "../src/logging/easylogging++.h"
#include <armadillo>

using std::cout;
using std::endl;

INITIALIZE_EASYLOGGINGPP

arma::mat WEIGHTS = {100, -20, 50, -0.5};

int main(int argc, char *argv[])
{
    el::Configurations conf("./logging-config.conf");
    el::Loggers::reconfigureLogger("default", conf);

    auto data_generator = DataGenerator();
    auto L = data_generator.generate_library();
    std::vector<int> outlier_indices = {0, 5, 11, 13, 15, 18, 20, 25, 40, 60};
    auto s = data_generator.generate_signal(WEIGHTS, outlier_indices);

    LOG(INFO) << "True: " << WEIGHTS;

    auto solver = GDSolver(L);
    arma::mat result = solver.solve(s);
    LOG(INFO) << "Regular fit: " << result;

    int n_channels = 4;
    int n_max_iter = 1000;
    float accepted_error = 0.1;
    int n_accepted_points = 70;
    float objective_value_threshold = 0.0001;
    auto ransac_solver = RansacSolver<GDSolver>(solver, n_channels, accepted_error, n_accepted_points, objective_value_threshold, n_max_iter);
    auto solution = ransac_solver.solve(s);
    LOG(INFO) << "RANSAC fit: " << solution;

    auto estimate = ransac_solver.get_signal_estimate();

#ifdef PLOT_FIGURES
    plot_arma_vec(s);
    plot_arma_vec(estimate);
    plt::show();
#endif

    return 0;
}