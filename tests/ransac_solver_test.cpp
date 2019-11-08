#define BOOST_TEST_MODULE ransac_solver_unit_tests

#include <boost/test/included/unit_test.hpp>
#include <armadillo>
#include "../src/analytical/linear/ls_solver.h"
#include "../src/robust/ransac_solver.cpp"
#include "test_utils.cpp"
#include "../src/logging/easylogging++.h"

INITIALIZE_EASYLOGGINGPP

using namespace arma;

BOOST_AUTO_TEST_CASE(fit_random_signals_with_outliers)
{
    mat weights = {-10, 1, 500};
    mat signals = arma::randu(3, 100);
    mat signal = sum_signal(weights, signals);

    // Add outliers to signal
    std::vector<int> outlier_indices = {1, 5, 10, 35, 67, 85, 98};
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, 10);
    for (auto &&i : outlier_indices)
    {
        signal[i] += distribution(generator);
    }

    LSSolver solver = LSSolver(signals);
    std::shared_ptr<LSSolver> solver_ptr(new LSSolver(solver));
    auto ransac_solver = RansacSolver(solver_ptr, 5, 0.1, 80);

    // With regular non-robust solver, we shouldn't get corrent solution
    mat result = solver.solve(signal);
    BOOST_CHECK(!is_equal(weights, result));

    // RANSAC solver should handle outliers
    mat result_ransac = ransac_solver.solve(signal);
    BOOST_CHECK(is_equal(weights, result_ransac));
}