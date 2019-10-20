#define BOOST_TEST_MODULE robust_ea_solver_unit_tests

#include <boost/test/included/unit_test.hpp>
#include <armadillo>
#include "../src/evolutionary_algorithm/robust_ea_solver.h"
#include "test_utils.cpp"
#include "../src/logging/easylogging++.h"

INITIALIZE_EASYLOGGINGPP

using namespace arma;

RobustEASolver solver = RobustEASolver(SIGNALS, 500, 1000, 0.0001, 100, 100);

BOOST_AUTO_TEST_CASE(test_common_cases)
{
    auto tester = SolverTester<RobustEASolver>(solver);
    tester.test_common(0.1);
}

BOOST_AUTO_TEST_CASE(fit_random_signals_with_outliers)
{
    mat weights = {-10, 1, 500};
    mat signals = arma::randu(3, 100);
    RobustEASolver solver = RobustEASolver(signals, 500, 1000, 0.0001, 100, 100);
    mat signal = sum_signal(weights, signals);
    signal[10] -= 1000;
    signal[20] += 1000;
    mat result = solver.solve(signal);
    BOOST_CHECK(is_equal(weights, result, 0.1));
}