#define BOOST_TEST_MODULE nnls_unit_tests

#include <boost/test/included/unit_test.hpp>
#include <armadillo>
#include "../src/linear/ls_solver.h"
#include "../src/non-negative/greedy_nn_solver.cpp"
#include "test_utils.cpp"
#include "../src/logging/easylogging++.h"

INITIALIZE_EASYLOGGINGPP

using namespace arma;

mat signals = create_signals();
LSSolver solver = LSSolver(signals);
auto nn_solver = GreedyNNSolver<LSSolver>(solver);

BOOST_AUTO_TEST_CASE(fit_positive_values)
{
    mat weights = {1, 1, 2};
    mat signal = sum_signal(weights, signals);
    mat result = nn_solver.solve(signal);
    BOOST_CHECK(is_equal(weights, result));
}

BOOST_AUTO_TEST_CASE(fit_zero_values)
{
    mat weights = {0, 0, 0};
    mat signal = sum_signal(weights, signals);
    mat result = nn_solver.solve(signal);
    BOOST_CHECK(is_equal(weights, result));
}

BOOST_AUTO_TEST_CASE(fit_positive_and_negative_values)
{
    mat weights = {-1, 1, 2};
    mat signal = sum_signal(weights, signals);
    mat result = nn_solver.solve(signal);
    BOOST_CHECK(result[0] == 0);
    BOOST_CHECK(result[1] > 0);
    BOOST_CHECK(result[2] > 0);

    weights = {1, -1, 2};
    signal = sum_signal(weights, signals);
    result = nn_solver.solve(signal);
    BOOST_CHECK(result[0] > 0);
    BOOST_CHECK(result[1] == 0);
    BOOST_CHECK(result[2] > 0);
}

BOOST_AUTO_TEST_CASE(fit_only_negative_values)
{
    mat weights = {-1, -1, -2};
    mat signal = sum_signal(weights, signals);
    mat result = nn_solver.solve(signal);
    BOOST_CHECK(is_equal(mat {0, 0, 0}, result));
}