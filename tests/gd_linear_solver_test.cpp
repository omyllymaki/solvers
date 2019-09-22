#define BOOST_TEST_MODULE nnls_unit_tests

#include <boost/test/included/unit_test.hpp>
#include <armadillo>
#include "../src/gd_linear_solver.h"
#include "test_utils.cpp"

using namespace arma;

mat signals = create_signals();
auto solver = GDLinearSolver(signals, 1.0, 10000, 0.00001);

BOOST_AUTO_TEST_CASE(gd_fit_positive_values)
{
    mat weights = {1, 1, 2};
    mat signal = sum_signal(weights, signals);
    mat result = solver.solve(signal);
    BOOST_CHECK(is_equal(weights, result));
}

BOOST_AUTO_TEST_CASE(gd_fit_zero_values)
{
    mat weights = {0, 0, 0};
    mat signal = sum_signal(weights, signals);
    mat result = solver.solve(signal);
    BOOST_CHECK(is_equal(weights, result));
}

BOOST_AUTO_TEST_CASE(gd_fit_positive_and_negative_values)
{
    mat weights = {-1, 1, 2};
    mat signal = sum_signal(weights, signals);
    mat result = solver.solve(signal);
    BOOST_CHECK(is_equal(weights, result));

    weights = {1, -1, 2};
    signal = sum_signal(weights, signals);
    result = solver.solve(signal);
    BOOST_CHECK(is_equal(weights, result));
}

BOOST_AUTO_TEST_CASE(gd_fit_very_small_and_large_values)
{
    mat weights = {0.01, 1, 50000};
    mat signal = sum_signal(weights, signals);
    mat result = solver.solve(signal);
    cout << result << endl;
    BOOST_CHECK(is_equal(weights, result));
}