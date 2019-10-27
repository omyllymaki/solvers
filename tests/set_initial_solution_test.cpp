#define BOOST_TEST_MODULE ea_solver_unit_tests

#include <boost/test/included/unit_test.hpp>
#include <armadillo>
#include "../src/numerical/evolutionary_algorithm/ea_solver.h"
#include "test_utils.cpp"
#include "../src/logging/easylogging++.h"

INITIALIZE_EASYLOGGINGPP

using namespace arma;

mat signals = create_signals();
auto solver = EASolver(SIGNALS, 200, 200, 0.0001, 100, 50);

BOOST_AUTO_TEST_CASE(good_initial_guess_should_converge_quickly)
{
    mat weights = {1, 1, 2};
    mat guess = {0.5, 1.5, 2.1};
    solver.set_initial_guess(guess);
    mat signal = sum_signal(weights, signals);
    mat result = solver.solve(signal);
    BOOST_CHECK(is_equal(weights, result, 0.1));
}

BOOST_AUTO_TEST_CASE(bad_initial_guess_should_not_converge)
{
    mat weights = {1, 1, 2};
    mat guess = {-50000, 50000, 500};
    solver.set_initial_guess(guess);
    mat signal = sum_signal(weights, signals);
    mat result = solver.solve(signal);
    BOOST_CHECK(!is_equal(weights, result, 0.1));       // Note: !is_equal = not equal
}