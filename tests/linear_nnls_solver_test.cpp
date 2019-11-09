#define BOOST_TEST_MODULE nnls_unit_tests

#include <boost/test/included/unit_test.hpp>
#include <armadillo>
#include "../src/analytical/linear/ls_solver.h"
#include "../src/non-negative/greedy_nn_solver.cpp"
#include "test_utils.cpp"
#include "../src/logging/easylogging++.h"

INITIALIZE_EASYLOGGINGPP

using namespace arma;

mat signals = create_signals();
LSSolver solver = LSSolver(signals);
std::shared_ptr<LSSolver> ls_solver_ptr(new LSSolver(solver));
GreedyNNSolver nn_solver(ls_solver_ptr);

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
    BOOST_CHECK(is_equal(mat{0, 0, 0}, result));
}

BOOST_AUTO_TEST_CASE(solver_should_produce_same_result_with_different_constructor_methods)
{

    mat weights = {1, -1, 2};
    mat signal = sum_signal(weights, signals);

    LSSolver solver = LSSolver(signals);
    std::shared_ptr<LSSolver> ls_solver_ptr(new LSSolver(solver));
    GreedyNNSolver nn_solver1(ls_solver_ptr);

    GreedyNNSolver nn_solver2(signals);

    mat result1 = nn_solver1.solve(signal);
    mat result2 = nn_solver2.solve(signal);

    BOOST_CHECK(is_equal(result1, result2));
}