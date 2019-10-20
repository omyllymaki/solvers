#define BOOST_TEST_MODULE ls_solver_unit_tests

#include <boost/test/included/unit_test.hpp>
#include <armadillo>
#include "../src/linear/ls_solver.h"
#include "test_utils.cpp"
#include "../src/logging/easylogging++.h"

INITIALIZE_EASYLOGGINGPP

BOOST_FIXTURE_TEST_SUITE(s, TestFixture<LSSolver>);

BOOST_AUTO_TEST_CASE(fit_only_positive_values)
{
    mat weights = {1, 1, 2};
    test_solve(weights);
}

BOOST_AUTO_TEST_CASE(fit_zero_values)
{
    mat weights = {0, 0, 0};
    test_solve(weights);
}

BOOST_AUTO_TEST_CASE(fit_positive_and_negative_values)
{
    mat weights = {-1, 1, 2};
    test_solve(weights);
}

BOOST_AUTO_TEST_CASE(fit_only_negative_values)
{
    mat weights = {-1, -1, -2};
    test_solve(weights);
}

BOOST_AUTO_TEST_SUITE_END()