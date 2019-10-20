#define BOOST_TEST_MODULE gd_linear_unit_tests

#include <boost/test/included/unit_test.hpp>
#include <armadillo>
#include "../src/gradient_descent/gd_linear_solver.h"
#include "test_utils.cpp"
#include "../src/logging/easylogging++.h"

INITIALIZE_EASYLOGGINGPP

GDLinearSolver solver(SIGNALS, 5, 5000);

BOOST_AUTO_TEST_CASE(test_common_cases)
{
    auto tester = SolverTester<GDLinearSolver>(solver);
    tester.test_common();
}