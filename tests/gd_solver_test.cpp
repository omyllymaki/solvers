#define BOOST_TEST_MODULE gd_solver_unit_tests

#include <boost/test/included/unit_test.hpp>
#include <armadillo>
#include "../src/numerical/gradient_descent/gd_solver.h"
#include "test_utils.cpp"
#include "../src/logging/easylogging++.h"

INITIALIZE_EASYLOGGINGPP

GDSolver solver(SIGNALS, 0.5, 5000);

BOOST_AUTO_TEST_CASE(test_common_cases)
{
    auto tester = SolverTester<GDSolver>(solver);
    tester.test_common();
}