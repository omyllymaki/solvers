#define BOOST_TEST_MODULE gn_solver_unit_tests

#include <boost/test/included/unit_test.hpp>
#include <armadillo>
#include "../src/numerical/gauss-newton/gn_solver.h"
#include "test_utils.cpp"
#include "../src/logging/easylogging++.h"

INITIALIZE_EASYLOGGINGPP

GNSolver solver(SIGNALS, 100, 0.00001);

BOOST_AUTO_TEST_CASE(test_common_cases)
{
    auto tester = SolverTester<GNSolver>(solver);
    tester.test_common();
}