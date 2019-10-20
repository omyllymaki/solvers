#define BOOST_TEST_MODULE ls_solver_unit_tests

#include <boost/test/included/unit_test.hpp>
#include <armadillo>
#include "../src/linear/ls_solver.h"
#include "test_utils.cpp"
#include "../src/logging/easylogging++.h"

INITIALIZE_EASYLOGGINGPP

LSSolver solver;

BOOST_AUTO_TEST_CASE(test_common_cases)
{
    auto tester = SolverTester<LSSolver>(solver);
    tester.test_common();
}
