#define BOOST_TEST_MODULE ea_solver_unit_tests

#include <boost/test/included/unit_test.hpp>
#include <armadillo>
#include "../src/numerical/evolutionary_algorithm/ea_solver.h"
#include "test_utils.cpp"
#include "../src/logging/easylogging++.h"

INITIALIZE_EASYLOGGINGPP

using namespace arma;

auto solver = EASolver(SIGNALS, 500, 1000, 0.0001, 100, 50);

BOOST_AUTO_TEST_CASE(test_common_cases)
{
    auto tester = SolverTester<EASolver>(solver);
    tester.test_common(0.1);
}