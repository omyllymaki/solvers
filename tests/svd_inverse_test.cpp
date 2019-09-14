#define BOOST_TEST_MODULE nnls_unit_tests

#include <boost/test/included/unit_test.hpp>
#include <armadillo>
#include "../src/common.h"
#include "test_utils.cpp"

using namespace arma;

BOOST_AUTO_TEST_CASE(full_rank_inv_with_random_square_matrix)
{
    mat x = randu<mat>(5, 5);
    mat expected = calculate_pseudoinverse(x);
    mat actual = calculate_svd_inverse(x);
    BOOST_CHECK(is_equal(expected, actual));
}

BOOST_AUTO_TEST_CASE(full_rank_inv_with_rank_argument_with_random_square_matrix)
{
    mat x = randu<mat>(5, 5);
    mat expected = calculate_pseudoinverse(x);
    mat actual = calculate_svd_inverse(x, 5);
    BOOST_CHECK(is_equal(expected, actual));
}

BOOST_AUTO_TEST_CASE(full_rank_inv_with_rank_argument_with_random_non_square_matrix)
{
    mat x = randu<mat>(5, 6);
    mat expected = calculate_pseudoinverse(x);
    mat actual = calculate_svd_inverse(x, 5);
    BOOST_CHECK(is_equal(expected, actual));
}

BOOST_AUTO_TEST_CASE(low_rank_inv_with_random_square_matrix)
{
    mat x = randu<mat>(10, 10);
    mat expected = calculate_pseudoinverse(x);
    mat actual = calculate_svd_inverse(x, 9);
    // actual is only approximately equal to expected due to low rank approximation
    // -> use larger threshold for this test case
    BOOST_CHECK(is_equal(expected, actual, 0.1));        
}

BOOST_AUTO_TEST_CASE(low_rank_inv_with_non_square_random_matrix)
{
    mat x = randu<mat>(10, 12);
    mat expected = calculate_pseudoinverse(x);
    mat actual = calculate_svd_inverse(x, 9);
    // actual is only approximately equal to expected due to low rank approximation
    // -> use larger threshold for this test case
    BOOST_CHECK(is_equal(expected, actual, 0.1));        
}