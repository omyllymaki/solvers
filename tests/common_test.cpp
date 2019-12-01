#define BOOST_TEST_MODULE common_unit_tests

#include <boost/test/included/unit_test.hpp>
#include <armadillo>
#include "../src/common.h"
#include "test_utils.cpp"
#include "../src/logging/easylogging++.h"

INITIALIZE_EASYLOGGINGPP

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

BOOST_AUTO_TEST_CASE(low_rank_approximation_with_non_square_random_matrix)
{
    mat x = randu<mat>(10, 12);
    mat x_low = low_rank_approximation(x, 9);
    // actual is only approximately equal to expected due to low rank approximation
    // -> use larger threshold for this test case
    BOOST_CHECK(is_equal(x, x_low, 0.1));
}

BOOST_AUTO_TEST_CASE(trimmed_mean_integer_argument)
{
    arma::vec x;
    double result;

    x = {1, 2, 5};
    result = arma::as_scalar(trimmed_mean(x, 1));
    BOOST_CHECK_EQUAL(result, 2);

    x = {5, 2, -1, 2};
    result = arma::as_scalar(trimmed_mean(x, 1));
    BOOST_CHECK_EQUAL(result, 2);

    x = {0, 10, 2, 12, 1};
    result = arma::as_scalar(trimmed_mean(x, 2));
    BOOST_CHECK_EQUAL(result, 2);
}

BOOST_AUTO_TEST_CASE(trimmed_mean_float_argument)
{
    arma::vec x;
    double result;

    x = {1, 2, 2, 5};
    result = arma::as_scalar(trimmed_mean(x, 0.2f)); // rejects 1 point from both side
    BOOST_CHECK_EQUAL(result, 2);

    x = {-1, 3, 5, 2, 7};
    result = arma::as_scalar(trimmed_mean(x, 0.4f)); // rejects 2 points from both side
    BOOST_CHECK_EQUAL(result, 3);
}

BOOST_AUTO_TEST_CASE(rmse_1d_matrix_arguments)
{
    arma::mat x1, x2, result;

    x1 = {1, 2, 2, 5};
    x2 = {1, 1, 2, 5};
    result = rmse(x1, x2);
    BOOST_CHECK_EQUAL(as_scalar(result), 0.5);

    x1 = {1, 2, 2, 5};
    x2 = {1, 2, 2, 5};
    result = rmse(x1, x2);
    BOOST_CHECK_EQUAL(as_scalar(result), 0);
}

BOOST_AUTO_TEST_CASE(rmse_2d_matrix_arguments)
{
    arma::mat x1, x2, result;

    x1 = {{0, 0, 0},
          {0, 0, 0},
          {0, 0, 0}};
    x2 = {{0, 0, 0},
          {0, 3, 0},
          {0, 0, 0}};
    result = rmse(x1, x2);
    BOOST_CHECK_EQUAL(as_scalar(result), 1);
}