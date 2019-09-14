#ifndef COMMON
#define COMMON

#include <iostream>
#include <armadillo>

using namespace arma;
using std::tuple;

mat calculate_pseudoinverse(mat x);

mat calculate_sum_signal(mat weigths, mat signals);

tuple<mat, mat, mat> calculate_svd(mat x);

mat calculate_svd_inverse(mat x, int rank=-1);

#endif