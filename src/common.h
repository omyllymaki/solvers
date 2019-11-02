#ifndef COMMON
#define COMMON

#include <iostream>
#include <armadillo>

//! Moore-Penrose inverse
arma::mat calculate_pseudoinverse(arma::mat x);

//! Weighted sum signal
arma::mat calculate_sum_signal(arma::mat weigths, arma::mat signals);

//! Full rank or rank reduced inverse of matrix x
//! If rank = -1, full rank inverse is calculated.
arma::mat calculate_svd_inverse(arma::mat x, int rank=-1);

//! Low rank approximation of matrix, using SVD composition
arma::mat low_rank_approximation(arma::mat x, int rank);

// Trimmed mean, rejecting largest and smallest values
arma::mat trimmed_mean(arma::vec x, int n_points);

// Trimmed mean, rejecting largest and smallest values
arma::mat trimmed_mean(arma::vec x, float proportion);

//! Root-mean-square error
arma::mat rmse(arma::mat estimate_values, arma::mat true_values);

//! Mean absolute error
arma::mat mae(arma::mat estimate_values, arma::mat true_values);

//! Trimmed mean absolute error, rejecting largest differences between estimate and expected
arma::mat trimmed_mae(arma::vec estimate, arma::vec expected, double rejection_threshold);

//! Sample n samples without replacement from array of integeres [lb, lb + 1, ..., ub]
std::vector<int> sample_without_replacement(int lb, int ub, int n);

#endif