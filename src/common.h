#ifndef COMMON
#define COMMON

#include <iostream>
#include <armadillo>

arma::mat calculate_pseudoinverse(arma::mat x);

arma::mat calculate_sum_signal(arma::mat weigths, arma::mat signals);

std::tuple<arma::mat, arma::mat, arma::mat> calculate_svd(arma::mat x);

arma::mat calculate_svd_inverse(arma::mat x, int rank=-1);

arma::mat low_rank_approximation(arma::mat x, int rank);

arma::mat trimmed_mean(arma::vec x, int n_points);

arma::mat trimmed_mean(arma::vec x, float proportion);

arma::mat rmse(arma::mat estimate_values, arma::mat true_values);

arma::mat mae(arma::mat estimate_values, arma::mat true_values);

arma::mat trimmed_mae(arma::vec estimate, arma::vec expected, double rejection_threshold);

#endif