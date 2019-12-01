#include <iostream>
#include <armadillo>
#include <random>
#include "common.h"

using arma::mat;
using arma::svd;
using arma::vec;
using std::vector;

mat calculate_pseudoinverse(mat x)
{
    return x.t() * inv(x * x.t());
}

mat calculate_sum_signal(mat weigths, mat signals)
{
    return weigths * signals;
}

mat calculate_svd_inverse(mat x, int rank)
{
    // If rank = -1, use full rank (= smaller dimension of matrix x)
    if (rank == -1)
    {
        rank = std::min(x.n_cols, x.n_rows);
    }

    mat u, v;
    vec s;
    svd(u, s, v, x);
    mat s_inv = 1 / s;

    int lb = s_inv.n_elem - rank;
    int ub = s_inv.n_elem - 1;

    s_inv = s_inv.rows(lb, ub);
    v = v.cols(lb, ub);
    u = u.cols(lb, ub);
    mat sigma_inv = diagmat(s_inv);

    return v * sigma_inv * u.t();
}

mat low_rank_approximation(mat x, int rank)
{
    mat u, v;
    vec s;
    svd(u, s, v, x);

    s = s.rows(0, rank - 1);
    v = v.cols(0, rank - 1);
    u = u.cols(0, rank - 1);

    return u * diagmat(s) * v.t();
}

arma::mat trimmed_mean(arma::vec x, int n_points)
{
    int lb = n_points;
    int ub = x.n_elem - n_points - 1;
    arma::vec x_sorted = arma::sort(x, "ascend");
    return arma::mean(x_sorted.rows(lb, ub), 0);
}

arma::mat trimmed_mean(arma::vec x, float proportion)
{
    int n_points = round(proportion * x.n_elem);
    return trimmed_mean(x, n_points);
}

arma::mat rmse(arma::mat estimate_values, arma::mat true_values)
{
    arma::mat difference = estimate_values - true_values;
    arma::mat difference_pow2 = pow(difference, 2);
    double mean_difference_pow2 = arma::mean(arma::mean(difference_pow2));
    return {sqrt(mean_difference_pow2)};
}

arma::mat mae(arma::mat estimate_values, arma::mat true_values)
{
    arma::mat difference = estimate_values - true_values;
    return arma::sum(arma::abs(difference), 1) / difference.n_elem;
}

arma::mat trimmed_mae(arma::vec estimate_values, arma::vec true_values, double rejection_threshold)
{
    arma::vec difference = estimate_values - true_values;
    arma::vec abs_difference = arma::abs(difference);
    int n_points = round(rejection_threshold * abs_difference.n_elem);
    arma::vec abs_difference_sorted = arma::sort(abs_difference, "descent");
    return arma::mean(abs_difference_sorted.rows(n_points, abs_difference_sorted.n_elem - 1), 0);
}

std::vector<int> sample_without_replacement(int lb, int ub, int n)
{
    std::vector<int> vec;
    for (size_t i = lb; i <= ub; i++)
    {
        vec.push_back(i);
    }
    std::random_device device;
    std::mt19937 generator(device());
    std::shuffle(vec.begin(), vec.end(), generator);
    std::vector<int> out(vec.begin(), vec.begin() + n);
    return out;
}