#include <iostream>
#include <armadillo>
#include "common.h"

using namespace arma;
using std::tie;
using std::tuple;
using std::vector;

mat calculate_pseudoinverse(mat x)
{
    return x.t() * inv(x * x.t());
}

mat calculate_sum_signal(mat weigths, mat signals)
{
    /* 
    Calculates weighted sum signal.

    weights vector (1 x n_components) includes weighting factor for every individual signal.
    signals matrix (n_components x n_channels) contains individual signals as rows.
    returns (1 x n_channels) sum signal.
    */
    return weigths * signals;
}

tuple<mat, mat, mat> calculate_svd(mat x)
{
    /*
    Usage: auto [u,s,v] = calculate_svd(x)

    Original matrix can be reconstructed by u * diagmat(s) * v.t()
    */
    mat u;
    vec s;
    mat v;
    svd(u, s, v, x);
    return {u, s, v};
}

mat calculate_svd_inverse(mat x, int rank)
{
    /*
    Full rank or rank reduced inverse of matrix x.

    By default (rank = -1), full rank inverse is calculated.
    */

    // By default, use full rank (= smaller dimension of matrix x)
    if (rank == -1)
    {
        if (x.n_cols < x.n_rows)
        {
            rank = x.n_cols;
        }
        else
        {
            rank = x.n_rows;
        };
    }

    mat u, s, v;
    tie(u, s, v) = calculate_svd(x);
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
    mat u, s, v;
    tie(u, s, v) = calculate_svd(x);

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
    return sqrt(sum(pow(difference, 2), 1) / difference.n_elem);
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