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