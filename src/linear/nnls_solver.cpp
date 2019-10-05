#include "nnls_solver.h"
#include "../common.h"
#include <iostream>
#include <vector>

using arma::mat;
using arma::zeros;
using std::vector;

NNLSSolver::NNLSSolver(const arma::mat &L)
{
    m_L = L;
}

arma::mat NNLSSolver::solve(const arma::mat &s)
{
    mat L = m_L;
    vector<int> indices;
    mat result;
    while (true)
    {

        result = s * calculate_pseudoinverse(L);

        float min_value = result.min();

        // Break loop if all analysed values are non-negative
        if (min_value >= 0)
        {
            break;
        }
        // If any of the analysed values is negative, remove most negative from fit
        else
        {
            int min_index = result.index_min();
            L.shed_row(min_index);
            indices.push_back(min_index);
        }

        // If there are no more component to fit, return zero vector (all component are negative)
        if (L.n_rows == 0)
        {
            mat result = zeros(indices.size()).t();
            return result;
        }
    }

    // For every removed component (negative analysis result), add zero as result
    for (int i = indices.size() - 1; i >= 0; i--)
    {
        result.insert_cols(indices[i], 1);
    }

    m_x = result;
    return m_x;
}

arma::mat NNLSSolver::model(mat x, mat L)
{
    return x * L;
}