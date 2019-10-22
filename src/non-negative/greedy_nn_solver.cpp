#include "greedy_nn_solver.h"
#include "../common.h"
#include "../logging/easylogging++.h"
#include <iostream>
#include <vector>

using arma::mat;
using arma::zeros;
using std::vector;

template <typename T>
GreedyNNSolver<T>::GreedyNNSolver(T solver)
{
    m_solver = solver;
}

template <typename T>
arma::mat GreedyNNSolver<T>::solve(const arma::mat &s)
{
    auto solver = m_solver;
    arma::mat L = solver.get_library();
    std::vector<int> indices;
    arma::mat result;
    while (true)
    {
        solver.set_library(L);
        result = solver.solve(s);
        LOG(DEBUG) << "Result: " << result;

        float min_value = result.min();
        LOG(DEBUG) << "Min value: " << min_value;

        // Break loop if all analysed values are non-negative
        if (min_value >= 0)
        {
            LOG(DEBUG) << "All analysed values are positive";
            break;
        }
        // If any of the analysed values is negative, remove most negative from fit
        else
        {
            int min_index = result.index_min();
            L.shed_row(min_index);
            indices.push_back(min_index);
            LOG(DEBUG) << "Remove component " << min_index << " from fit";
        }

        // If there are no more component to fit, return zero vector (all component are negative)
        if (L.n_rows == 0)
        {
            LOG(DEBUG) << "No more components to fit";
            arma::mat result = arma::zeros(indices.size()).t();
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