#include <iostream>
#include <armadillo>
#include "ls.h"
#include "nnls.h"


using namespace arma;
using std::vector;


mat nnls_fit(mat L, mat s)
{
    /* 
    Finds best linear least squares fit to system of linear equations: x*L = s with constaint xi >= 0 
    
    Returns fitted x values as solution.
    */
    vector<int> indices;
    mat result;
    while (true)
    {

        result = ls_fit(L, s);

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

    return result;
}