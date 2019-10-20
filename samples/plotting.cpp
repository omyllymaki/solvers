#ifdef PLOT_FIGURES

#include <math.h>
#include <armadillo>
#include "contrib/matplotlibcpp/matplotlibcpp.h"

namespace plt = matplotlibcpp;
using namespace arma;
using namespace std;
using stdvec = std::vector<double>;
using stdnestedvec = std::vector<std::vector<double>>;

stdvec arma_vec_to_std_vector(arma::mat x)
{
    return arma::conv_to<stdvec>::from(x);
}

stdnestedvec arma_mat_to_std_vec(arma::mat &A)
{
    stdnestedvec V(A.n_rows);
    for (size_t i = 0; i < A.n_rows; ++i)
    {
        V[i] = arma::conv_to<stdvec>::from(A.row(i));
    };
    return V;
};

void plot_arma_vec(arma::mat x, long figure = 1, string title = "")
{
    plt::figure(figure);
    plt::plot(arma_vec_to_std_vector(x));
    plt::title(title);
}

void plot_arma_mat(arma::mat x, long figure = 1, string title = "")
{
    plt::figure(figure);
    for (auto &&row : arma_mat_to_std_vec(x))
    {
        plt::plot(row);
    }
    plt::title(title);
}

#endif