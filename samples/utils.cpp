#include <iostream>
#include <bits/stdc++.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <string>
#include <math.h>
#include <armadillo>

using namespace arma;
using namespace std;
using stdvec = std::vector<double>;
using stdnestedvec = std::vector<std::vector<double>>;

template <typename T>
void print(T x, bool add_endl = true)
{
    if (add_endl)
    {
        cout << x << endl;
    }
    else
    {
        cout << x;
    }
}

void create_dir_if_not_exist(const char *dir_path)
{
    mkdir(dir_path, 0777);
}

const double EulerConstant = std::exp(1.0);

mat pow_to_vector(float x, mat y)
{
    mat result;
    result.copy_size(y);
    for (std::size_t i = 0; i < result.n_elem; ++i)
    {
        result[i] = std::pow(x, y[i]);
    }
    return result;
}

mat calculate_gaussian(float sigma, float center, mat x)
{
    float scaling_factor = 1 / (sigma * sqrt(2 * M_PI));
    mat exponent = -pow((x - center) / sigma, 2) / 2;
    mat result = scaling_factor * pow_to_vector(EulerConstant, exponent);
    return result;
}

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

#ifdef PLOT_FIGURES

#include "matplotlib-cpp/matplotlibcpp.h"
namespace plt = matplotlibcpp;

void plot_arma_vec(arma::mat x, long figure, string title = "")
{
    plt::figure(figure);
    plt::plot(arma_vec_to_std_vector(x));
    plt::title(title);
}

void plot_arma_mat(arma::mat x, long figure, string title = "")
{
    plt::figure(figure);
    for (auto &&row : arma_mat_to_std_vec(x))
    {
        plt::plot(row);
    }
    plt::title(title);
}

#endif