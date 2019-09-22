#include "gd_linear_solver.h"
#include <armadillo>

using arma::mat;
using arma::pow;
using arma::sum;

GDLinearSolver::GDLinearSolver(const arma::mat &L,
                               const double lr,
                               const int max_iter,
                               const double termination_threshold)
    : GDSolver(L, lr, max_iter, termination_threshold){};

mat GDLinearSolver::f_objective(mat estimate, mat expected)
{
    mat residual = estimate - expected;
    return sqrt(sum(pow(residual, 2), 1) / residual.n_elem);
}

mat GDLinearSolver::f_model(mat x, mat L)
{
    return x * L;
}