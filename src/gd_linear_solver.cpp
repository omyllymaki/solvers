#include "gd_linear_solver.h"
#include <armadillo>

using arma::mat;
using arma::pow;
using arma::sum;

GDLinearSolver::GDLinearSolver(const arma::mat &L,
                               const double lr,
                               const int max_iter)
    : GDSolver(L, lr, max_iter, 1)
{
    m_lr_max = lr;
};

mat GDLinearSolver::f_objective(mat estimate, mat expected)
{
    // Mean absolute error
    mat residual = estimate - expected;
    return arma::sum(arma::abs(residual), 1) / residual.n_elem;
}

void GDLinearSolver::update_learning_rate()
{
    // Value changes from 0 -> m_lr_max -> 0 during iteration
    m_lr = m_lr_max * sin(m_round * M_PI / (m_max_iter));
}

bool GDLinearSolver::is_termination_condition_filled()
{
    return false;
}