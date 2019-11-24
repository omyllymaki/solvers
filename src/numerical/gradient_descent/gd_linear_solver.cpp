#include "gd_linear_solver.h"
#include "../../common.h"
#include "../../logging/easylogging++.h"
#include <armadillo>

using arma::mat;
using arma::pow;
using arma::sum;

GDLinearSolver::GDLinearSolver(const arma::mat &L,
                               const double lr,
                               const int max_iter)
    : GDSolver(L, lr, max_iter)
{
    m_lr_max = lr;
};

mat GDLinearSolver::objective(mat estimate, mat expected)
{
    arma::uvec indices = arma::find(m_x < 0);
    arma::mat d = arma::abs(m_x.cols(indices));
    arma::mat penalty = 0.0005*arma::sum(d, 1) * m_round;
    std::cout << "Round " << m_round << std::endl;
    std::cout << "Solution " << m_x << std::endl;
    std::cout << "Penalty " << penalty << std::endl;
    std::cout << "RMSE " << rmse(estimate, expected) << std::endl;
    return rmse(estimate, expected) + penalty;
}

// void GDLinearSolver::update_learning_rate()
// {
//     // Value changes from 0 -> m_lr_max -> 0 during iteration
//     m_lr = m_lr_max * sin(m_round * M_PI / (m_max_iter));
// }

// bool GDLinearSolver::is_termination_condition_filled()
// {
//     return false;
// }