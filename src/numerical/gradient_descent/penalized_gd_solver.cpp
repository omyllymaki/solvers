#include "penalized_gd_solver.h"
#include "../../common.h"
#include "../../logging/easylogging++.h"
#include <armadillo>

using arma::mat;
using arma::pow;
using arma::sum;

PenalizedGDSolver::PenalizedGDSolver(const arma::mat &L,
                                     const double lr,
                                     const int max_iter,
                                     const double termination_threshold,
                                     const double penalty)
    : GDSolver(L, lr, max_iter, termination_threshold)
{
    m_penalty = penalty;
};

mat PenalizedGDSolver::objective(mat estimate, mat expected)
{
    arma::uvec negative_indices = arma::find(m_x < 0);
    arma::mat x_negative_only = arma::abs(m_x.cols(negative_indices));
    arma::mat penalty = m_penalty * arma::sum(x_negative_only, 1);
    arma::mat rmse_value = rmse(estimate, expected);
    arma::mat objective_value = rmse_value + penalty;
    LOG(DEBUG) << "Round " << m_round;
    LOG(DEBUG) << "Solution " << m_x;
    LOG(DEBUG) << "Penalty " << penalty;
    LOG(DEBUG) << "RMSE " << rmse_value;
    LOG(DEBUG) << "Objective " << objective_value;
    return objective_value;
}