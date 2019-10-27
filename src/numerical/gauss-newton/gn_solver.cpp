#include "gn_solver.h"
#include "../../common.h"
#include "../../logging/easylogging++.h"
#include <iostream>

using arma::mat;
using arma::randn;

GNSolver::GNSolver(const arma::mat &L,
                   const int max_iter,
                   const double termination_threshold)
{
    m_L = L;
    m_max_iter = max_iter;
    m_termination_threshold = termination_threshold;
}

arma::mat GNSolver::solve(const arma::mat &s)
{
    m_objective_prev = {std::pow(10, 16)};
    initialize_solution();
    m_s = s;

    arma::mat s_estimate, residual;

    for (m_round = 0; m_round < m_max_iter; m_round++)
    {
        update_residual();
        update_jacobian();
        update_solution();
        update_objective();

        if (is_termination_condition_filled())
        {
            LOG(INFO) << "Change in objective value smaller than specified threshold";
            LOG(INFO) << "Iteration terminated at round " << m_round;
            return m_x;
        }

        m_objective_prev = m_objective;
    }

    return m_x;
}

arma::mat GNSolver::objective(arma::mat estimate, arma::mat expected)
{
    return rmse(estimate, expected);
}

void GNSolver::update_solution()
{
    arma::mat inv_jacobian = calculate_svd_inverse(m_jacobian.t());
    arma::mat step = inv_jacobian * m_residual.t();
    m_x = m_x - step.t();
}

void GNSolver::update_jacobian()
{
    m_jacobian.reset();
    arma::mat x, s_estimate, residual, derivative;
    for (size_t i = 0; i < m_x.n_elem; i++)
    {
        x = m_x;
        x[i] = x[i] + m_x_delta;
        s_estimate = model(x, m_L);
        residual = s_estimate - m_s;
        derivative = (residual - m_residual) / m_x_delta;
        m_jacobian.insert_rows(i, derivative);
    }
}

bool GNSolver::is_termination_condition_filled()
{
    mat rel_obj_change = (m_objective_prev - m_objective) / m_objective;
    if (as_scalar(rel_obj_change) < m_termination_threshold)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void GNSolver::update_objective()
{
    arma::mat s_estimate = model(m_x, m_L);
    m_objective = objective(s_estimate, m_s);
}

void GNSolver::update_residual()
{
    arma::mat s_estimate = model(m_x, m_L);
    m_residual = s_estimate - m_s;
}