#include "gn_solver.h"
#include "../common.h"
#include <iostream>

using arma::mat;
using arma::randn;
using std::cout;
using std::endl;

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
    m_x = randn(1, m_L.n_rows);
    m_s = s;

    arma::mat s_estimate, residual;

    for (m_round = 0; m_round < m_max_iter; m_round++)
    {
        update_residual();
        update_gradient();
        update_solution();
        update_objective();

        if (is_termination_condition_filled())
        {
            cout << "Change in objective value smaller than specified threshold" << endl;
            cout << "Iteration terminated at round " << m_round << endl;
            return m_x;
        }

        m_objective_prev = m_objective;
    }

    return m_x;
}

arma::mat GNSolver::model(mat x, mat L)
{
    return x * L;
}

arma::mat GNSolver::objective(arma::mat estimate, arma::mat expected)
{
    return rmse(estimate, expected);
}

void GNSolver::update_solution()
{
    arma::mat inv_gradient = calculate_svd_inverse(m_gradient.t());
    arma::mat step = inv_gradient * m_residual.t();
    m_x = m_x - step.t();
}

void GNSolver::update_gradient()
{
    m_gradient.reset();
    arma::mat x, s_estimate, residual, derivative;
    for (size_t i = 0; i < m_x.n_elem; i++)
    {
        x = m_x;
        x[i] = x[i] + m_x_delta;
        s_estimate = model(x, m_L);
        residual = s_estimate - m_s;
        derivative = (residual - m_residual) / m_x_delta;
        m_gradient.insert_rows(i, derivative);
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